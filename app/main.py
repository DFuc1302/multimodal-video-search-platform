"""
Frame Search Backend — Large Scale Edition
==========================================
main.py — FastAPI app. Search, status, backup, and serving routes. All
indexing/scanning logic now lives in ocr_index.py (images + OCR) and
asr_index.py (videos + speech) — this file just orchestrates them and
exposes the API, unchanged from the original app.py.

Dependencies:
    pip install fastapi uvicorn ultralytics transformers torch torchvision
                pillow numpy tqdm faiss-cpu paddlepaddle==3.0.0 paddleocr==2.9.1
                openai-whisper

Run:
    uvicorn main:app --port 8080
"""

import json
import os
from redis_store import REDIS_TEXT_ENABLED, ENABLED as REDIS_ENABLED, store as redis_store
import re
import sqlite3
from redis.exceptions import RedisError
import tempfile
import threading
import time
import webbrowser
from contextlib import asynccontextmanager, closing
from bisect import bisect_left
from collections import OrderedDict
from pathlib import Path
from typing import Optional
from query_hints import keyword_hints
from asset_paths import relative_asset_path

from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, Response

from shared import (
    PROJECT_ROOT, DATA_ROOT, KEYFRAMES_DIR, ASR_CACHE_FILE, FAISS_DIR, META_DB, VIDEOS_DIR,
    IMAGE_EXTS, VIDEO_EXTS, TOP_K_MAX,
    YOLO_MODEL_NAME, YOLO_IMGSZ, CLIP_MODEL_NAME, BATCH_SIZE, WHISPER_MODEL_NAME,
    log, get_db, init_db, get_collection, get_frame_meta, get_all_meta,
    get_meta_by_files, query_frame_files,
    CATEGORY_MAP, ALL_CATEGORIES, COLOR_ALIASES,
    embed_texts, embed_text_groups, expand_query, normalize_text, normalize_ascii_text, join_ocr_texts,
    load_image, warm_meta_cache, cache_update_ocr, unload_ocr, unload_whisper,
    _index_progress, _index_lock,
)
import shared as _shared
from server_data_health import ALLOW_PARTIAL_DATA, record_issue, scan_dataset, snapshot, EmptyCollection
from asr_index import (
    load_asr_cache,
    transcribe_video,
    find_phrase_segments,
    asr_cache_key,
    asr_cache_video_name,
    is_asr_excluded_video,
    is_current_asr_segments,
    normalize_asr_text,
)
from vlm_reranker import from_environment
_vlm_reranker = from_environment()

from kisc import start_session, get_session, end_session, process_turn

FEDERATED_DATASET_CONFIG = os.getenv("FEDERATED_DATASET_CONFIG", "").strip()
FEDERATED_ALLOW_OFFLINE = os.getenv("FEDERATED_ALLOW_OFFLINE", "1").lower() in {
    "1", "true", "yes"
}
_federation = None
_external_assets = None


def _federated_runtime():
    global _federation
    if not FEDERATED_DATASET_CONFIG:
        return None
    if _federation is None:
        from federated_runtime import FederatedRuntime
        _federation = FederatedRuntime(
            FEDERATED_DATASET_CONFIG,
            allow_offline=FEDERATED_ALLOW_OFFLINE,
        )
    return _federation


def _external_asset_resolver():
    global _external_assets
    if _federated_runtime():
        return None
    if _external_assets is None:
        from external_assets import ExternalAssetResolver
        _external_assets = ExternalAssetResolver.load(DATA_ROOT)
    return _external_assets


_single_get_collection = get_collection
_single_get_frame_meta = get_frame_meta
_single_get_meta_by_files = get_meta_by_files
_single_get_all_meta = get_all_meta
_single_query_frame_files = query_frame_files
_single_load_asr_cache = load_asr_cache


def get_collection():
    try:
        runtime = _federated_runtime()
        return runtime.collection if runtime else _single_get_collection()
    except Exception as exc:
        if not ALLOW_PARTIAL_DATA:
            raise
        record_issue('vectors', str(exc), path=FAISS_DIR)
        return EmptyCollection()


def get_frame_meta(filename: str):
    runtime = _federated_runtime()
    return runtime.get_frame_meta(filename) if runtime else _single_get_frame_meta(filename)


def get_meta_by_files(files: list):
    runtime = _federated_runtime()
    return runtime.get_meta_by_files(files) if runtime else _single_get_meta_by_files(files)


def get_all_meta():
    runtime = _federated_runtime()
    return runtime.get_all_meta() if runtime else _single_get_all_meta()


def query_frame_files(**kwargs):
    runtime = _federated_runtime()
    return runtime.query_frame_files(**kwargs) if runtime else _single_query_frame_files(**kwargs)


def load_asr_cache():
    runtime = _federated_runtime()
    if not runtime:
        cache = _single_load_asr_cache()
        if not cache:
            record_issue('asr', 'No external ASR available; audio search returns no matches', path=ASR_CACHE_FILE)
        for key, segments in cache.items():
            if not is_current_asr_segments(segments):
                record_issue('asr', 'Incomplete or unsupported transcript', path=key)
        return cache
    combined = {}
    for reader, cache in runtime.asr_caches():
        for raw_key, segments in cache.items():
            video_name = asr_cache_video_name(raw_key)
            if is_asr_excluded_video(video_name):
                continue
            key = f"video-name-v1::{video_name}"
            if key in combined:
                # Same priority rule as visual data: the first configured
                # storage owns a duplicate video.
                continue
            combined[key] = segments
    return combined


def _asr_cache_files() -> list[Path]:
    if REDIS_TEXT_ENABLED:
        return []
    runtime = _federated_runtime()
    if runtime:
        return [
            reader.root / "asr_cache.json"
            for reader in runtime.readers
            if (reader.root / "asr_cache.json").is_file()
        ]
    return [ASR_CACHE_FILE] if ASR_CACHE_FILE.is_file() else []


# KIS imported single-root helpers at module import time. Point its read-only
# search hooks at the same federated wrappers without loading another FAISS set.
import kisc as _kisc
_kisc.get_collection = get_collection
_kisc.get_meta_by_files = get_meta_by_files

# ══════════════════════════════════════════════════════════════════════════════
# App startup
# ══════════════════════════════════════════════════════════════════════════════

_readiness = {
    "status": "starting",
    "phase": "not_started",
    "ready": False,
    "error": None,
    "frames": 0,
    "timings_ms": {},
}
SERVE_SCHEMA_VERSION = 2
SEARCH_ASR_BOOST = os.getenv("SEARCH_ASR_BOOST", "0").lower() in {"1", "true", "yes"}
ALLOW_HEAVY_RUNTIME_JOBS = os.getenv(
    "ALLOW_HEAVY_RUNTIME_JOBS", "0" if _shared.SERVE_READ_ONLY else "1"
).lower() in {"1", "true", "yes"}

def _load_serve_manifest() -> dict:
    path = DATA_ROOT / "serve_manifest.json"
    required = os.getenv("REQUIRE_SERVE_BUNDLE", "0").lower() in {"1", "true", "yes"}
    if not path.is_file():
        if required:
            raise RuntimeError(
                f"Missing {path}; run prepare_serve_bundle.py before starting the server"
            )
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    required_fields = {
        "format_version",
        "frame_count",
        "faiss_count",
        "faiss_dimension",
        "faiss_ids_verified",
        "pending_ocr",
        "processing_ocr",
        "ocr_errors",
        "ocr_cached_count",
        "clip_model",
        "clip_text_artifact",
        "faiss_mapping",
        "facet_index",
        "frame_locator",
        "fts_probe",
    }
    missing = sorted(required_fields - payload.keys())
    if missing:
        raise RuntimeError(
            "Serving manifest is missing required fields: " + ", ".join(missing)
        )
    if int(payload["format_version"]) != SERVE_SCHEMA_VERSION:
        raise RuntimeError(
            f"Unsupported serving schema {payload['format_version']}; "
            f"expected {SERVE_SCHEMA_VERSION}"
        )
    if (
        payload["faiss_mapping"] != "meta.db:frames.faiss_id"
        or payload["facet_index"] != "meta.db:frame_facets"
        or payload["frame_locator"] != "meta.db:frame_locator"
    ):
        raise RuntimeError("Serving manifest uses unsupported SQLite mappings")
    if payload["clip_text_artifact"] != "clip_text":
        raise RuntimeError("Serving manifest has no compatible local CLIP text artifact")
    if payload["faiss_ids_verified"] is not True:
        raise RuntimeError("Serving manifest has not verified the FAISS/SQLite id mapping")
    if int(payload["frame_count"]) != int(payload["faiss_count"]):
        raise RuntimeError("Serving manifest SQLite/FAISS counts differ")
    if int(payload["faiss_dimension"]) <= 0:
        raise RuntimeError("Serving manifest has an invalid FAISS dimension")
    if int(payload["ocr_cached_count"]) > 0 and not payload.get("fts_probe"):
        raise RuntimeError("Serving manifest has OCR text but no real FTS probe")
    if (payload.get("pending_ocr", 0) or payload.get("processing_ocr", 0)
            or payload.get("ocr_errors", 0)):
        if ALLOW_PARTIAL_DATA:
            record_issue('ocr', 'Serving completed OCR while other rows are pending or failed', path=path,
                         count=sum(int(payload.get(k, 0)) for k in ('pending_ocr', 'processing_ocr', 'ocr_errors')))
            return payload
        raise RuntimeError(
            "Serving bundle is incomplete: "
            f"pending OCR={payload.get('pending_ocr', 0)}, "
            f"processing OCR={payload.get('processing_ocr', 0)}, "
            f"OCR errors={payload.get('ocr_errors', 0)}"
        )
    return payload

@asynccontextmanager
async def partial_lifespan(app: FastAPI):
    """Start independent search components without rejecting unfinished rows."""
    global REDIS_TEXT_ENABLED
    _readiness.update(status='starting', ready=False, phase='database', capabilities={})
    capabilities = _readiness['capabilities']
    if os.getenv('SERVER_REDIS_FALLBACK') == '1':
        record_issue('redis', 'Redis is unavailable or incomplete; serving local SQLite/FAISS')
    try:
        with get_db() as conn:
            tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            capabilities['metadata'] = 'frames' in tables
            _shared.FTS5_AVAILABLE = 'frames_fts' in tables
            if not capabilities['metadata']:
                raise RuntimeError('Missing frames table')
            _readiness['metadata_frames'] = conn.execute('SELECT COUNT(*) FROM frames').fetchone()[0]
            for table in {'frames_fts', 'frame_facets', 'frame_locator', 'serve_meta'} - tables:
                record_issue('database', 'Optional serving table missing', path=table)
        capabilities['ocr'] = True
    except (OSError, sqlite3.Error, RuntimeError) as exc:
        capabilities.update(metadata=False, ocr=False)
        record_issue('database', str(exc), path=META_DB)
    if REDIS_TEXT_ENABLED:
        try:
            from redis_serving import text_startup
            text_startup()
        except Exception as exc:
            record_issue('redis', 'Using local text indexes: ' + str(exc))
            REDIS_TEXT_ENABLED = False
            import asr_index
            asr_index.REDIS_TEXT_ENABLED = False
            import search_pages
            search_pages.REDIS_TEXT_ENABLED = False
    try:
        manifest = _load_serve_manifest()
        if manifest and manifest.get('clip_model') not in {None, CLIP_MODEL_NAME}:
            raise RuntimeError('CLIP model differs from manifest')
        collection = get_collection()
        _readiness['frames'] = collection.count()
        if not _readiness['frames']:
            raise RuntimeError('No available vectors; OCR and metadata search remain available')
        if os.getenv('PRELOAD_CLIP_ON_STARTUP', '1').lower() in {'1', 'true', 'yes'}:
            vector = embed_texts(['server readiness probe'])
            if collection.dimension() and vector.shape[-1] != collection.dimension():
                raise RuntimeError('CLIP/FAISS dimension mismatch')
            collection.query([vector.tolist()], n_results=1)
        capabilities['visual'] = True
    except Exception as exc:
        capabilities['visual'] = False
        record_issue('visual', str(exc), path=FAISS_DIR)
    # The full inventory checks missing files without delaying HTTP availability.
    threading.Thread(target=scan_dataset, args=(DATA_ROOT, KEYFRAMES_DIR, VIDEOS_DIR),
                     name='dataset-health', daemon=True).start()
    _readiness.update(status='ready', ready=True, phase='ready',
                      degraded=not all(capabilities.values()),
                      backend='hybrid' if REDIS_TEXT_ENABLED else 'sqlite')
    log.info('Server ready with available data. Diagnostics: %s', snapshot()['log_file'])
    yield


@asynccontextmanager
async def lifespan(app: FastAPI):
    if ALLOW_PARTIAL_DATA and not REDIS_ENABLED:
        async with partial_lifespan(app):
            yield
        return
    if REDIS_ENABLED:
        from redis_serving import startup
        startup(_readiness)
        log.info("Server ready: http://localhost:%s", os.getenv("SERVER_PORT", "8080"))
        yield
        return
    total_started = time.perf_counter()
    _readiness.update(
        status="starting", phase="database", ready=False, error=None, timings_ms={}
    )
    try:
        federation = _federated_runtime()
        step = time.perf_counter()
        if federation:
            log.info(
                "=== Startup: opening %d federated dataset(s) ===",
                len(federation.readers),
            )
        else:
            log.info("=== Startup: opening serving database ===")
            init_db()
        if REDIS_TEXT_ENABLED:
            from redis_serving import text_startup
            text_startup()
        _readiness["timings_ms"]["database"] = round(
            (time.perf_counter() - step) * 1000, 1
        )

        auto_index = os.getenv("AUTO_INDEX_ON_STARTUP", "0").lower() in {"1", "true", "yes"}
        if auto_index and _shared.SERVE_READ_ONLY:
            raise RuntimeError("AUTO_INDEX_ON_STARTUP cannot be used in read-only serving mode")
        if auto_index:
            from ocr_index import build_index, backfill_detections, backfill_ocr
            log.info("=== Startup: indexing keyframes ===")
            build_index()
            backfill_detections()
            backfill_ocr()

        _readiness["phase"] = "manifest"
        manifest = (
            {
                "clip_model": federation.clip_model,
                "frame_count": federation.collection.count(),
                "faiss_count": federation.collection.count(),
                "faiss_dimension": federation.dimension,
                "fts_probe": None,
            }
            if federation else _load_serve_manifest()
        )
        if manifest and manifest.get("clip_model") not in {None, CLIP_MODEL_NAME}:
            raise RuntimeError(
                f"Manifest/CLIP model mismatch: "
                f"{manifest.get('clip_model')} != {CLIP_MODEL_NAME}"
            )

        _readiness["phase"] = "metadata"
        step = time.perf_counter()
        if not federation:
            warm_meta_cache()
        _readiness["timings_ms"]["metadata"] = round(
            (time.perf_counter() - step) * 1000, 1
        )

        _readiness["phase"] = "faiss"
        step = time.perf_counter()
        collection = get_collection()
        frame_count = collection.count()
        if frame_count <= 0:
            raise RuntimeError("FAISS index is empty")
        if manifest and int(manifest.get("frame_count", -1)) != frame_count:
            raise RuntimeError(
                f"Manifest/FAISS count mismatch: "
                f"{manifest.get('frame_count')} != {frame_count}"
            )
        if manifest and int(manifest.get("faiss_count", -1)) != frame_count:
            raise RuntimeError(
                f"Manifest/FAISS count mismatch: "
                f"{manifest.get('faiss_count')} != {frame_count}"
            )
        _readiness["frames"] = frame_count
        _readiness["timings_ms"]["faiss"] = round(
            (time.perf_counter() - step) * 1000, 1
        )

        preload = os.getenv("PRELOAD_CLIP_ON_STARTUP", "1").lower() in {"1", "true", "yes"}
        _readiness["phase"] = "clip_and_search_warmup"
        step = time.perf_counter()
        if preload:
            log.info("Loading CLIP text encoder and running search readiness probe...")
            probe_vector = embed_texts(["server readiness probe"])
            faiss_dimension = collection.dimension()
            if faiss_dimension and int(probe_vector.shape[-1]) != faiss_dimension:
                raise RuntimeError(
                    f"CLIP/FAISS dimension mismatch: {probe_vector.shape[-1]} != {faiss_dimension}"
                )
            probe = collection.query([probe_vector.tolist()], n_results=1)
            probe_meta = probe.get("metadatas", [[]])[0]
            if not probe_meta:
                raise RuntimeError("FAISS mapping probe returned no frame metadata")
            if not get_meta_by_files([probe_meta[0]["file"]]):
                raise RuntimeError("SQLite metadata probe could not resolve FAISS result")
            log.info("CLIP text encoder ready; search readiness probe passed.")
        else:
            log.info("CLIP text model loading deferred until the first visual query.")
        if not federation:
            with get_db() as conn:
                tables = {
                    row[0]
                    for row in conn.execute(
                        "SELECT name FROM sqlite_master WHERE type IN ('table','view')"
                    )
                }
                missing_tables = {
                    "frames_fts", "frame_facets", "frame_locator", "serve_meta"
                } - tables
                if missing_tables:
                    raise RuntimeError(
                        f"Serving database is missing: {', '.join(sorted(missing_tables))}"
                    )
                serve_meta = dict(conn.execute("SELECT key, value FROM serve_meta"))
                if int(serve_meta.get("schema_version", -1)) != SERVE_SCHEMA_VERSION:
                    raise RuntimeError("SQLite serve_meta schema does not match the manifest")
                if int(serve_meta.get("frame_count", -1)) != frame_count:
                    raise RuntimeError("SQLite serve_meta frame count does not match FAISS")
                fts_probe = manifest.get("fts_probe") if manifest else None
                if fts_probe:
                    term = str(fts_probe.get("term", "")).replace('"', '""')
                    fts_row = conn.execute(
                        "SELECT file FROM frames_fts "
                        "WHERE frames_fts MATCH ? AND rowid=? LIMIT 1",
                        (f'"{term}"', int(fts_probe["rowid"])),
                    ).fetchone()
                    if not fts_row or fts_row["file"] != fts_probe.get("file"):
                        raise RuntimeError("Real FTS MATCH readiness probe failed")
                conn.execute("SELECT file FROM frame_facets LIMIT 1").fetchone()
                conn.execute("SELECT file FROM frame_locator LIMIT 1").fetchone()
        _readiness["timings_ms"]["warmup"] = round(
            (time.perf_counter() - step) * 1000, 1
        )

        _readiness.update(
            status="ready",
            phase="ready",
            ready=True,
            backend="hybrid" if REDIS_TEXT_ENABLED else "sqlite",
            total_ms=round((time.perf_counter() - total_started) * 1000, 1),
        )
        log.info(
            "=== Startup complete — %d frames search-ready in %.1f ms ===",
            frame_count,
            _readiness["total_ms"],
        )
        server_url = f"http://localhost:{os.getenv('SERVER_PORT', '8080')}"
        log.info("Server ready: %s", server_url)
        if os.getenv("OPEN_BROWSER", "0").lower() in {"1", "true", "yes"}:
            threading.Thread(
                target=lambda: (time.sleep(1.0), webbrowser.open(server_url)),
                daemon=True,
            ).start()
        yield
    except Exception as exc:
        _readiness.update(
            status="failed",
            phase="failed",
            ready=False,
            error=str(exc),
            total_ms=round((time.perf_counter() - total_started) * 1000, 1),
        )
        log.exception("Server readiness failed: %s", exc)
        raise

from search_telemetry import LoggedSearchRoute, router as evaluation_router

app = FastAPI(title="Frame Search", lifespan=lifespan)


@app.exception_handler(sqlite3.DatabaseError)
@app.exception_handler(OSError)
@app.exception_handler(RedisError)
async def unavailable_data(request, exc):
    record_issue('request.data', str(exc), path=request.url.path)
    if ALLOW_PARTIAL_DATA and ('search' in request.url.path or request.url.path == '/video-match'):
        return JSONResponse({'results': [], 'total': 0, 'partial': True,
                             'warnings': ['Requested data is unavailable; see sever_log.json']})
    return JSONResponse({'error': 'Requested data unavailable', 'log_file': 'sever_log.json'}, status_code=503)


@app.get('/data-health')
def data_health():
    return {**snapshot(), 'capabilities': _readiness.get('capabilities', {})}
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
app.mount("/static", StaticFiles(directory=REPOSITORY_ROOT / "static"), name="static")
app.router.route_class = LoggedSearchRoute
app.include_router(evaluation_router)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], expose_headers=["X-Search-Id", "X-Query-Id", "X-Search-Session"])

# Disable OCR model loading during normal search/lightbox by default.
# Search still uses already-indexed OCR text through SQLite/FTS.
# Set OCR_ON_DEMAND=1 only when you intentionally want server-side live OCR.
OCR_ON_DEMAND = os.getenv("OCR_ON_DEMAND", "0").lower() in {"1", "true", "yes"}

# Dedicated Video Match tuning. This route intentionally bypasses OCR/ASR and
# returns grouped, lightweight results so the UI does not wait on work that is
# irrelevant to visual video matching.
VIDEO_MATCH_CANDIDATES = max(100, int(os.getenv("VIDEO_MATCH_CANDIDATES", "1200")))
VIDEO_MATCH_TOP_VIDEOS = max(1, int(os.getenv("VIDEO_MATCH_TOP_VIDEOS", "50")))
VIDEO_MATCH_FRAMES_PER_VIDEO = max(1, int(os.getenv("VIDEO_MATCH_FRAMES_PER_VIDEO", "4")))
VIDEO_MATCH_MIN_FRAME_GAP = max(0, int(os.getenv("VIDEO_MATCH_MIN_FRAME_GAP", "6")))
_VIDEO_ID_RE = re.compile(r"^[LSN]\d+[-_]V\d+$", re.IGNORECASE)

# ══════════════════════════════════════════════════════════════════════════════
# Frame index mapping (AIC-style CSVs: n, pts_time, fps, frame_idx)
# Place one CSV per video in map-keyframes/, named exactly like the video,
# e.g. map-keyframes/L20_V001.csv
# ══════════════════════════════════════════════════════════════════════════════

MAPFILES_DIR = Path(os.getenv("MAPFILES_DIR", str(DATA_ROOT / "map-keyframes")))
_frame_idx_cache: dict = {}   # video_name -> {n (int) -> frame_idx (int)}


_RESOLVED_VIDEO_SOURCE: dict[str, Path] = {}

def _find_asset_in_candidate_roots(sub_dir: str, filename: str) -> Path | None:
    filename = relative_asset_path(filename)
    if filename is None:
        return None
    parts = filename.replace("\\", "/").split("/")
    video = parts[0]
    
    # Check cache first
    cached_root = _RESOLVED_VIDEO_SOURCE.get(video.casefold())
    if cached_root is not None:
        p = cached_root / sub_dir / filename
        if p.is_file():
            return p
        if len(parts) == 2 and parts[1].startswith(parts[0] + "_"):
            short = parts[1][len(parts[0]) + 1:]
            alt_p = cached_root / sub_dir / parts[0] / short
            if alt_p.is_file():
                return alt_p

    # Search all candidate roots
    search_dirs = [
        Path("/mnt/d/AIC/data/processed"),
        Path("/mnt/d/AIC/data"),
        DATA_ROOT,
        PROJECT_ROOT / "data",
    ]
    candidate_roots = []
    for sdir in search_dirs:
        if sdir.is_dir():
            try:
                candidate_roots.extend([p for p in sdir.iterdir() if p.is_dir()])
            except OSError:
                pass
            candidate_roots.append(sdir)

    for root in candidate_roots:
        p = root / sub_dir / filename
        if p.is_file():
            _RESOLVED_VIDEO_SOURCE[video.casefold()] = root
            return p
        p = root / filename
        if p.is_file():
            _RESOLVED_VIDEO_SOURCE[video.casefold()] = root
            return p
        if len(parts) == 2 and parts[1].startswith(parts[0] + "_"):
            short = parts[1][len(parts[0]) + 1:]
            alt_p = root / sub_dir / parts[0] / short
            if alt_p.is_file():
                _RESOLVED_VIDEO_SOURCE[video.casefold()] = root
                return alt_p
    return None

def _keyframe_path(filename: str) -> Path | None:
    filename = relative_asset_path(filename)
    if filename is None:
        return None
    federation = _federated_runtime()
    if federation:
        p = federation.resolve_image(filename)
        if p is not None and p.is_file():
            return p

    external = _external_asset_resolver()
    if external:
        p = external.resolve_image(filename)
        if p is not None and p.is_file():
            return p

    p = KEYFRAMES_DIR / filename
    if p.is_file():
        return p

    # Fallback to smart dynamic search across external shards
    return _find_asset_in_candidate_roots("keyframes", filename)

def load_frame_idx_map(video: str) -> dict:
    if relative_asset_path(video) is None or '/' in video.replace('\\', '/'):
        return {}
    if REDIS_ENABLED:
        return {item['n']: item['frame_idx'] for item in redis_store().timeline(video)}
    if video in _frame_idx_cache:
        return _frame_idx_cache[video]

    csv_path = None
    federation = _federated_runtime()
    if federation:
        csv_path = federation.resolve_map(video)
        if csv_path and not csv_path.is_file():
            csv_path = None

    if csv_path is None:
        external = _external_asset_resolver()
        if external:
            csv_path = external.resolve_map(video)
            if csv_path and not csv_path.is_file():
                csv_path = None

    if csv_path is None or not csv_path.is_file():
        p = MAPFILES_DIR / f"{video}.csv"
        if p.is_file():
            csv_path = p
        else:
            csv_path = _find_asset_in_candidate_roots("map-keyframes", f"{video}.csv")

    mapping = {}
    if csv_path is not None and csv_path.exists():
        import csv as csv_module
        try:
            with open(csv_path, newline="", encoding="utf-8") as f:
                reader = csv_module.DictReader(f)
                for row in reader:
                    try:
                        n = int(float(row["n"]))
                        mapping[n] = int(float(row["frame_idx"]))
                    except (KeyError, ValueError):
                        continue
        except Exception as exc:
            log.warning(f"Could not load CSV map for {video}: {exc}")

    _frame_idx_cache[video] = mapping
    return mapping

@app.get("/frame-idx")
def get_frame_idx(video: str = Query(...), n: int = Query(...)):
    """Resolve n (keyframe filename number) -> real frame_idx via map-keyframes/{video}.csv."""
    mapping = load_frame_idx_map(video)
    if not mapping:
        return {"video": video, "n": n, "frame_idx": n, "mapped": False}
    return {"video": video, "n": n, "frame_idx": mapping.get(n, n), "mapped": n in mapping}

# ══════════════════════════════════════════════════════════════════════════════
# Routes
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/")
def serve_ui():
    p = REPOSITORY_ROOT / "index.html"
    return FileResponse(str(p), media_type="text/html") if p.exists() \
        else JSONResponse({"error": "index.html not found"}, status_code=404)

@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    # Avoid a noisy 404 in the terminal when no custom favicon is bundled.
    return Response(status_code=204)

@app.get("/health")
def health():
    import torch
    return {
        "status": "ok",
        "ready": bool(_readiness["ready"]),
        "frames": int(_readiness.get("frames", 0)),
        "gpu": torch.cuda.is_available(),
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "profile": {
            "yolo_model": YOLO_MODEL_NAME,
            "yolo_imgsz": YOLO_IMGSZ,
            "clip_model": CLIP_MODEL_NAME,
            "clip_mode": "text-only" if _shared.CLIP_TEXT_ONLY else "full",
            "clip_batch_size": BATCH_SIZE,
            "whisper_model": WHISPER_MODEL_NAME,
        },
        "models": {
            "yolo": _shared._yolo_model is not None,
            "clip_image": _shared._clip_model is not None,
            "clip_text": _shared._clip_text_model is not None,
            "ocr": _shared._ocr_model is not None,
            "whisper": _shared._whisper_model is not None,
        },
    }

@app.get("/ready")
def ready():
    payload = dict(_readiness)
    return JSONResponse(payload, status_code=200 if payload["ready"] else 503)

# ── on-demand indexing for new frame batches ──────────────────────────────────

def _run_indexing_job():
    try:
        if _shared.SERVE_READ_ONLY:
            raise RuntimeError("Runtime indexing is disabled in read-only serving mode")
        from ocr_index import build_index, backfill_detections, backfill_ocr
        build_index()
        backfill_detections()
        backfill_ocr()
        warm_meta_cache(force=True)
    except Exception as e:
        log.error(f"Indexing job failed: {e}")
        with _index_lock:
            _index_progress.update(running=False, stage="idle", error=str(e))

@app.post("/index/run")
def index_run():
    """
    Trigger indexing of any new frames found in keyframes/ (e.g. after
    dropping in batch 2). Runs in a background thread — poll /index/progress
    for status. No-op if already running.
    """
    if _shared.SERVE_READ_ONLY:
        return JSONResponse({"error": "Runtime indexing is disabled"}, status_code=403)
    with _index_lock:
        if _index_progress["running"]:
            return {"status": "already_running", "progress": dict(_index_progress)}
    threading.Thread(target=_run_indexing_job, daemon=True).start()
    return {"status": "started"}

@app.get("/index/progress")
def index_progress():
    with _index_lock:
        prog = dict(_index_progress)
    prog["collection_count"] = get_collection().count()
    if prog["started_at"]:
        prog["elapsed_sec"] = round(time.time() - prog["started_at"], 1)
        if prog["done"] > 0 and prog["total"] > 0:
            rate = prog["elapsed_sec"] / prog["done"]
            prog["eta_sec"] = round(rate * (prog["total"] - prog["done"]), 1)
    return prog


def _fetch_frame_rows(files: list[str], chunk_size: int = 800) -> list:
    """SQLite has a bound-parameter limit, so large searches fetch in chunks."""
    if REDIS_ENABLED:
        return [{**r, 'ocr_texts': json.dumps(r.get('ocr_texts', []))} for r in redis_store().rows(files)]
    federation = _federated_runtime()
    if federation:
        return federation.get_meta_by_files(files, full=True)
    rows = []
    with get_db() as conn:
        for start in range(0, len(files), chunk_size):
            chunk = files[start:start + chunk_size]
            if not chunk:
                continue
            placeholders = ",".join("?" * len(chunk))
            rows.extend(conn.execute(
                f"SELECT * FROM frames WHERE file IN ({placeholders})", chunk
            ).fetchall())
    return rows

def _escape_fts_phrase(text: str) -> str:
    return text.strip().replace('"', '""')

def _ocr_match_files(text: str, limit: int) -> tuple[list[str], bool]:
    """Fast OCR lookup via FTS5; searches both original and no-accent text."""
    if REDIS_TEXT_ENABLED:
        return ([item['file'] for item, score in redis_store().text_search('frame', text, limit)], True)
    federation = _federated_runtime()
    if not text:
        return [], False
    if not federation and not _shared.FTS5_AVAILABLE:
        return _ocr_scan_fallback(text, limit) if ALLOW_PARTIAL_DATA else ([], False)

    q_norm = normalize_text(text)
    q_ascii = normalize_ascii_text(text)
    terms = [q_norm]
    if q_ascii and q_ascii != q_norm:
        terms.append(q_ascii)
    match = " OR ".join(f'"{_escape_fts_phrase(t)}"' for t in terms if t)

    if federation:
        return federation.ocr_match_files(match, limit)

    try:
        with get_db() as conn:
            rows = conn.execute(
                "SELECT file FROM frames_fts WHERE frames_fts MATCH ? ORDER BY rank LIMIT ?",
                (match, limit),
            ).fetchall()
        return [r["file"] for r in rows], True
    except sqlite3.OperationalError as exc:
        if ALLOW_PARTIAL_DATA:
            record_issue('ocr.fts', str(exc))
            return _ocr_scan_fallback(text, limit)
        return [], False


def _ocr_scan_fallback(text, limit):
    """Keep already recognized text searchable before FTS has been built."""
    try:
        with get_db() as conn:
            rows = conn.execute(
                "SELECT file FROM frames WHERE instr(lower(COALESCE(ocr_joined,'')), ?) > 0 "
                "OR instr(lower(COALESCE(ocr_ascii,'')), ?) > 0 LIMIT ?",
                (normalize_text(text), normalize_ascii_text(text) or normalize_text(text), limit)).fetchall()
        return [r['file'] for r in rows], True
    except sqlite3.Error as exc:
        record_issue('ocr.database', str(exc), path=META_DB)
        return [], True


def _video_id_from_frame(file_rel: str) -> str:
    """Resolve the source video id from a keyframe path."""
    normalized = (file_rel or "").replace("\\", "/")
    parts = [p for p in normalized.split("/") if p]

    # Prefer the deepest AIC-style folder, e.g. L21_V001. This avoids grouping
    # everything under only the batch folder (L21).
    for part in reversed(parts[:-1]):
        if _VIDEO_ID_RE.fullmatch(part):
            return part

    name = parts[-1] if parts else normalized
    match = re.search(r"([LSN]\d+[-_]V\d+)", name, re.IGNORECASE)
    if match:
        return match.group(1)

    if len(parts) > 1:
        return parts[-2]

    stem = Path(name).stem
    fallback = re.sub(r"[_-]?\d+$", "", stem)
    return fallback or stem or "UNKNOWN"


def _frame_number_from_path(file_rel: str) -> Optional[int]:
    name = Path((file_rel or "").replace("\\", "/")).name
    match = re.search(r"(\d+)(?=\.[^.]+$)", name)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _comma_query_terms(query: Optional[str]) -> list[str]:
    """Return non-empty comma-separated clauses when at least two exist."""
    if not query or "," not in query:
        return []
    terms = [term.strip() for term in query.split(",") if term.strip()]
    return terms if len(terms) >= 2 else []


def _comma_and_terms(query: Optional[str]) -> list[str]:
    """Backward-compatible helper for the default comma-AND behavior."""
    return _comma_query_terms(query)


def _intersect_ranked_scores(
    score_maps: list[dict[str, float]],
) -> tuple[list[str], dict[str, float]]:
    """Intersect ranked searches and score each file by its weakest clause."""
    if not score_maps:
        return [], {}

    common_files = set(score_maps[0])
    for scores in score_maps[1:]:
        common_files.intersection_update(scores)
        if not common_files:
            return [], {}

    combined_scores = {
        filename: min(scores[filename] for scores in score_maps)
        for filename in common_files
    }
    candidate_files = sorted(
        common_files,
        key=lambda filename: combined_scores[filename],
        reverse=True,
    )
    return candidate_files, combined_scores


def _union_ranked_scores(
    score_maps: list[dict[str, float]],
) -> tuple[list[str], dict[str, float]]:
    """Union ranked searches and keep each file's strongest clause score."""
    combined_scores: dict[str, float] = {}
    for scores in score_maps:
        for filename, score in scores.items():
            combined_scores[filename] = max(
                score,
                combined_scores.get(filename, float("-inf")),
            )
    candidate_files = sorted(
        combined_scores,
        key=lambda filename: combined_scores[filename],
        reverse=True,
    )
    return candidate_files, combined_scores

_TEXT_EMBED_CACHE: OrderedDict[str, object] = OrderedDict()
_TEXT_EMBED_CACHE_LOCK = threading.Lock()


def _embed_query_terms(terms: list[str]):
    """Reuse hot query vectors and batch all cold terms into one GPU pass."""
    import numpy as np

    keys = [term.strip() for term in terms]
    vectors: list[object] = [None] * len(keys)
    missing: list[tuple[int, str]] = []

    with _TEXT_EMBED_CACHE_LOCK:
        for index, key in enumerate(keys):
            cached = _TEXT_EMBED_CACHE.get(key)
            if cached is None:
                missing.append((index, key))
            else:
                _TEXT_EMBED_CACHE.move_to_end(key)
                vectors[index] = cached

    if missing:
        groups = [expand_query(key) for _, key in missing]
        fresh_vectors = embed_text_groups(groups)
        with _TEXT_EMBED_CACHE_LOCK:
            for (index, key), vector in zip(missing, fresh_vectors):
                _TEXT_EMBED_CACHE[key] = vector
                _TEXT_EMBED_CACHE.move_to_end(key)
                while len(_TEXT_EMBED_CACHE) > 256:
                    _TEXT_EMBED_CACHE.popitem(last=False)
                vectors[index] = vector

    return np.stack(vectors)

def _clip_and_candidates(col, terms: list[str], n_results: int):
    """Run one CLIP retrieval per clause, then keep only their intersection."""
    text_features = _embed_query_terms(terms)
    result = col.query(query_embeddings=text_features.tolist(), n_results=n_results)
    score_maps: list[dict[str, float]] = []
    for metas, distances in zip(
        result.get("metadatas", []),
        result.get("distances", []),
    ):
        scores = {
            meta["file"]: 1.0 - float(distance)
            for meta, distance in zip(metas, distances)
            if meta.get("file")
        }
        score_maps.append(scores)
    return _intersect_ranked_scores(score_maps)


def _clip_or_candidates(col, terms: list[str], n_results: int):
    """Run all CLIP clauses in one batch, then keep their ranked union."""
    text_features = _embed_query_terms(terms)
    result = col.query(query_embeddings=text_features.tolist(), n_results=n_results)
    score_maps: list[dict[str, float]] = []
    for metas, distances in zip(
        result.get("metadatas", []),
        result.get("distances", []),
    ):
        score_maps.append({
            meta["file"]: 1.0 - float(distance)
            for meta, distance in zip(metas, distances)
            if meta.get("file")
        })
    return _union_ranked_scores(score_maps)






def _dedupe_ranked_frames(frames: list[dict], min_gap: int = 3) -> list[dict]:
    """Keep the best-ranked representative among adjacent keyframes."""
    if min_gap <= 0:
        return frames
    selected: list[dict] = []
    blocked_numbers: dict[str, set[int]] = {}
    for item in frames:
        video_id = _video_id_from_frame(item["file"])
        frame_no = _frame_number_from_path(item["file"])
        if frame_no is None:
            selected.append(item)
            continue
        blocked = blocked_numbers.setdefault(video_id, set())
        if frame_no in blocked:
            continue
        blocked.update(range(frame_no - min_gap + 1, frame_no + min_gap))
        selected.append(item)
    return selected



def _select_diverse_video_frames(frames: list[dict], limit: int, min_gap: int) -> list[dict]:
    """Keep strong frames while avoiding four nearly identical neighbours."""
    ordered = sorted(frames, key=lambda item: item["_score"], reverse=True)
    if limit <= 0:
        return []
    if min_gap <= 0:
        return ordered[:limit]

    selected: list[dict] = []
    selected_numbers: list[int] = []
    selected_files: set[str] = set()

    for item in ordered:
        frame_no = _frame_number_from_path(item["file"])
        if frame_no is None or all(abs(frame_no - old) >= min_gap for old in selected_numbers):
            selected.append(item)
            selected_files.add(item["file"])
            if frame_no is not None:
                selected_numbers.append(frame_no)
        if len(selected) >= limit:
            return selected

    # If diversity filtering left too few frames, fill the remainder by score.
    for item in ordered:
        if item["file"] in selected_files:
            continue
        selected.append(item)
        if len(selected) >= limit:
            break
    return selected


def _video_rank_score(frames: list[dict]) -> float:
    """Robust video score: best frame plus support from other strong frames."""
    if not frames:
        return 0.0
    scores = sorted((float(f["_score"]) for f in frames), reverse=True)
    best = scores[0]
    top3 = sum(scores[:3]) / min(3, len(scores))
    top5 = sum(scores[:5]) / min(5, len(scores))
    return 0.65 * best + 0.25 * top3 + 0.10 * top5

# ── backup ─────────────────────────────────────────────────────────────────────

@app.post("/backup")
def create_backup():
    """
    Zip faiss_index/ + meta.db + asr_cache.json into a timestamped backup file
    under backups/. Returns the path for download.
    """
    if not ALLOW_HEAVY_RUNTIME_JOBS:
        return JSONResponse(
            {"error": "Backup is disabled in immutable serving mode"},
            status_code=403,
        )
    import shutil
    from datetime import datetime

    backups_dir = Path("backups")
    backups_dir.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out_base = backups_dir / f"index_backup_{stamp}"

    # Stage files into a temp dir to zip together
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        if FAISS_DIR.exists():
            shutil.copytree(FAISS_DIR, tmp_path / "faiss_index")
        if META_DB.exists():
            # Copy committed WAL pages too; copying only meta.db loses data.
            with closing(sqlite3.connect(META_DB.resolve().as_uri() + '?mode=ro', uri=True)) as source:
                with closing(sqlite3.connect(tmp_path / 'meta.db')) as target:
                    source.backup(target)
        if ASR_CACHE_FILE.exists():
            shutil.copy2(ASR_CACHE_FILE, tmp_path / "asr_cache.json")

        archive_path = shutil.make_archive(str(out_base), "zip", root_dir=tmp_path)

    size_mb = round(Path(archive_path).stat().st_size / 1e6, 1)
    return {"status": "ok", "file": archive_path, "size_mb": size_mb}

@app.get("/backup/list")
def list_backups():
    backups_dir = Path("backups")
    if not backups_dir.exists():
        return {"backups": []}
    files = sorted(backups_dir.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
    return {"backups": [
        {"name": f.name, "size_mb": round(f.stat().st_size/1e6, 1), "mtime": f.stat().st_mtime}
        for f in files
    ]}

@app.get("/backup/download/{filename}")
def download_backup(filename: str):
    if relative_asset_path(filename) is None or '/' in filename.replace('\\', '/'):
        return JSONResponse({"error": "Not found"}, status_code=404)
    path = Path("backups") / filename
    if not path.is_file() or path.suffix.lower() != '.zip':
        return JSONResponse({"error": "Not found"}, status_code=404)
    return FileResponse(str(path), filename=filename, media_type="application/zip")

@app.get("/status")
def status():
    if REDIS_ENABLED:
        from redis_serving import status as redis_status
        return redis_status()
    federation = _federated_runtime()
    if federation:
        combined = federation.status()
        object_counts = combined.get("objects", {})
        combined["obj_to_cat"] = {
            obj: next(
                (
                    category for category, members in CATEGORY_MAP.items()
                    if obj.casefold() in {member.casefold() for member in members}
                ),
                "Other",
            )
            for obj in object_counts
        }
        combined.update({
            "indexed": federation.collection.count(),
            "all_categories": ALL_CATEGORIES,
        })
        return combined
    stats_path = DATA_ROOT / "serve_stats.json"
    if not stats_path.is_file():
        if ALLOW_PARTIAL_DATA:
            record_issue('stats', 'Serving statistics have not been generated', path=stats_path)
            return dict(indexed=_readiness.get('metadata_frames', 0), categories={}, objects={},
                        colors={}, obj_to_cat={}, all_categories=ALL_CATEGORIES, partial=True)
        return JSONResponse(
            {"error": f"Missing {stats_path}; run prepare_serve_bundle.py"},
            status_code=503,
        )
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    object_counts = stats.get("objects", {})
    object_to_category = {
        obj: next(
            (
                category
                for category, members in CATEGORY_MAP.items()
                if obj.casefold() in {member.casefold() for member in members}
            ),
            "Other",
        )
        for obj in object_counts
    }
    return {
        "indexed": get_collection().count(),
        "categories": stats.get("categories", {}),
        "objects": object_counts,
        "obj_to_cat": object_to_category,
        "colors": stats.get("colors", {}),
        "all_categories": ALL_CATEGORIES,
        "has_text_count": stats.get("has_text_count", 0),
        "ocr_cached_count": stats.get("ocr_cached_count", 0),
        "pending_ocr": stats.get("pending_ocr", 0),
        "processing_ocr": stats.get("processing_ocr", 0),
        "ocr_errors": stats.get("ocr_errors", 0),
        "backend": "hybrid" if REDIS_TEXT_ENABLED else "sqlite",
        "visual_backend": "sqlite_faiss",
        "text_backend": "redis_local" if REDIS_TEXT_ENABLED else "sqlite_fts",
    }

_summary_engine = None
_summary_engine_lock = __import__('threading').Lock()


@app.get("/summary/search")
def search_summaries(query: str = Query(..., min_length=1, max_length=2000),
                     top_k: int = Query(60, ge=1, le=1000),
                     category: str = Query('', pattern='^(0[1-9])?$'),
                     video: str = '', mode: str = Query('summary', pattern='^(summary|combined)$'),
                     page: int = 0, page_size: int = 1000):
    global _summary_engine
    try:
        with _summary_engine_lock:
            if _summary_engine is None:
                if REDIS_TEXT_ENABLED:
                    from redis_serving import RedisSummarySearch
                    _summary_engine = RedisSummarySearch()
                else:
                    from summary_search import SummarySearch
                    _summary_engine = SummarySearch()
        result = _summary_engine.search(query, None if page > 0 else top_k, category, video)
        if mode == 'combined':
            visual = search(query=query, ocr_query=None, objects=None, categories=None,
                            colors=None, video=video or None, video_mode='include',
                            min_conf=0.0, top_k=top_k, query_operator='or',
                            ocr_operator='and', search_mode='clip', rerank=False, timeline=False,
                            page=1 if page > 0 else 0, page_size=get_collection().count(), domain=category)
            if isinstance(visual, dict):
                fused, scores = {}, {}
                for ranking in (result['results'], visual.get('results', [])):
                    for rank, item in enumerate(ranking, 1):
                        if category and not item['file'].startswith(category + '_'):
                            continue
                        key = item['file']
                        fused.setdefault(key, item)
                        scores[key] = scores.get(key, 0) + 1 / (60 + rank)
                result['results'] = [{**fused[key], 'score': scores[key]} for key in sorted(scores, key=lambda key: (-scores[key], key))[:None if page > 0 else top_k]]
                result['total'] = len(result['results'])
            else:
                result['warning'] = 'Visual search unavailable; summary results only'
        if page > 0:
            page_size = max(1, page_size)
            result['total'] = len(result['results'])
            page = min(page, max(1, (result['total'] + page_size - 1) // page_size))
            result['results'] = result['results'][(page-1)*page_size:page*page_size]
            result.update(page=page, page_size=page_size)
        for item in result['results']:
            item.setdefault('detections', [])
            item.setdefault('ocr_texts', [])
        return result
    except (FileNotFoundError, OSError) as exc:
        log.warning('Summary index unavailable: %s', exc)
        if ALLOW_PARTIAL_DATA:
            record_issue('summary', str(exc))
            return {'results': [], 'total': 0, 'partial': True,
                    'warnings': ['Summary data unavailable; see sever_log.json']}
        return JSONResponse({'error': 'Summary index/model unavailable. Run summary_search.py --build.'}, status_code=503)


@app.get("/search")
def search(
    query:      Optional[str] = Query(None),
    ocr_query:  Optional[str] = Query(None),
    objects:    Optional[str] = Query(None),
    categories: Optional[str] = Query(None),
    colors:     Optional[str] = Query(None),
    video:      Optional[str] = Query(None, description="Filter to frames from this video (matches folder name)"),
    video_mode: str           = Query("include", description="include | exclude"),
    min_conf:   float         = Query(0.0, ge=0.0, le=1.0),
    top_k:      int           = Query(1000, ge=1, le=TOP_K_MAX),
    query_operator: str       = Query("and", description="and | or"),
    ocr_operator: str         = Query("and", description="and | or"),
    search_mode: str          = Query("hybrid", description="hybrid | ocr_first | clip"),
    rerank: bool              = Query(False),
    timeline: bool            = Query(False),
    page: int = 0,
    page_size: int = 1000,
    domain: str = '',
    exact_total: bool = False,
):
    if page <= 0 and not any([query, ocr_query, objects, categories, colors, video, domain]):
        return JSONResponse({"error": "Provide at least one filter"}, status_code=400)

    requested_operator = (query_operator or "and").strip().lower()
    if requested_operator not in {"and", "or"}:
        return JSONResponse(
            {"error": "query_operator must be 'and' or 'or'"},
            status_code=400,
        )

    requested_video_mode = (video_mode or "include").strip().lower()
    if requested_video_mode not in {"include", "exclude"}:
        return JSONResponse(
            {"error": "video_mode must be 'include' or 'exclude'"},
            status_code=400,
        )

    requested_ocr_operator = (ocr_operator or "and").strip().lower()
    if requested_ocr_operator not in {"and", "or"}:
        return JSONResponse(
            {"error": "ocr_operator must be 'and' or 'or'"},
            status_code=400,
        )

    col = get_collection()
    # Pagination is a presentation concern, not a request to score every vector.
    # The old `page > 0` check made the browser's very first request reconstruct
    # and score the complete FAISS index (177k+ frames), turning a sub-second
    # ANN lookup into a multi-minute exhaustive scan.  Keep that expensive mode
    # available only for explicit offline/export callers.
    exhaustive = exact_total is True
    video = (video or "").strip() or None
    scan_all = exhaustive
    requested_window = max(
        max(1, top_k),
        max(1, page) * max(1, page_size) if page > 0 else 0,
    )
    candidate_multiplier = max(1, int(os.getenv("SEARCH_CANDIDATE_MULTIPLIER", "4")))
    candidate_cap = max(1, min(
        TOP_K_MAX,
        int(os.getenv("SEARCH_CANDIDATE_LIMIT", "5000")),
    ))
    bounded_candidate_limit = min(
        candidate_cap,
        max(1000, requested_window * candidate_multiplier),
    )
    candidate_limit = (
        max(1, col.count(), _readiness.get('metadata_frames', TOP_K_MAX))
        if scan_all else bounded_candidate_limit
    )
    if domain and not re.fullmatch(r'0[1-9]', domain):
        return JSONResponse({'error': 'Invalid domain'}, status_code=400)

    comma_terms = _comma_query_terms(query)
    and_terms = comma_terms if requested_operator == "and" else []
    or_terms = comma_terms if requested_operator == "or" else []
    ocr_comma_terms = _comma_query_terms(ocr_query) if ocr_query else []
    query_meta = {
        "query_operator": requested_operator if comma_terms else "semantic",
        "query_terms": comma_terms,
        "ocr_operator": requested_ocr_operator if ocr_comma_terms else None,
        "ocr_terms": ocr_comma_terms,
        "video_mode": requested_video_mode if video else None,
    }
    # ── detect color keywords in query ────────────────────────────────────────
    query_colors = keyword_hints(query, COLOR_ALIASES)

    search_mode = (search_mode or "hybrid").lower()
    ocr_limit = candidate_limit if scan_all else min(max(top_k * 80, 500), TOP_K_MAX)
    if ocr_comma_terms:
        ocr_lookup_text = ocr_query
        ocr_term_matches = [
            _ocr_match_files(term, ocr_limit) for term in ocr_comma_terms
        ]
        ocr_sets = [set(files) for files, _ in ocr_term_matches]
        if ocr_sets:
            matched_ocr = (
                set.intersection(*ocr_sets)
                if requested_ocr_operator == "and"
                else set().union(*ocr_sets)
            )
            ocr_seed_files = sorted(matched_ocr)
            ocr_seed_prematched = all(prematched for _, prematched in ocr_term_matches)
        else:
            ocr_seed_files = []
            ocr_seed_prematched = False
    elif comma_terms and not ocr_query and search_mode in {"hybrid", "ocr_first", "ocr"}:
        ocr_lookup_text = query
        ocr_term_matches = [
            _ocr_match_files(term, ocr_limit) for term in comma_terms
        ]
        if all(prematched for _, prematched in ocr_term_matches):
            ocr_sets = [set(files) for files, _ in ocr_term_matches]
            matched_ocr = (
                set.intersection(*ocr_sets)
                if and_terms
                else set().union(*ocr_sets)
            )
            ocr_seed_files = sorted(matched_ocr)
            ocr_seed_prematched = True
        else:
            ocr_seed_files = []
            ocr_seed_prematched = False
    else:
        ocr_lookup_text = ocr_query or (
            query if search_mode in {"hybrid", "ocr_first", "ocr"} else None
        )
        ocr_seed_files, ocr_seed_prematched = _ocr_match_files(
            ocr_lookup_text or "",
            ocr_limit,
        )

    # ── CLIP/OCR candidate retrieval ─────────────────────────────────────────
    ocr_prematched = False
    visual_available = not isinstance(col, EmptyCollection) and _readiness.get('capabilities', {}).get('visual', True)
    if query and not visual_available:
        # Match the query against OCR; never silently return unrelated frames.
        if not ocr_lookup_text:
            ocr_lookup_text = query
            ocr_seed_files, ocr_seed_prematched = _ocr_match_files(query, ocr_limit)
        candidate_files = ocr_seed_files
        raw_scores = {f: 1.0 for f in candidate_files}
        ocr_prematched = True
        query_meta.update(partial=True, warnings=['Visual search unavailable; using available OCR'])
    elif query and search_mode in {"ocr_first", "ocr"} and ocr_seed_files:
        candidate_files = ocr_seed_files
        raw_scores = {f: 1.0 for f in candidate_files}
        ocr_prematched = True

    elif query:
        try:
            if comma_terms:
                n_results = candidate_limit
                if and_terms:
                    candidate_files, raw_scores = _clip_and_candidates(
                        col, and_terms, n_results
                    )
                else:
                    candidate_files, raw_scores = _clip_or_candidates(
                        col, or_terms, n_results
                    )
            else:
                n_results = candidate_limit
                text_feat = _embed_query_terms([query])[0].tolist()
                cr = col.query(query_embeddings=[text_feat], n_results=n_results)
                candidate_files = [m["file"] for m in cr["metadatas"][0]]
                raw_scores = {
                    m["file"]: 1.0 - float(d)
                    for m, d in zip(cr["metadatas"][0], cr["distances"][0])
                }

        except Exception as exc:
            if not ALLOW_PARTIAL_DATA:
                raise
            record_issue('visual.query', str(exc))
            candidate_files = ocr_seed_files
            raw_scores = {f: 1.0 for f in candidate_files}
            ocr_prematched = True
            query_meta.update(partial=True, warnings=['Visual query unavailable; showing OCR matches'])

        if search_mode == "hybrid" and ocr_seed_files:
            if and_terms:
                # OCR may boost an intersected CLIP candidate, but may not add
                # an OR-only candidate to a strict comma-AND search.
                ocr_prematched = ocr_seed_prematched
            else:
                seen = set(candidate_files)
                for f in ocr_seed_files:
                    if f not in seen:
                        candidate_files.append(f)
                        raw_scores[f] = max(raw_scores.get(f, 0.0), 0.86)
                        seen.add(f)
                ocr_prematched = True

    elif ocr_query and not (objects or categories or colors or video):
        # OCR-only search — use the FTS5 inverted index for a real search-engine
        # style lookup instead of scanning every has_text row in Python.
        if ocr_seed_files:
            candidate_files = ocr_seed_files
            raw_scores = {f: 1.0 for f in candidate_files}
            ocr_prematched = ocr_seed_prematched
        else:
            # A valid FTS no-hit is an empty result, never a full-table fallback.
            candidate_files = []
            raw_scores = {}
            ocr_prematched = bool(_federated_runtime() or _shared.FTS5_AVAILABLE)
    else:
        explicit_objects = {
            value.strip().casefold()
            for value in (objects or "").split(",")
            if value.strip()
        }
        explicit_categories = {
            value.strip().casefold()
            for value in (categories or "").split(",")
            if value.strip()
        }
        explicit_colors = {
            value.strip().casefold()
            for value in (colors or "").split(",")
            if value.strip()
        }
        candidate_files = query_frame_files(
            objects=explicit_objects,
            categories=explicit_categories,
            colors=explicit_colors,
            video=video,
            exclude_video=requested_video_mode == "exclude",
            limit=candidate_limit,
        )
        if (exhaustive or domain) and not any([objects, categories, colors, video]):
            candidate_files = [entry['file'] for entry in get_all_meta()]
        raw_scores = {filename: 1.0 for filename in candidate_files}

    # FTS matches stay authoritative when OCR is combined with other filters.
    if ocr_query and ocr_seed_prematched:
        ocr_prematched = True

    # ── load metadata for candidates ─────────────────────────────────────────
    if not candidate_files:
        return {
            "results": [],
            "total": 0,
            "auto_filtered": [],
            **query_meta,
        }

    pool = []
    for r in get_meta_by_files(candidate_files):
        item = dict(r)
        item["ocr_texts"] = []
        item["_score"] = raw_scores.get(item["file"], 0)
        pool.append(item)

    if domain:
        pool = [e for e in pool if e['file'].startswith(domain + '_')]

    # ── video filter ──────────────────────────────────────────────────────────
    if video:
        pool = [e for e in pool if _shared.matches_folder(e["file"], video)
                != (requested_video_mode == "exclude")]

    # ── OCR filter ────────────────────────────────────────────────────────────
    ocr_hits: set = set()
    if ocr_lookup_text and ocr_prematched:
        # FTS5 already identified the OCR hits. In hybrid mode they are a
        # ranking signal only; an explicit OCR query remains a hard filter.
        pool_files = {e["file"] for e in pool}
        prematched_hits = {f for f in ocr_seed_files if f in pool_files}
        ocr_hits.update(prematched_hits)
        if ocr_query or search_mode in {"ocr_first", "ocr"}:
            pool = [e for e in pool if e["file"] in prematched_hits]

    elif ocr_lookup_text:
        fallback_terms = ocr_comma_terms or (comma_terms if not ocr_query else []) or [ocr_lookup_text]
        fallback_operator = requested_ocr_operator if ocr_query else requested_operator

        def matches_ocr(joined, ascii_text):
            matches = [normalize_text(term) in joined or bool(normalize_ascii_text(term) and
                       normalize_ascii_text(term) in ascii_text) for term in fallback_terms]
            return all(matches) if fallback_operator == 'and' else any(matches)

        # Frames already cached with matching text — instant, no OCR needed
        already_matched  = [
            e for e in pool
            if matches_ocr(e["ocr_joined"], e.get("ocr_ascii", ""))
        ]
        ocr_hits.update(e["file"] for e in already_matched)

        # Frames flagged has_text but not yet OCR'd get processed one by one.
        # PaddleOCR is GPU-backed and guarded by a lock, so a thread pool only
        # adds overhead without true parallel OCR on a single 8GB GPU.
        needs_ocr = [
            e for e in pool
            if e["has_text"]
            and not matches_ocr(e["ocr_joined"], e.get("ocr_ascii", ""))
        ]

        ocr_pool = list(already_matched)
        if OCR_ON_DEMAND and needs_ocr and not _federated_runtime():
            def _ocr_one(entry):
                try:
                    from ocr_index import run_ocr
                    texts = run_ocr(
                        KEYFRAMES_DIR / entry["file"],
                        entry["width"] or 1,
                        entry["height"] or 1,
                    )
                    joined, ascii_text = join_ocr_texts(texts)
                    return entry, texts, joined, ascii_text
                except Exception:
                    return entry, None, None, None

            for entry in needs_ocr:
                entry, texts, joined, ascii_text = _ocr_one(entry)
                if texts is None:
                    continue
                entry["ocr_texts"]  = texts
                entry["ocr_joined"] = joined
                entry["ocr_ascii"]  = ascii_text
                entry["has_text"]   = len(texts) > 0
                if matches_ocr(joined, ascii_text):
                    ocr_hits.add(entry["file"])
                    ocr_pool.append(entry)
                # persist regardless of match, so future searches skip re-OCR
                with get_db() as conn:
                    conn.execute(
                        "UPDATE frames SET has_text=?, ocr_texts=?, ocr_joined=?, ocr_ascii=? WHERE file=?",
                        (int(len(texts) > 0), json.dumps(texts), joined, ascii_text, entry["file"])
                    )
                cache_update_ocr(entry["file"], texts, joined, ascii_text)
            unload_ocr()

        # A normal hybrid scene query must not lose every CLIP result merely
        # because the same sentence is absent from OCR. OCR becomes mandatory
        # only when the OCR box was used or an OCR-first mode was requested.
        if ocr_query or search_mode in {"ocr_first", "ocr"}:
            pool = ocr_pool

    # ── category filter ───────────────────────────────────────────────────────
    if categories:
        wanted = {
            c.strip().casefold() for c in categories.split(",") if c.strip()
        }
        pool = [
            e for e in pool
            if wanted & {value.casefold() for value in e["categories"]}
        ]

    # ── object filter ─────────────────────────────────────────────────────────
    all_known = {o.lower() for e in pool for o in e["objects"]}
    auto_filter: set = set()
    if query:
        query_words = set(re.findall(r"[\w-]+", normalize_ascii_text(query)))
        auto_filter = query_words & all_known
    explicit_objects = {
        value.strip().lower()
        for value in (objects or "").split(",")
        if value.strip()
    }
    if and_terms and auto_filter:
        pool = [
            e for e in pool
            if auto_filter <= {o.lower() for o in e["objects"]}
        ]
    else:
        wanted_obj = explicit_objects or auto_filter
        if wanted_obj:
            pool = [
                e for e in pool
                if wanted_obj & {o.lower() for o in e["objects"]}
            ]

    if and_terms and explicit_objects:
        pool = [
            e for e in pool
            if explicit_objects & {o.lower() for o in e["objects"]}
        ]

    # ── color filter ──────────────────────────────────────────────────────────
    if colors:
        wanted_colors = {
            c.strip().casefold() for c in colors.split(",") if c.strip()
        }
        pool = [
            e for e in pool
            if wanted_colors & {value.casefold() for value in e["colors"]}
        ]

    # color keyword filter from query
    if query_colors:
        if and_terms:
            pool = [
                e for e in pool
                if query_colors <= set(e["colors"])
            ]
        else:
            filtered = [e for e in pool if query_colors & set(e["colors"])]
            if filtered:
                pool = filtered

    # ── confidence filter ─────────────────────────────────────────────────────
    if min_conf > 0:
        pool = [
            e for e in pool
            if any(d.get("conf", 0) >= min_conf for d in e["detections"])
        ]

    if not pool:
        return {
            "results": [],
            "total": 0,
            "auto_filtered": sorted(auto_filter),
            **query_meta,
        }

    # ── apply boosts and sort ─────────────────────────────────────────────────
    q_norm = normalize_text(query) if query else ""
    asr_terms = (
        [normalize_text(term) for term in comma_terms]
        if comma_terms
        else ([q_norm] if q_norm else [])
    )
    asr_videos: set[str] = set()
    # CLIP-only searches, including the dedicated Video Match workflow, must
    # never pay the cost of loading/scanning ASR transcripts.
    if SEARCH_ASR_BOOST and asr_terms and search_mode != "clip":
        def _matches_asr_terms(segments) -> bool:
            matches = [
                any(
                    term in normalize_text(segment.get("text", ""))
                    for segment in segments
                )
                for term in asr_terms
            ]
            return all(matches) if and_terms else any(matches)

        asr_cache = load_asr_cache()
        asr_videos = {
            Path(asr_cache_video_name(key)).stem
            for key, segs in asr_cache.items()
            if _matches_asr_terms(segs)
        }

    for e in pool:
        ocr_boost   = 0.45 if e["file"] in ocr_hits else 0.0
        asr_boost   = 0.05 if any(v in e["file"] for v in asr_videos) else 0.0
        color_boost = 0.08 * len(query_colors & set(e["colors"]))
        e["_score"] += ocr_boost + asr_boost + color_boost

    pre_dedupe_total = len(pool)
    if not query:
        pool.sort(key=lambda e: e["file"])
    else:
        pool.sort(key=lambda e: (-e["_score"], e["file"]))
        dedupe_gap = max(0, int(os.getenv("SEARCH_DEDUP_FRAME_GAP", "3")))
        # Let the optional relationship reranker see the full bounded head;
        # otherwise adjacent-frame deduplication could discard its best match
        # before the model evaluates it.
        pool = pool if (exhaustive or rerank) else _dedupe_ranked_frames(pool, dedupe_gap)
    deduplicated = pre_dedupe_total - len(pool)

    reranking = {"status": "disabled"}
    if rerank is True and query:
        pool, reranking = _vlm_reranker.rerank(
            query, pool, lambda file: _vlm_context(file, use_timeline=timeline is True)
        )
    if page > 0:
        page_size = max(1, page_size)
        page = min(page, max(1, (len(pool) + page_size - 1) // page_size))
        top_pool = pool[(page-1)*page_size:page*page_size]
    else:
        top_pool = pool[:top_k]
    full_rows = _fetch_frame_rows([e["file"] for e in top_pool])
    ocr_texts_by_file = {
        r["file"]: _shared._metadata_list(r, 'ocr_texts')
        for r in full_rows
    }

    results = [{
        "file":        e["file"],
        "width":       e["width"],
        "height":      e["height"],
        "objects":     e["objects"],
        "categories":  e["categories"],
        "colors":      e["colors"],
        "detections":  e["detections"],
        "ocr_texts":   e["ocr_texts"] or ocr_texts_by_file.get(e["file"], []),
        "ocr_matched": e["file"] in ocr_hits,
        **({"vlm_score": e["vlm_score"], "rerank_score": round(e["rerank_score"], 6),
            "timeline": e["timeline"]} if "vlm_score" in e else {}),
        "score":       round(e["_score"], 4),
    } for e in top_pool]

    return {
        "results": results,
        "total": len(pool),
        "page": page,
        "page_size": page_size,
        "deduplicated": deduplicated,
        "reranking": reranking,
        "auto_filtered": sorted(auto_filter),
        **query_meta,
    }




@app.get("/video-match")
def video_match(
    query: str = Query(..., min_length=1),
    top_videos: int = Query(VIDEO_MATCH_TOP_VIDEOS, ge=1, le=200),
    frames_per_video: int = Query(VIDEO_MATCH_FRAMES_PER_VIDEO, ge=1, le=8),
    candidate_frames: int = Query(VIDEO_MATCH_CANDIDATES, ge=100, le=TOP_K_MAX),
):
    """
    Fast visual-only video retrieval.

    Unlike /search?search_mode=hybrid, this endpoint performs only:
      1. one CLIP text embedding,
      2. one FAISS query,
      3. in-RAM metadata grouping.

    It deliberately skips OCR, ASR, SQLite full-row reads and on-demand model
    work. Results are grouped and ranked server-side so the browser receives
    only a few representative frames for each source video.
    """
    query = query.strip()
    if not query:
        return JSONResponse({"error": "Provide a visual description"}, status_code=400)

    total_started = time.perf_counter()
    col = get_collection()

    if ALLOW_PARTIAL_DATA and (isinstance(col, EmptyCollection) or
                              not _readiness.get('capabilities', {}).get('visual', True)):
        record_issue('video.search', 'Visual search data unavailable')
        return {'query': query, 'videos': [], 'total_videos': 0, 'total_frames': 0,
                'partial': True, 'warnings': ['Visual data unavailable; see sever_log.json']}

    embed_started = time.perf_counter()
    expanded = expand_query(query)
    text_feat = embed_texts(expanded).tolist()
    embed_ms = (time.perf_counter() - embed_started) * 1000

    # Always retrieve enough frames to give each requested video a fair chance,
    # while respecting the configured/global safety limit.
    n_results = min(
        TOP_K_MAX,
        max(candidate_frames, top_videos * frames_per_video * 8),
    )

    faiss_started = time.perf_counter()
    cr = col.query(query_embeddings=[text_feat], n_results=n_results)
    faiss_ms = (time.perf_counter() - faiss_started) * 1000

    metadatas = cr.get("metadatas", [[]])[0]
    distances = cr.get("distances", [[]])[0]
    raw_scores = {
        meta["file"]: 1.0 - float(distance)
        for meta, distance in zip(metadatas, distances)
        if meta.get("file")
    }
    candidate_files = [meta["file"] for meta in metadatas if meta.get("file")]

    grouping_started = time.perf_counter()
    query_colors = keyword_hints(query, COLOR_ALIASES)

    groups: dict[str, list[dict]] = {}
    for meta in get_meta_by_files(candidate_files):
        item = dict(meta)
        score = float(raw_scores.get(item["file"], 0.0))
        # Color is a small soft boost, never a hard filter. Dominant-color
        # metadata can be noisy, so it must not erase a valid CLIP match.
        if query_colors:
            score += 0.03 * len(query_colors & set(item.get("colors", [])))
        item["_score"] = score
        groups.setdefault(_video_id_from_frame(item["file"]), []).append(item)

    ranked_groups = []
    for video_name, frames in groups.items():
        frames.sort(key=lambda item: item["_score"], reverse=True)
        representative = _select_diverse_video_frames(
            frames, frames_per_video, VIDEO_MATCH_MIN_FRAME_GAP
        )
        ranked_groups.append({
            "video": video_name,
            "_rank_score": _video_rank_score(frames),
            "best_score": float(frames[0]["_score"]),
            "candidate_frame_count": len(frames),
            "frames": representative,
        })

    ranked_groups.sort(key=lambda group: group["_rank_score"], reverse=True)
    ranked_groups = ranked_groups[:top_videos]

    videos = []
    for group in ranked_groups:
        videos.append({
            "video": group["video"],
            "score": round(float(group["_rank_score"]), 4),
            "best_score": round(float(group["best_score"]), 4),
            "candidate_frame_count": group["candidate_frame_count"],
            "frames": [{
                "file": item["file"],
                "storage_id": item.get("storage_id"),
                "width": item.get("width"),
                "height": item.get("height"),
                "objects": item.get("objects", []),
                "categories": item.get("categories", []),
                "colors": item.get("colors", []),
                "detections": item.get("detections", []),
                "ocr_texts": [],
                "ocr_matched": False,
                "score": round(float(item["_score"]), 4),
            } for item in group["frames"]],
        })

    grouping_ms = (time.perf_counter() - grouping_started) * 1000
    total_ms = (time.perf_counter() - total_started) * 1000
    log.info(
        "Video match: candidates=%d, videos=%d, total=%.1f ms "
        "(embed=%.1f, faiss=%.1f, grouping=%.1f)",
        len(candidate_files), len(videos), total_ms, embed_ms, faiss_ms, grouping_ms,
    )

    return {
        "query": query,
        "videos": videos,
        "total_videos": len(groups),
        "candidate_frames": len(candidate_files),
        "query_colors": sorted(query_colors),
        "timings": {
            "embed_ms": round(embed_ms, 1),
            "faiss_ms": round(faiss_ms, 1),
            "grouping_ms": round(grouping_ms, 1),
            "total_ms": round(total_ms, 1),
        },
    }


def _public_frame_results(entries: list[dict]) -> list[dict]:
    return [{
        "file": item["file"],
        "width": item.get("width"),
        "height": item.get("height"),
        "objects": item.get("objects", []),
        "categories": item.get("categories", []),
        "colors": item.get("colors", []),
        "detections": item.get("detections", []),
        "ocr_texts": [],
        "ocr_matched": False,
        "score": round(float(item.get("_score", 1.0)), 4),
    } for item in entries]


@app.get("/similar")
def similar_frames(
    file: str = Query(..., min_length=1),
    top_k: int = Query(120, ge=1, le=1000),
    page: int = 0,
):
    """Find visually similar frames using the selected frame's stored vector."""
    col = get_collection()
    vectors = col.reconstruct_by_files([file])
    vector = vectors.get(file)
    if vector is None:
        if ALLOW_PARTIAL_DATA:
            record_issue('vectors', 'Frame vector missing', path=file)
            return {'source': file, 'results': [], 'total': 0, 'partial': True}
        return JSONResponse({"error": "Frame vector not found"}, status_code=404)

    n_results = max(1, col.count()) if page > 0 else min(max(top_k * 4, 200), TOP_K_MAX)
    result = col.query(query_embeddings=[vector.tolist()], n_results=n_results)
    candidates = []
    for meta, distance in zip(
        result.get("metadatas", [[]])[0],
        result.get("distances", [[]])[0],
    ):
        filename = meta.get("file")
        if filename and filename != file:
            candidates.append((filename, 1.0 - float(distance)))

    score_by_file = dict(candidates)
    entries = []
    for item in get_meta_by_files([filename for filename, _ in candidates]):
        entry = dict(item)
        entry["_score"] = score_by_file.get(entry["file"], 0.0)
        entries.append(entry)
    entries.sort(key=lambda item: (-item["_score"], item["file"]))
    total = len(entries)
    if page > 0:
        page = min(page, max(1, (total + 999) // 1000))
        entries = entries[(page-1)*1000:page*1000]
    else:
        entries = _dedupe_ranked_frames(entries, 3)[:top_k]
        total = len(entries)
    return {
        "source": file,
        "results": _public_frame_results(entries),
        "total": total,
        "page": page,
        "query_operator": "similar",
        "query_terms": [],
    }


@app.get("/frame/neighbors")
def frame_neighbors(
    file: str = Query(..., min_length=1),
    radius: int = Query(6, ge=1, le=30),
):
    """Return keyframes immediately before/after the selected frame."""
    if REDIS_ENABLED:
        from redis_serving import neighbors
        return neighbors(file, radius)
    federation = _federated_runtime()
    if federation:
        reader = federation.owner(file)
        conn_ctx = reader.connect() if reader else get_db()
    else:
        conn_ctx = get_db()
    with conn_ctx as conn:
        center = conn.execute(
            "SELECT video, n FROM frame_locator WHERE file=?",
            (file,),
        ).fetchone()
        if not center:
            return JSONResponse({"error": "Frame locator not found"}, status_code=404)
        before = conn.execute(
            "SELECT file FROM frame_locator WHERE video=? AND n<? "
            "ORDER BY n DESC LIMIT ?",
            (center["video"], center["n"], radius),
        ).fetchall()
        after = conn.execute(
            "SELECT file FROM frame_locator WHERE video=? AND n>? "
            "ORDER BY n ASC LIMIT ?",
            (center["video"], center["n"], radius),
        ).fetchall()

    files = [row["file"] for row in reversed(before)] + [file] + [
        row["file"] for row in after
    ]
    entries = []
    for item in get_meta_by_files(files):
        entry = dict(item)
        entry["_score"] = 1.0
        entry["is_center"] = entry["file"] == file
        entries.append(entry)
    results = _public_frame_results(entries)
    for result, entry in zip(results, entries):
        result["is_center"] = entry["is_center"]
    return {"video": center["video"], "center": file, "results": results}



@app.get("/search/features")
def search_features():
    return {"vlm_available": _vlm_reranker.scorer is not None,
            "vlm_top_n": _vlm_reranker.top_n, "timeline_available": True}


def _timeline_window(file, window_seconds=10.0, max_frames=9):
    video = _video_id_from_frame(file)
    frames, mapping = _load_asr_keyframe_timeline(video)
    center = next((f for f in frames if f["file"] == file), None)
    if center is None:
        return [], mapping
    nearby = [f for f in frames if abs(f["timestamp"]-center["timestamp"]) <= window_seconds]
    if len(nearby) > max_frames:
        # Keep the target plus evenly spaced context across the entire window.
        indices = {round(i*(len(nearby)-1)/(max_frames-1)) for i in range(max_frames)}
        sampled = [nearby[i] for i in sorted(indices)]
        if center not in sampled:
            nearest = min(range(len(sampled)), key=lambda i: abs(sampled[i]["timestamp"]-center["timestamp"]))
            sampled[nearest] = center
        nearby = sorted(sampled, key=lambda f: f["timestamp"])
    return [dict(f, is_center=f["file"] == file) for f in nearby], mapping


def _vlm_context(file, *, use_timeline=False):
    frames = _timeline_window(file, max_frames=3)[0] if use_timeline else []
    if not frames:
        frames = [{"file": file, "is_center": True}]
    return [dict(f, path=str(path)) for f in frames
            if (path := _keyframe_path(f["file"])) is not None]


@app.get("/frame/timeline")
def frame_timeline(file: str = Query(..., min_length=1),
                   window_seconds: float = Query(10.0, ge=.5, le=120),
                   max_frames: int = Query(9, ge=3, le=31)):
    frames, mapping = _timeline_window(file, window_seconds, max_frames)
    return {"center": file, "video": _video_id_from_frame(file), "results": frames,
            "mapping_mode": mapping.get("mapping_mode"), "window_seconds": window_seconds}


@app.get("/frame/meta/{filename:path}")
def frame_meta(filename: str):
    """Return full metadata including detections + OCR for lightbox."""
    meta = get_frame_meta(filename)
    if not meta:
        return JSONResponse({"error": "Not found"}, status_code=404)
    return meta

@app.post("/frame/ocr")
def frame_ocr_on_demand(body: dict):
    """Run OCR on a single frame and return results (for lightbox)."""
    if not OCR_ON_DEMAND:
        return JSONResponse({
            "error": "OCR on-demand is disabled. Run offline OCR indexing first or start server with OCR_ON_DEMAND=1."
        }, status_code=403)
    if _federated_runtime() or _shared.SERVE_READ_ONLY:
        return JSONResponse({
            "error": "OCR on-demand cannot write to a federated/read-only dataset. Run offline OCR indexing instead."
        }, status_code=403)
    filename = body.get("file", "").strip()
    if not filename:
        return JSONResponse({"error": "Provide file"}, status_code=400)
    path = _keyframe_path(filename)
    if path is None:
        return JSONResponse({"error": "Not found"}, status_code=404)
    try:
        from ocr_index import run_ocr
        image, _ = load_image(path)
        h, w = image.shape[:2]
        texts = run_ocr(image, w, h)
        joined, ascii_text = join_ocr_texts(texts)
        with get_db() as conn:
            conn.execute(
                "UPDATE frames SET has_text=?, ocr_texts=?, ocr_joined=?, ocr_ascii=? WHERE file=?",
                (int(len(texts) > 0), json.dumps(texts), joined, ascii_text, filename)
            )
        cache_update_ocr(filename, texts, joined, ascii_text)
        unload_ocr()
        return {"ocr_texts": texts}
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/image/{filename:path}")
def serve_image(filename: str):
    path = _keyframe_path(filename)
    if path is None or not path.exists() or path.suffix.lower() not in IMAGE_EXTS:
        record_issue('keyframe', 'Requested image is missing', path=filename)
        return JSONResponse({"error": "Not found"}, status_code=404)
    return FileResponse(str(path))

# ══════════════════════════════════════════════════════════════════════════════
# ASR routes — fixed 2-second transcript windows + existing keyframes only
# ══════════════════════════════════════════════════════════════════════════════

_asr_keyframe_index: dict[str, dict[int, str]] = {}
_asr_timeline_cache: dict[str, tuple[list[dict], dict]] = {}
ASR_TIMELINE_CACHE_MAX = max(1, int(os.getenv("ASR_TIMELINE_CACHE_MAX", "32")))
_asr_videos_dir_lock = threading.RLock()
_asr_timeline_lock = threading.RLock()
_asr_videos_dir_override: Optional[Path] = None


def _remember_asr_timeline(
    cache_key: str,
    result: tuple[list[dict], dict],
) -> tuple[list[dict], dict]:
    _asr_timeline_cache.pop(cache_key, None)
    _asr_timeline_cache[cache_key] = result
    while len(_asr_timeline_cache) > ASR_TIMELINE_CACHE_MAX:
        expired = next(iter(_asr_timeline_cache))
        _asr_timeline_cache.pop(expired)
        for video in list(_asr_keyframe_index):
            if video.casefold() == expired:
                _asr_keyframe_index.pop(video, None)
    return result


def _configured_videos_dir() -> Path:
    """Return the active original-video folder as an absolute path."""
    with _asr_videos_dir_lock:
        folder = _asr_videos_dir_override or VIDEOS_DIR
    return folder.resolve()


def _videos_in_configured_dir() -> list[Path]:
    if REDIS_TEXT_ENABLED:
        # Redis serving reads transcripts, never scans media to check readiness.
        return []
    folder = _configured_videos_dir()
    if not folder.is_dir():
        return []
    return sorted(
        path for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTS
    )


def _asr_cache_inventory(cache: Optional[dict] = None) -> dict:
    """Describe portable ASR entries without consulting original videos."""
    cache = load_asr_cache() if cache is None else cache
    entries: list[dict] = []
    invalid = 0

    for key, segments in cache.items():
        # Portable keys should already contain only a filename. Taking the
        # final component also prevents legacy machine paths leaking into APIs.
        raw_name = asr_cache_video_name(key).replace("\\", "/")
        video_name = raw_name.rsplit("/", 1)[-1].strip()
        if is_asr_excluded_video(video_name):
            continue
        if not video_name:
            invalid += 1
            continue
        current = is_current_asr_segments(segments)
        if not current:
            invalid += 1
        entries.append({
            "key": key,
            "video_name": video_name,
            "video_id": Path(video_name).stem,
            "segments": segments,
            "current": current,
        })

    return {
        "entries": entries,
        "cached_videos": len(entries),
        "valid_cached_videos": sum(1 for entry in entries if entry["current"]),
        "invalid_cached_videos": invalid,
    }


def _extract_asr_frame_number(filename: str) -> tuple[Optional[int], Optional[str]]:
    stem = Path(filename).stem
    match = re.search(r"^(.*?)[_-](\d+)$", stem)
    if match:
        return int(match.group(2)), match.group(1)
    match = re.search(r"(\d+)$", stem)
    if match:
        return int(match.group(1)), None
    return None, None


def _rebuild_asr_keyframe_index(
    video_stem: Optional[str] = None,
) -> dict[str, dict[int, str]]:
    """Load one video's offline locator rows; never scan every frame."""
    if not video_stem:
        return _asr_keyframe_index
    for key in _asr_keyframe_index:
        if key.casefold() == video_stem.casefold():
            return _asr_keyframe_index
    federation = _federated_runtime()
    if federation:
        rows = federation.frame_locator(video_stem)
    else:
        with closing(get_db()) as conn:
            rows = conn.execute(
                "SELECT n, file FROM frame_locator "
                "WHERE video=? COLLATE NOCASE ORDER BY n",
                (video_stem,),
            ).fetchall()
    _asr_keyframe_index[video_stem] = {
        int(row["n"]): str(row["file"]) for row in rows
    }
    return _asr_keyframe_index


def _scan_asr_keyframes_for_video(video_stem: str) -> dict[int, str]:
    """Find existing keyframes without creating any new image."""
    found: dict[int, str] = {}
    seen_paths: set[Path] = set()
    candidate_paths: list[Path] = []

    direct_dir = KEYFRAMES_DIR / video_stem
    if direct_dir.is_dir():
        candidate_paths.extend(direct_dir.rglob("*"))

    try:
        for directory in KEYFRAMES_DIR.rglob(video_stem):
            if directory.is_dir() and directory != direct_dir:
                candidate_paths.extend(directory.rglob("*"))
    except Exception:
        pass

    try:
        candidate_paths.extend(KEYFRAMES_DIR.rglob(f"{video_stem}_*"))
    except Exception:
        pass

    for path in candidate_paths:
        if (
            path in seen_paths
            or not path.is_file()
            or path.suffix.lower() not in IMAGE_EXTS
        ):
            continue

        seen_paths.add(path)
        number, _ = _extract_asr_frame_number(path.name)
        if number is None:
            continue
        try:
            found[number] = str(path.relative_to(KEYFRAMES_DIR)).replace("\\", "/")
        except ValueError:
            continue

    return found


def _find_asr_map_csv(video_stem: str) -> Optional[Path]:
    federation = _federated_runtime()
    if federation:
        return federation.resolve_map(video_stem)
    external = _external_asset_resolver()
    if external:
        return external.resolve_map(video_stem)
    direct = MAPFILES_DIR / f"{video_stem}.csv"
    if direct.exists():
        return direct

    target = f"{video_stem}.csv".casefold()
    try:
        for candidate in MAPFILES_DIR.rglob("*.csv"):
            if candidate.name.casefold() == target:
                return candidate
    except Exception:
        pass
    return None


def _load_asr_keyframe_timeline(video_stem: str) -> tuple[list[dict], dict]:
    try:
        if REDIS_TEXT_ENABLED:
            return redis_store().timeline(video_stem), {'mapping_mode': 'redis_csv_n'}
        with _asr_timeline_lock:
            return _load_asr_keyframe_timeline_locked(video_stem)
    except (OSError, sqlite3.Error, ValueError) as exc:
        if not ALLOW_PARTIAL_DATA:
            raise
        record_issue('timeline', str(exc), path=video_stem)
        return [], {'mapping_mode': 'unavailable'}


def _can_map_keyframe(rel):
    # Read-only locators were validated offline. Checking thousands of USB
    # files here blocks every first search; /image verifies requested assets.
    return _shared.SERVE_READ_ONLY or _keyframe_path(rel) is not None


def _load_asr_keyframe_timeline_locked(video_stem: str) -> tuple[list[dict], dict]:
    """
    Map timestamps to files already present under keyframes/.

    Mapping priority:
      1. CSV n
      2. CSV frame_idx
      3. positional pairing
      4. inferred ASR_KEYFRAME_INTERVAL, default 0.5 seconds
    """
    import csv

    cache_key = video_stem.casefold()
    cached = _asr_timeline_cache.pop(cache_key, None)
    if cached is not None:
        _asr_timeline_cache[cache_key] = cached
        return cached

    frame_index = _rebuild_asr_keyframe_index(video_stem)
    files_by_number: dict[int, str] = {}
    requested_key = video_stem.casefold()

    for video_id, mapping in frame_index.items():
        if video_id.casefold() == requested_key:
            files_by_number.update(mapping)

    metadata_keyframe_count = len(files_by_number)

    disk_files = (
        {} if _shared.SERVE_READ_ONLY
        else _scan_asr_keyframes_for_video(video_stem)
    )
    disk_keyframe_count = len(disk_files)
    if disk_files:
        files_by_number.update(disk_files)
        frame_index[video_stem] = dict(files_by_number)

    csv_path = _find_asr_map_csv(video_stem)
    debug = {
        "video_id": video_stem,
        "map_csv": str(csv_path) if csv_path else None,
        "metadata_keyframe_count": metadata_keyframe_count,
        "disk_keyframe_count": disk_keyframe_count,
        "keyframe_count": len(files_by_number),
        "csv_row_count": 0,
        "mapped_count": 0,
        "mapping_mode": None,
    }

    if not files_by_number:
        debug["mapping_mode"] = "no_keyframes"
        result = ([], debug)
        return _remember_asr_timeline(cache_key, result)

    rows: list[dict] = []
    if csv_path:
        try:
            with csv_path.open(newline="", encoding="utf-8-sig") as handle:
                for row in csv.DictReader(handle):
                    try:
                        number = int(float(row["n"]))
                        timestamp = float(row["pts_time"])
                    except (KeyError, TypeError, ValueError):
                        continue

                    try:
                        frame_index_value = int(float(row.get("frame_idx", number)))
                    except (TypeError, ValueError):
                        frame_index_value = number

                    rows.append({
                        "n": number,
                        "timestamp": timestamp,
                        "frame_idx": frame_index_value,
                    })
        except Exception as exc:
            log.warning(f"Could not read map CSV for {video_stem}: {exc}")
            rows = []

    rows.sort(key=lambda item: (item["timestamp"], item["n"]))
    debug["csv_row_count"] = len(rows)

    timeline: list[dict] = []
    used_files: set[str] = set()

    for row in rows:
        rel = files_by_number.get(row["n"])
        mode = "csv_n"
        if not rel:
            rel = files_by_number.get(row["frame_idx"])
            mode = "csv_frame_idx"

        if not rel or rel in used_files or not _can_map_keyframe(rel):
            continue

        used_files.add(rel)
        timeline.append({
            "file": rel,
            "timestamp": round(row["timestamp"], 3),
            "n": row["n"],
            "frame_idx": row["frame_idx"],
        })

        if debug["mapping_mode"] is None:
            debug["mapping_mode"] = mode
        elif debug["mapping_mode"] != mode:
            debug["mapping_mode"] = "csv_mixed"

    exact_count = len(timeline)
    minimum_expected = min(len(rows), len(files_by_number))
    poor_exact_match = (
        bool(rows)
        and bool(minimum_expected)
        and exact_count < max(1, int(minimum_expected * 0.10))
    )
    if poor_exact_match:
        ordered_files = [
            rel
            for _, rel in sorted(files_by_number.items())
            if _can_map_keyframe(rel)
        ]
        positional = [
            {
                "file": rel,
                "timestamp": round(row["timestamp"], 3),
                "n": row["n"],
                "frame_idx": row["frame_idx"],
            }
            for row, rel in zip(rows, ordered_files)
        ]
        if positional:
            timeline = positional
            debug["mapping_mode"] = "csv_positional"

    if not rows:
        interval = float(os.getenv("ASR_KEYFRAME_INTERVAL", "0.5"))
        timeline = [
            {
                "file": rel,
                "timestamp": round(index * interval, 3),
                "n": number,
                "frame_idx": number,
            }
            for index, (number, rel) in enumerate(sorted(files_by_number.items()))
            if _can_map_keyframe(rel)
        ]
        if timeline:
            debug["mapping_mode"] = "inferred_interval"
            debug["inferred_interval"] = interval

    timeline.sort(key=lambda item: (item["timestamp"], item["n"]))
    debug["mapped_count"] = len(timeline)
    if debug["mapping_mode"] is None:
        debug["mapping_mode"] = "unmapped"
    result = (timeline, debug)
    return _remember_asr_timeline(cache_key, result)


def _frames_in_asr_interval(
    timeline: list[dict],
    start: float,
    end: float,
) -> list[dict]:
    """Return the nearest left frame, all frames in [start, end), and right frame."""
    if not timeline or end <= start:
        return []
    epsilon = 1e-6
    first = bisect_left(timeline, start-epsilon, key=lambda frame: float(frame["timestamp"]))
    last = bisect_left(timeline, end-epsilon, key=lambda frame: float(frame["timestamp"]))
    neighbors = []
    if first > 0:
        neighbors.append({**timeline[first - 1], "asr_position": "before"})
    neighbors.extend(timeline[first:last])
    if last < len(timeline):
        neighbors.append({**timeline[last], "asr_position": "after"})
    return neighbors


@app.get("/asr/config")
def asr_config():
    """Expose independent search-cache and transcription-source readiness."""
    folder = _configured_videos_dir()
    source_videos = _videos_in_configured_dir()
    cache = load_asr_cache()
    inventory = _asr_cache_inventory(cache)
    cache_files = _asr_cache_files()
    cache_exists = bool(redis_store().json('asr_cache')) if REDIS_TEXT_ENABLED else bool(cache_files)
    search_ready = cache_exists and inventory["valid_cached_videos"] > 0
    transcription_ready = folder.is_dir() and bool(source_videos)
    return {
        "cache_file": str(ASR_CACHE_FILE.resolve()),
        "cache_files": [str(path.resolve()) for path in cache_files],
        "cache_exists": cache_exists,
        "search_ready": search_ready,
        "cached_videos": inventory["cached_videos"],
        "valid_cached_videos": inventory["valid_cached_videos"],
        "invalid_cached_videos": inventory["invalid_cached_videos"],
        "videos_folder": str(folder),
        "videos_folder_exists": folder.is_dir(),
        "using_override": _asr_videos_dir_override is not None,
        "source_videos": len(source_videos),
        "transcription_ready": transcription_ready,
        "transcription_enabled": transcription_ready and ALLOW_HEAVY_RUNTIME_JOBS,
        # Backward compatibility for clients that used exists for videos/.
        "exists": folder.is_dir(),
    }


@app.post("/asr/config")
def update_asr_config(body: dict):
    """Select or reset the source-video folder used by ASR runtime jobs."""
    global _asr_videos_dir_override

    if body.get("reset"):
        with _asr_videos_dir_lock:
            _asr_videos_dir_override = None
        result = asr_config()
        result["updated"] = True
        return result

    raw_folder = body.get("videos_folder")
    if not isinstance(raw_folder, str) or not raw_folder.strip():
        return JSONResponse(
            {"error": "videos_folder must be a non-empty absolute path"},
            status_code=400,
        )
    if "\x00" in raw_folder:
        return JSONResponse({"error": "Invalid videos folder path"}, status_code=400)

    folder = Path(raw_folder.strip()).expanduser()
    if not folder.is_absolute():
        return JSONResponse(
            {"error": "Please enter an absolute path, for example /mnt/d/AIC/data/videos"},
            status_code=400,
        )

    folder = folder.resolve()
    if not folder.is_dir():
        return JSONResponse(
            {"error": f"Videos folder not found: {folder}"},
            status_code=404,
        )

    source_videos = sorted(
        path for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTS
    )
    if not source_videos:
        return JSONResponse(
            {"error": f"No supported video files found in: {folder}"},
            status_code=400,
        )

    with _asr_videos_dir_lock:
        _asr_videos_dir_override = folder

    result = asr_config()
    result["updated"] = True
    return result


@app.get("/asr/status")
def asr_status():
    cache = load_asr_cache()
    inventory = _asr_cache_inventory(cache)
    cache_files = _asr_cache_files()
    source_videos = _videos_in_configured_dir()
    valid_keys = {
        entry["key"] for entry in inventory["entries"] if entry["current"]
    }
    missing_cache = sum(
        1 for video in source_videos if asr_cache_key(video) not in valid_keys
    )
    return {
        # Preserve old fields, now defined solely by cache contents.
        "transcribed": inventory["valid_cached_videos"],
        "total": inventory["cached_videos"],
        "pending": inventory["invalid_cached_videos"],
        "cached_videos": inventory["cached_videos"],
        "valid_cached_videos": inventory["valid_cached_videos"],
        "invalid_cached_videos": inventory["invalid_cached_videos"],
        "cache_exists": bool(cache) if REDIS_TEXT_ENABLED else bool(cache_files),
        "cache_files": [str(path.resolve()) for path in cache_files],
        "search_ready": (
            (REDIS_TEXT_ENABLED or bool(cache_files)) and inventory["valid_cached_videos"] > 0
        ),
        "source_videos": len(source_videos),
        "source_videos_missing_cache": missing_cache,
        "videos_folder_exists": _configured_videos_dir().is_dir(),
        "transcription_ready": bool(source_videos),
        "videos_folder": str(_configured_videos_dir()),
        "chunk_seconds": 2.0,
        "frame_source": "keyframes",
    }


@app.post("/asr/transcribe")
def asr_transcribe_all(body: dict):
    if os.getenv('ASR_PROCESSING_ENABLED', '0').lower() not in {'1', 'true', 'yes'}:
        return JSONResponse({'error': 'ASR is processed externally; imported ASR remains searchable'}, status_code=403)
    if not ALLOW_HEAVY_RUNTIME_JOBS:
        return JSONResponse(
            {"error": "ASR transcription is disabled in immutable serving mode"},
            status_code=403,
        )
    folder = _configured_videos_dir()
    if not folder.is_dir():
        return JSONResponse({"error": f"Videos folder not found: {folder}"}, status_code=404)

    videos = _videos_in_configured_dir()
    cache = load_asr_cache()
    done = 0
    skipped = 0
    failed = 0

    for video in videos:
        key = asr_cache_key(video)
        if key in cache and is_current_asr_segments(cache[key]):
            skipped += 1
            continue
        try:
            transcribe_video(video, cache)
            done += 1
        except Exception as exc:
            failed += 1
            log.error(f"Transcribe failed {video.name}: {exc}")

    unload_whisper()
    return {
        "transcribed": done,
        "skipped_cached": skipped,
        "failed": failed,
        "total": len(videos),
        "chunk_seconds": 2.0,
        "frame_source": "keyframes",
        "videos_folder": str(folder),
    }


def _search_asr_fts(phrase: str, *, offset: int, limit: int, operator: str = "and") -> Optional[dict]:
    """Return one stable ASR page, supporting comma-separated terms with AND/OR logic."""
    # A single-root FTS connection cannot cover every federated reader.
    if REDIS_TEXT_ENABLED:
        return redis_store().asr_matches(phrase, offset, limit, operator)
    if _federated_runtime():
        return None
    raw_terms = [t.strip() for t in phrase.split(",") if t.strip()] if "," in phrase else [phrase.strip()]
    clauses = []
    for term in raw_terms:
        tokens = re.findall(r"\w+", normalize_asr_text(term), flags=re.UNICODE)
        if tokens:
            clauses.append(chr(34) + " ".join(tokens) + chr(34))
    if not clauses:
        return {"matches": [], "total": 0, "videos": 0}

    is_multi = len(clauses) > 1
    op = (operator or "and").lower()

    try:
        with closing(get_db()) as conn:
            tables = {
                str(row[0])
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type IN ('table','view') "
                    "AND name IN ('asr_segments','asr_segments_fts')"
                )
            }
            if tables != {"asr_segments", "asr_segments_fts"}:
                return None

            conn.create_function("asr_excluded", 1, is_asr_excluded_video, deterministic=True)
            if not is_multi or op == "or":
                match_query = " OR ".join(clauses)
                total = int(conn.execute(
                    "SELECT COUNT(*) FROM asr_segments_fts "
                    "JOIN asr_segments AS s ON s.id=asr_segments_fts.rowid "
                    "WHERE asr_segments_fts MATCH ? AND NOT asr_excluded(s.video_id)",
                    (match_query,),
                ).fetchone()[0])
                rows = conn.execute(
                    "SELECT s.video_name,s.video_id,s.start,s.end,s.text,s.search_text,-bm25(asr_segments_fts) AS relevance "
                    "FROM asr_segments_fts "
                    "JOIN asr_segments AS s ON s.id=asr_segments_fts.rowid "
                    "WHERE asr_segments_fts MATCH ? AND NOT asr_excluded(s.video_id) "
                    "ORDER BY relevance DESC,s.video_name COLLATE NOCASE,s.start,s.id "
                    "LIMIT ? OFFSET ?",
                    (match_query, limit, offset),
                ).fetchall()
                videos = int(conn.execute(
                    "SELECT COUNT(DISTINCT s.video_id) FROM asr_segments_fts "
                    "JOIN asr_segments AS s ON s.id=asr_segments_fts.rowid "
                    "WHERE asr_segments_fts MATCH ? AND NOT asr_excluded(s.video_id)",
                    (match_query,),
                ).fetchone()[0])
            else:
                # operator == "and": video must contain all clauses
                sub_intersects = " INTERSECT ".join(
                    "SELECT video_id FROM asr_segments_fts JOIN asr_segments ON asr_segments.id=asr_segments_fts.rowid WHERE asr_segments_fts MATCH ? AND NOT asr_excluded(video_id)"
                    for _ in clauses
                )
                any_match = " OR ".join(clauses)
                count_sql = (
                    f"SELECT COUNT(*) FROM asr_segments_fts JOIN asr_segments AS s ON s.id=asr_segments_fts.rowid "
                    f"WHERE s.video_id IN ({sub_intersects}) AND asr_segments_fts MATCH ?"
                )
                total = int(conn.execute(count_sql, (*clauses, any_match)).fetchone()[0])
                select_sql = (
                    f"SELECT s.video_name,s.video_id,s.start,s.end,s.text,s.search_text,-bm25(asr_segments_fts) AS relevance "
                    f"FROM asr_segments_fts JOIN asr_segments AS s ON s.id=asr_segments_fts.rowid "
                    f"WHERE s.video_id IN ({sub_intersects}) AND asr_segments_fts MATCH ? "
                    f"ORDER BY relevance DESC,s.video_name COLLATE NOCASE,s.start,s.id "
                    f"LIMIT ? OFFSET ?"
                )
                rows = conn.execute(select_sql, (*clauses, any_match, limit, offset)).fetchall()
                video_count_sql = f"SELECT COUNT(*) FROM ({sub_intersects})"
                videos = int(conn.execute(video_count_sql, clauses).fetchone()[0])

    except sqlite3.DatabaseError as exc:
        log.warning(f"ASR FTS5 query unavailable; using JSON cache: {exc}")
        return None

    matches = [
        (
            {
                "video_name": str(row["video_name"]),
                "video_id": str(row["video_id"]),
            },
            {
                "start": float(row["start"]),
                "end": float(row["end"]),
                "text": str(row["text"]),
                "search_text": str(row["search_text"]),
                "score": float(row["relevance"]),
            },
        )
        for row in rows
    ]
    return {"matches": matches, "total": total, "videos": videos}

@app.post("/asr/search")
def asr_search(body: dict):
    phrase = str(body.get("phrase", "")).strip()
    operator = str(body.get("operator", "and")).strip().lower()
    if operator not in {"and", "or"}:
        operator = "and"
    try:
        limit = int(body.get("limit", body.get("top_k", 100)))
    except (TypeError, ValueError):
        limit = 100
    try:
        offset = int(body.get("offset", 0))
    except (TypeError, ValueError):
        offset = 0
    limit = max(1, min(limit, TOP_K_MAX))
    offset = max(0, offset)

    if not phrase:
        return JSONResponse({"error": "Provide phrase"}, status_code=400)

    indexed_page = _search_asr_fts(phrase, offset=offset, limit=limit, operator=operator)
    if indexed_page is not None:
        selected_matches = indexed_page["matches"]
        total_matches = indexed_page["total"]
        videos_with_current_cache = indexed_page["videos"]
        transcript_source = "asr_fts5"
    else:
        cache = load_asr_cache()
        inventory = _asr_cache_inventory(cache)
        videos_with_current_cache = 0
        all_matches: list[tuple[dict, dict]] = []
        raw_terms = [t.strip() for t in phrase.split(",") if t.strip()] if "," in phrase else [phrase.strip()]
        for cached_video in inventory["entries"]:
            if not cached_video["current"]:
                continue
            segs = cached_video["segments"]
            term_matches = [list(find_phrase_segments(segs, term)) for term in raw_terms]
            if len(raw_terms) > 1 and operator == "and":
                if all(len(tm) > 0 for tm in term_matches):
                    videos_with_current_cache += 1
                    for tm in term_matches:
                        for match in tm:
                            all_matches.append((cached_video, match))
            else:
                matched_segs = []
                for tm in term_matches:
                    matched_segs.extend(tm)
                if matched_segs:
                    videos_with_current_cache += 1
                    for match in matched_segs:
                        all_matches.append((cached_video, match))

        # Match a segment once even when several clauses hit it, like FTS.
        unique_matches = {}
        for cached_video, match in all_matches:
            key = (cached_video["video_name"], match["start"], match["end"], match["text"])
            unique_matches.setdefault(key, (cached_video, match))
        all_matches = sorted(unique_matches.values(), key=lambda pair: (
            pair[0]["video_name"].casefold(), pair[1]["start"], pair[1]["end"], pair[1]["text"]
        ))
        total_matches = len(all_matches)
        selected_matches = all_matches[offset:offset + limit]
        transcript_source = "asr_cache"

    # Match keyframe windows
    results = []
    page_timelines = {}
    for cached_video, match in selected_matches:
        video_name = cached_video["video_name"]
        video_id = cached_video.get("video_id") or Path(video_name).stem
        if video_id not in page_timelines:
            page_timelines[video_id] = _load_asr_keyframe_timeline(video_id)
        timeline, mapping = page_timelines[video_id]
        frames = _frames_in_asr_interval(timeline, match["start"], match["end"])
        results.append({
            "video": video_name,
            "video_id": video_id,
            "video_name": video_name,
            "start": round(match["start"], 2),
            "end": round(match["end"], 2),
            "duration": round(match["end"] - match["start"], 1),
            "text": match["text"],
            "frames": frames,
            "frame_count": len(frames),
            "mapping_mode": mapping.get("mapping_mode"),
        })

    total_frames = sum(item["frame_count"] for item in results)
    return {
        "phrase": phrase,
        "operator": operator,
        "results": results,
        "total": total_matches,
        "total_frames": total_frames,
        "videos_scanned": videos_with_current_cache,
        "transcript_source": transcript_source,
        "offset": offset,
        "limit": limit,
        "next_offset": min(offset + len(results), total_matches),
        "has_more": offset + len(results) < total_matches,
        "videos_folder_exists": _configured_videos_dir().is_dir(),
    }

@app.post("/kis/start")
def kis_start():
    sid = start_session()
    return {"session_id": sid}

@app.post("/kis/turn")
def kis_turn(body: dict):
    session_id = (body.get("session_id") or "").strip()
    message    = (body.get("message") or "").strip()
    try:
        top_k = int(body.get("top_k", 1000))
    except (TypeError, ValueError):
        top_k = 1000
    top_k = max(1, min(top_k, TOP_K_MAX))
    if not session_id or not message:
        return JSONResponse({"error": "Provide session_id and message"}, status_code=400)
    if get_session(session_id) is None:
        return JSONResponse({"error": "Unknown or expired session_id"}, status_code=404)
    try:
        return process_turn(session_id, message, top_k=top_k,
                            collection=get_collection(), metadata_loader=get_meta_by_files)
    except Exception as e:
        log.error(f"KIS turn failed: {e}")
        if ALLOW_PARTIAL_DATA:
            record_issue('conversation.search', str(e))
            return {'session_id': session_id, 'results': [], 'total': 0, 'partial': True,
                    'warnings': ['Search data unavailable; see sever_log.json']}
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/kis/session/{session_id}")
def kis_session_info(session_id: str):
    s = get_session(session_id)
    if s is None:
        return JSONResponse({"error": "Not found"}, status_code=404)
    return {
        "session_id":   s.id,
        "turns":        s.turns,
        "slots_filled": s.slot_values,
        "pool_size":    len(s.pool) if s.pool is not None else None,
    }

@app.post("/kis/end")
def kis_end(body: dict):
    session_id = (body.get("session_id") or "").strip()
    if session_id:
        end_session(session_id)
    return {"status": "ended"}


@app.post("/asr/frames")
def asr_frame_page(body: dict):
    from search_pages import asr_page
    return asr_page(body)


@app.get("/video-match/frames")
def video_frame_page(query: str, page: int = 1):
    from search_pages import video_page
    return video_page(query, page)


@app.get("/kis/results")
def kis_result_page(session_id: str, page: int = 1):
    from search_pages import kis_page
    return kis_page(session_id, page)
