"""Local pseudonymous retrieval logs. No IP, cookies, raw query or answer text."""
import hashlib
import hmac
import json
import logging
import os
import re
import sqlite3
from redis_store import ENABLED as REDIS_ENABLED, store as redis_store
import threading
import time
import unicodedata
import uuid
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.routing import APIRoute
from starlette.concurrency import run_in_threadpool

TEXT_FIELDS = {"query", "ocr_query", "phrase", "message", "objects", "categories", "colors", "video", "file", "category", "domain", "session_id"}
ENUM_FIELDS = {"query_operator", "ocr_operator", "operator", "search_mode", "video_mode", "mode"}
NUMBER_FIELDS = {"min_conf", "top_k", "offset", "limit", "top_videos", "frames_per_video", "candidate_frames"}
RETRIEVAL_PATHS = {"/asr/frames", "/video-match/frames", "/summary/search", "/kis/results", "/search", "/asr/search", "/video-match", "/similar", "/kis/turn"}


def valid_id(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{32}", value) is not None


def portable_file(value):
    value = str(value).replace("\\", "/")
    parts = value.split("/")
    # Strip local storage prefixes by accepting only relative dataset paths.
    if value.startswith("/") or ":" in value or any(p in {"", ".", ".."} for p in parts):
        return None
    return value[:500]


class SearchLogger:
    def __init__(self, directory, *, enabled=True, max_bytes=20_000_000):
        self.directory = Path(directory)
        self.enabled, self.max_bytes = enabled, max_bytes
        self.lock = threading.Lock()
        self.searches = OrderedDict()
        self.last_error = None
        self.key = None
        source_root = Path(__file__).resolve().parents[1]
        revision = hashlib.sha256()
        for name in ("main.py", "shared.py", "vlm_reranker.py", "search_telemetry.py"):
            revision.update((source_root / name).read_bytes())
        self.configuration = {"code_revision": revision.hexdigest(),
                              "dedup_frame_gap": os.getenv("SEARCH_DEDUP_FRAME_GAP", "3"),
                              "asr_boost": os.getenv("SEARCH_ASR_BOOST", "0"),
                              "vlm_top_n": os.getenv("VLM_TOP_N", "12"),
                              "vlm_weight": os.getenv("VLM_WEIGHT", ".65"),
                              "vlm_timeout_ms": os.getenv("VLM_TIMEOUT_MS", "3000"),
                              "vlm_model": hashlib.sha256(os.getenv("VLM_MODEL_PATH", "").encode()).hexdigest(),
                              "vlm_device": hashlib.sha256(os.getenv("VLM_DEVICE", "cpu").encode()).hexdigest(),
                              "clip_model": hashlib.sha256(os.getenv("CLIP_MODEL", "openai/clip-vit-large-patch14").encode()).hexdigest(),
                              "federated_config": None,
                              "dataset_manifest": None}
        manifest = Path(os.getenv("DATA_ROOT", str(source_root / "data"))) / "serve_manifest.json"
        if manifest.is_file():
            self.configuration["dataset_manifest"] = hashlib.sha256(manifest.read_bytes()).hexdigest()
        federation = os.getenv("FEDERATED_DATASET_CONFIG", "")
        if federation and Path(federation).is_file():
            self.configuration["federated_config"] = hashlib.sha256(Path(federation).read_bytes()).hexdigest()
        model_config = Path(os.getenv("VLM_MODEL_PATH", ".")) / "config.json"
        if os.getenv("VLM_MODEL_PATH") and model_config.is_file():
            self.configuration["vlm_model_config"] = hashlib.sha256(model_config.read_bytes()).hexdigest()
        self.configuration_id = hashlib.sha256(json.dumps(self.configuration, sort_keys=True).encode()).hexdigest()

    def _initialize(self):
        if REDIS_ENABLED:
            db=redis_store();key=db.prefix+'telemetry:hmac'
            db.r.set(key,os.urandom(32),nx=True)
            self.key=db.r.get(key)
            return
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        key_path = self.directory / ".hmac-key"
        try:
            with key_path.open("xb") as handle:
                os.chmod(key_path, 0o600)
                handle.write(os.urandom(32))
        except FileExistsError:
            pass
        self.key = key_path.read_bytes()
        if len(self.key) != 32:
            raise ValueError("Invalid telemetry key")
        path = self.directory / ".receipts.db"
        with sqlite3.connect(path) as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS receipts (id INTEGER PRIMARY KEY, search_id TEXT UNIQUE, session_id TEXT, query_id TEXT)")
        os.chmod(path, 0o600)

    def digest(self, value):
        return hmac.new(self.key, str(value).encode("utf-8"), hashlib.sha256).hexdigest()

    def private_text(self, value):
        text = " ".join(unicodedata.normalize("NFC", str(value)).casefold().split())
        return {"hash": self.digest(text), "length": len(text)}

    def _append(self, event):
        if REDIS_ENABLED:
            db=redis_store()
            db.r.xadd(db.prefix+'telemetry:events',{'event':json.dumps(dict(schema_version=1,
                at=datetime.now(timezone.utc).isoformat(),**event),ensure_ascii=False,allow_nan=False)})
            return
        name = datetime.now(timezone.utc).strftime("search-%Y-%m-%d")
        path = self.directory / (name+".jsonl")
        n = 0
        while path.exists() and path.stat().st_size >= self.max_bytes:
            n += 1
            path = self.directory / f"{name}.{n}.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            os.chmod(path, 0o600)
            handle.write(json.dumps(dict(schema_version=1, at=datetime.now(timezone.utc).isoformat(),
                                         **event), ensure_ascii=False, allow_nan=False)+"\n")

    def _guarded(self, operation):
        if not self.enabled:
            return None
        try:
            with self.lock:
                if REDIS_ENABLED:
                    if self.key is None:self._initialize()
                    result=operation();self.last_error=None
                    return result
                self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
                # Serialize key creation and JSONL writes across server workers.
                import fcntl
                with (self.directory / ".writer-lock").open("a") as lock:
                    fcntl.flock(lock, fcntl.LOCK_EX)
                    if self.key is None:
                        self._initialize()
                    result = operation()
                    self.last_error = None
                    return result
        except Exception as exc:
            self.last_error = type(exc).__name__
            logging.getLogger(__name__).warning("Search log unavailable (%s)", self.last_error)
            return None

    def retrieval(self, endpoint, params, payload, status, latency_ms, session):
        def write():
            filters = {}
            for key in sorted(TEXT_FIELDS | ENUM_FIELDS | NUMBER_FIELDS | {"rerank", "timeline"}):
                if key not in params:
                    continue
                value = params[key]
                if key in TEXT_FIELDS:
                    filters[key] = self.private_text(value)
                elif key in ENUM_FIELDS:
                    # Values outside recognized enums cannot smuggle text into logs.
                    allowed = {"and", "or", "hybrid", "ocr_first", "ocr", "clip", "include", "exclude", "summary", "combined"}
                    filters[key] = str(value) if str(value) in allowed else "invalid"
                elif key in NUMBER_FIELDS:
                    try:
                        number = float(value)
                        filters[key] = number if abs(number) < 1e12 else None
                    except (ValueError, TypeError):
                        filters[key] = None
                else:
                    filters[key] = str(value).lower() in {"true", "1"}
            identity = {k: v for k, v in filters.items() if k not in {"offset", "limit", "top_k"}}
            query_id = self.digest(json.dumps([endpoint, identity], sort_keys=True))
            search_id = uuid.uuid4().hex
            session_id = session if valid_id(session) else uuid.uuid4().hex
            results = payload.get("results", [])
            unit = "frame"
            if endpoint == "/video-match":
                unit = "video"
                ranked = [portable_file(v.get("video", "")) for v in payload.get("videos", [])]
            elif endpoint == "/asr/search":
                unit = "segment"
                ranked = [f'{r.get("video_id", r.get("video"))}@{r["start"]:.2f}-{r["end"]:.2f}' for r in results]
            elif endpoint == '/asr/frames':
                ranked = [portable_file(frame.get('file', '')) for row in results for frame in row.get('frames', [])]
            elif endpoint == '/video-match/frames':
                ranked = [portable_file(frame.get('file', '')) for video in payload.get('videos', []) for frame in video.get('frames', [])]
            else:
                ranked = [portable_file(r.get("file", "")) for r in results]
            rank_offset = payload.get('offset')
            if rank_offset is None:
                rank_offset = max(0, int(payload.get('page', 1)) - 1) * max(1, int(payload.get('page_size', 1000)))
            event = {"event": "search", "search_id": search_id, "query_id": query_id,
                     "session_id": session_id, "endpoint": endpoint, "filters": filters,
                     "configuration_id": self.configuration_id, "configuration": self.configuration,
                     "status": status, "latency_ms": round(latency_ms, 3), "unit": unit,
                     "ranking": [r for r in ranked if r], "total": payload.get("total", payload.get("total_videos", 0)),
                     "rank_offset": rank_offset,
                     "reranking": payload.get("reranking", {"status": "disabled"})}
            self._append(event)
            link = {k: event[k] for k in ("search_id", "query_id", "session_id")}
            if REDIS_ENABLED:
                redis_store().put_json('receipt:'+search_id,link,ttl=86400*30)
            else:
                with sqlite3.connect(self.directory / ".receipts.db") as conn:
                    conn.execute("INSERT INTO receipts(search_id,session_id,query_id) VALUES (?,?,?)",
                                 (search_id, session_id, query_id))
                    conn.execute("DELETE FROM receipts WHERE id < (SELECT MAX(id)-10000 FROM receipts)")
            self.searches[search_id] = link
            while len(self.searches) > 10000:
                self.searches.popitem(last=False)
            return link
        return self._guarded(write)

    def feedback(self, body, session):
        def write():
            # Link only to a search from this session, never treat a choice as truth.
            search_id = body.get("search_id")
            if not valid_id(search_id):
                return {"accepted": False, "reason": "unknown_search"}
            link = self.searches.get(search_id)
            if link is None:
                if REDIS_ENABLED:
                    link=redis_store().json('receipt:'+search_id)
                else:
                    with sqlite3.connect(self.directory / ".receipts.db") as conn:
                        row = conn.execute("SELECT session_id,query_id FROM receipts WHERE search_id=?", (search_id,)).fetchone()
                    if row:
                        link = {"search_id": search_id, "session_id": row[0], "query_id": row[1]}
            if not link or session != link["session_id"]:
                return {"accepted": False, "reason": "unknown_search"}
            kind = body.get("event")
            if kind not in {"shortlist", "answer"}:
                return {"accepted": False, "reason": "invalid_event"}
            frames = body.get("frames", [])
            if not isinstance(frames, list) or len(frames) > 100:
                return {"accepted": False, "reason": "invalid_frames"}
            mode = body.get("task_mode")
            self._append({**link, "event": kind, "task_mode": mode if mode in {"kis", "vqa", "temporal"} else "unknown",
                          "frames": [{"file": file,
                                      "frame_idx": int(f["frameIdx"]) if str(f.get("frameIdx", "")).isdigit() and len(str(f["frameIdx"])) < 12 else None,
                                      "mapped": f.get("mapped") is True}
                                     for f in frames if isinstance(f, dict)
                                     and (file := portable_file(f.get("file", "")))],
                          "answer": self.private_text(body.get("answer", "")),
                          "question": self.private_text(body.get("question", ""))})
            return {"accepted": True}
        return self._guarded(write)


logger = SearchLogger(os.getenv("SEARCH_LOG_DIR", str(Path(__file__).resolve().parents[1] / "runtime_logs")),
                      enabled=os.getenv("SEARCH_LOG_ENABLED", "1").lower() in {"1", "true", "yes"})


class LoggedSearchRoute(APIRoute):
    def get_route_handler(self):
        original = super().get_route_handler()
        async def handler(request: Request):
            if request.url.path not in RETRIEVAL_PATHS or not logger.enabled:
                return await original(request)
            started = time.perf_counter()
            params = dict(request.query_params)
            if request.method == "POST":
                try:
                    body = await request.json()
                    if isinstance(body, dict):
                        params.update(body)
                except (ValueError, UnicodeError):
                    pass
            try:
                response = await original(request)
            except Exception as exc:
                status = 422 if isinstance(exc, RequestValidationError) else getattr(exc, "status_code", 500)
                await run_in_threadpool(logger.retrieval, request.url.path, params, {}, status,
                                        (time.perf_counter()-started)*1000, request.headers.get("X-Search-Session"))
                raise
            try:
                payload = json.loads(response.body)
                if not isinstance(payload, dict):
                    payload = {}
            except (ValueError, AttributeError):
                payload = {}
            receipt = await run_in_threadpool(logger.retrieval, request.url.path, params, payload,
                                             response.status_code, (time.perf_counter()-started)*1000,
                                             request.headers.get("X-Search-Session"))
            if receipt:
                response.headers["X-Search-Id"] = receipt["search_id"]
                response.headers["X-Query-Id"] = receipt["query_id"]
                response.headers["X-Search-Session"] = receipt["session_id"]
            return response
        return handler


router = APIRouter()


@router.post("/evaluation/event")
async def evaluation_event(request: Request):
    if len(await request.body()) > 32_768:
        return JSONResponse({"error": "Event too large"}, status_code=413)
    try:
        body = await request.json()
        if not isinstance(body, dict):
            raise ValueError()
    except ValueError:
        return JSONResponse({"error": "Invalid event"}, status_code=400)
    result = await run_in_threadpool(logger.feedback, body, request.headers.get("X-Search-Session"))
    return result or {"accepted": False, "reason": "logging_unavailable"}


@router.get("/evaluation/status")
def evaluation_status():
    return {"enabled": logger.enabled, "healthy": logger.last_error is None,
            "text_storage": "hmac_sha256", "ground_truth_required": True}
