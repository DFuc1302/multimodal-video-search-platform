#!/usr/bin/env python3
"""Prepare an indexed dataset for low-latency, read-only serving.

This command is intentionally an offline finalization step.  It moves work
which used to happen in the FastAPI lifespan (JSON parsing, metadata scans and
mapping construction) out of the server cold-start path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sqlite3
import tempfile
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


SERVE_SCHEMA_VERSION = 2
DEFAULT_OCR_PIPELINE_VERSION = "easyocr-batched-v2"
VIDEO_ID_RE = re.compile(r"L\d+_V\d+", re.IGNORECASE)
FRAME_NUMBER_RE = re.compile(r"(\d+)(?=\.[^.]+$)")


class PrepareError(RuntimeError):
    pass


def stable_faiss_id(fid: str) -> int:
    digest = hashlib.blake2b(fid.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") & 0x7FFFFFFFFFFFFFFF


def _json_list(raw: object) -> list[str]:
    try:
        value = json.loads(str(raw or "[]"))
    except (TypeError, json.JSONDecodeError):
        return []
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item)]


def _atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}


def _ensure_frame_columns(conn: sqlite3.Connection) -> None:
    columns = _columns(conn, "frames")
    additions = {
        "faiss_id": "INTEGER",
        "ocr_status": "TEXT DEFAULT 'legacy'",
        "ocr_version": "TEXT DEFAULT ''",
        "ocr_source_file": "TEXT DEFAULT ''",
        "ocr_content_hash": "TEXT DEFAULT ''",
        "ocr_regions": "TEXT DEFAULT '[]'",
        "ocr_attempts": "INTEGER DEFAULT 0",
        "ocr_error": "TEXT DEFAULT ''",
        "ocr_updated_at": "REAL",
    }
    for name, declaration in additions.items():
        if name not in columns:
            conn.execute(f"ALTER TABLE frames ADD COLUMN {name} {declaration}")


def _backfill_runtime_columns(
    conn: sqlite3.Connection,
    *,
    ocr_version: str,
    batch_size: int,
) -> int:
    updated = 0
    cursor = conn.execute(
        "SELECT rowid, id FROM frames WHERE faiss_id IS NULL ORDER BY rowid"
    )
    while True:
        rows = cursor.fetchmany(batch_size)
        if not rows:
            break
        conn.executemany(
            "UPDATE frames SET faiss_id=? WHERE rowid=?",
            [(stable_faiss_id(str(row[1])), int(row[0])) for row in rows],
        )
        updated += len(rows)

    # Legacy shards remain valid.  Completed text is terminal, has_text=0 was
    # excluded by the old candidate filter, and only has_text=1 + empty OCR is
    # genuinely pending.  Never relabel legacy work as the new pipeline.
    conn.execute(
        """
        UPDATE frames
        SET ocr_status = CASE
            WHEN COALESCE(ocr_joined, '') != '' THEN 'done'
            WHEN COALESCE(has_text, 0) = 0 THEN 'skipped'
            ELSE 'pending'
        END
        WHERE COALESCE(ocr_status, '') NOT IN
              ('pending','processing','done','no_text','skipped','error','reused')
           OR ocr_status IS NULL
        """
    )
    conn.execute(
        "UPDATE frames SET ocr_version='legacy-v0' "
        "WHERE ocr_status IN ('done','no_text','skipped','reused') "
        "AND COALESCE(ocr_version,'')=''"
    )
    conn.execute(
        "UPDATE frames SET ocr_source_file=file "
        "WHERE ocr_status IN ('done','no_text','skipped') "
        "AND COALESCE(ocr_source_file,'')=''"
    )
    conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_frames_faiss_id ON frames(faiss_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ocr_status ON frames(ocr_status)")
    return updated


def _iter_facets(row: sqlite3.Row) -> Iterable[tuple[str, str, str]]:
    filename = str(row["file"])
    for kind, column in (
        ("object", "objects"),
        ("category", "categories"),
        ("color", "colors"),
    ):
        for value in _json_list(row[column]):
            yield filename, kind, value.casefold()
    for value in _video_facet_values(filename):
        yield filename, "video", value


def _video_facet_values(filename: str) -> list[str]:
    normalized = str(filename or "").replace("\\", "/")
    parts = [part for part in normalized.split("/") if part]
    values = {match.group(0).casefold() for match in VIDEO_ID_RE.finditer(normalized)}
    if len(parts) >= 2:
        values.add(parts[-2].casefold())
    return sorted(value for value in values if value)


def _frame_locator(filename: str) -> tuple[str, int, str] | None:
    values = _video_facet_values(filename)
    match = FRAME_NUMBER_RE.search(Path(filename).name)
    if not values or match is None:
        return None
    preferred = next((value for value in values if VIDEO_ID_RE.fullmatch(value)), values[0])
    return preferred, int(match.group(1)), filename


def _build_facets_and_stats(
    conn: sqlite3.Connection, *, batch_size: int
) -> dict:
    conn.executescript(
        """
        DROP TABLE IF EXISTS frame_facets;
        CREATE TABLE frame_facets (
            file  TEXT NOT NULL,
            kind  TEXT NOT NULL,
            value TEXT NOT NULL,
            PRIMARY KEY (file, kind, value)
        ) WITHOUT ROWID;
        CREATE INDEX idx_frame_facets_lookup ON frame_facets(kind, value, file);
        DROP TABLE IF EXISTS frame_locator;
        CREATE TABLE frame_locator (
            video TEXT NOT NULL COLLATE NOCASE,
            n     INTEGER NOT NULL,
            file  TEXT NOT NULL PRIMARY KEY
        ) WITHOUT ROWID;
        CREATE INDEX idx_frame_locator_video_n
            ON frame_locator(video, n, file);
        """
    )

    object_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    color_counts: Counter[str] = Counter()
    object_display: dict[str, str] = {}
    category_display: dict[str, str] = {}
    color_display: dict[str, str] = {}
    facet_batch: list[tuple[str, str, str]] = []
    locator_batch: list[tuple[str, int, str]] = []

    cursor = conn.execute(
        "SELECT file, objects, categories, colors FROM frames ORDER BY rowid"
    )
    for row in cursor:
        values_by_kind = {
            "object": _json_list(row["objects"]),
            "category": _json_list(row["categories"]),
            "color": _json_list(row["colors"]),
            "video": _video_facet_values(str(row["file"])),
        }
        for kind, values in values_by_kind.items():
            for display_value in values:
                normalized = display_value.casefold()
                facet_batch.append((str(row["file"]), kind, normalized))
                if kind == "object":
                    object_counts[normalized] += 1
                    object_display.setdefault(normalized, display_value)
                elif kind == "category":
                    category_counts[normalized] += 1
                    category_display.setdefault(normalized, display_value)
                elif kind == "color":
                    color_counts[normalized] += 1
                    color_display.setdefault(normalized, display_value)
        locator = _frame_locator(str(row["file"]))
        if locator is not None:
            locator_batch.append(locator)
        if len(facet_batch) >= batch_size:
            conn.executemany(
                "INSERT OR IGNORE INTO frame_facets(file,kind,value) VALUES (?,?,?)",
                facet_batch,
            )
            facet_batch.clear()
        if len(locator_batch) >= batch_size:
            conn.executemany(
                "INSERT OR REPLACE INTO frame_locator(video,n,file) VALUES (?,?,?)",
                locator_batch,
            )
            locator_batch.clear()
    if facet_batch:
        conn.executemany(
            "INSERT OR IGNORE INTO frame_facets(file,kind,value) VALUES (?,?,?)",
            facet_batch,
        )
    if locator_batch:
        conn.executemany(
            "INSERT OR REPLACE INTO frame_locator(video,n,file) VALUES (?,?,?)",
            locator_batch,
        )

    def display_counts(counter: Counter[str], names: dict[str, str]) -> dict[str, int]:
        return {names[key]: value for key, value in sorted(counter.items())}

    frame_count = int(conn.execute("SELECT COUNT(*) FROM frames").fetchone()[0])
    has_text_count = int(
        conn.execute("SELECT COUNT(*) FROM frames WHERE COALESCE(has_text,0)=1").fetchone()[0]
    )
    ocr_cached_count = int(
        conn.execute(
            "SELECT COUNT(*) FROM frames WHERE ocr_status IN ('done','reused')"
        ).fetchone()[0]
    )
    pending_ocr = int(
        conn.execute("SELECT COUNT(*) FROM frames WHERE ocr_status='pending'").fetchone()[0]
    )
    processing_ocr = int(
        conn.execute("SELECT COUNT(*) FROM frames WHERE ocr_status='processing'").fetchone()[0]
    )
    error_ocr = int(
        conn.execute("SELECT COUNT(*) FROM frames WHERE ocr_status='error'").fetchone()[0]
    )
    ocr_versions = {
        str(row[0]): int(row[1])
        for row in conn.execute(
            "SELECT ocr_version, COUNT(*) FROM frames "
            "WHERE COALESCE(ocr_version,'')!='' GROUP BY ocr_version"
        )
    }
    return {
        "frames": frame_count,
        "has_text_count": has_text_count,
        "ocr_cached_count": ocr_cached_count,
        "pending_ocr": pending_ocr,
        "processing_ocr": processing_ocr,
        "ocr_errors": error_ocr,
        "ocr_versions": ocr_versions,
        "objects": display_counts(object_counts, object_display),
        "categories": display_counts(category_counts, category_display),
        "colors": display_counts(color_counts, color_display),
    }


def _validate_faiss_mapping(
    conn: sqlite3.Connection, root: Path, *, required: bool
) -> dict:
    index_path = root / "faiss_index" / "frames.faiss"
    if not index_path.exists():
        if required:
            raise PrepareError(f"Missing FAISS index: {index_path}")
        return {"count": None, "dimension": None, "ids_verified": False}
    try:
        import faiss
        import numpy as np

        index = faiss.read_index(str(index_path))
        if not hasattr(index, "id_map"):
            raise PrepareError("FAISS serving index must expose an external id_map")
        index_ids = np.asarray(faiss.vector_to_array(index.id_map), dtype=np.int64)
        db_values = [
            row[0] for row in conn.execute("SELECT faiss_id FROM frames ORDER BY faiss_id")
        ]
        if any(value is None for value in db_values):
            raise PrepareError("SQLite has frames without faiss_id")
        db_ids = np.asarray(db_values, dtype=np.int64)
        if len(index_ids) != int(index.ntotal) or len(db_ids) != int(index.ntotal):
            raise PrepareError(
                "SQLite/FAISS count mismatch: "
                f"SQLite={len(db_ids)}, FAISS={index.ntotal}, id_map={len(index_ids)}"
            )
        if len(np.unique(index_ids)) != len(index_ids):
            raise PrepareError("FAISS contains duplicate external ids")
        if len(np.unique(db_ids)) != len(db_ids):
            raise PrepareError("SQLite contains duplicate faiss_id values")
        if not np.array_equal(np.sort(index_ids), db_ids):
            only_index = np.setdiff1d(index_ids, db_ids)[:3].tolist()
            only_db = np.setdiff1d(db_ids, index_ids)[:3].tolist()
            raise PrepareError(
                "SQLite/FAISS id mapping differs: "
                f"only FAISS={only_index}, only SQLite={only_db}"
            )
        return {
            "count": int(index.ntotal),
            "dimension": int(index.d),
            "ids_verified": True,
        }
    except Exception as exc:
        if isinstance(exc, PrepareError):
            raise
        raise PrepareError(f"Cannot validate FAISS index {index_path}: {exc}") from exc


def _ensure_fts(conn: sqlite3.Connection, *, optimize: bool) -> dict | None:
    tables = {
        row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    columns = _columns(conn, "frames_fts") if "frames_fts" in tables else set()
    needs_rebuild = not columns or not {"file", "ocr_joined", "ocr_ascii"} <= columns
    if columns and needs_rebuild:
        conn.executescript(
            """
            DROP TRIGGER IF EXISTS frames_ai;
            DROP TRIGGER IF EXISTS frames_ad;
            DROP TRIGGER IF EXISTS frames_au;
            DROP TABLE frames_fts;
            """
        )
    conn.executescript(
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS frames_fts USING fts5(
            file UNINDEXED,
            ocr_joined,
            ocr_ascii,
            content='frames',
            content_rowid='rowid'
        );
        CREATE TRIGGER IF NOT EXISTS frames_ai AFTER INSERT ON frames BEGIN
            INSERT INTO frames_fts(rowid, file, ocr_joined, ocr_ascii)
            VALUES (new.rowid, new.file, new.ocr_joined, new.ocr_ascii);
        END;
        CREATE TRIGGER IF NOT EXISTS frames_ad AFTER DELETE ON frames BEGIN
            INSERT INTO frames_fts(frames_fts, rowid, file, ocr_joined, ocr_ascii)
            VALUES ('delete', old.rowid, old.file, old.ocr_joined, old.ocr_ascii);
        END;
        DROP TRIGGER IF EXISTS frames_au;
        CREATE TRIGGER frames_au
        AFTER UPDATE OF file, ocr_joined, ocr_ascii ON frames BEGIN
            INSERT INTO frames_fts(frames_fts, rowid, file, ocr_joined, ocr_ascii)
            VALUES ('delete', old.rowid, old.file, old.ocr_joined, old.ocr_ascii);
            INSERT INTO frames_fts(rowid, file, ocr_joined, ocr_ascii)
            VALUES (new.rowid, new.file, new.ocr_joined, new.ocr_ascii);
        END;
        """
    )
    try:
        conn.execute(
            "INSERT INTO frames_fts(frames_fts, rank) VALUES('integrity-check', 1)"
        )
    except sqlite3.DatabaseError:
        conn.execute("INSERT INTO frames_fts(frames_fts) VALUES('rebuild')")
        conn.execute(
            "INSERT INTO frames_fts(frames_fts, rank) VALUES('integrity-check', 1)"
        )
    if optimize:
        conn.execute("INSERT INTO frames_fts(frames_fts) VALUES('optimize')")

    for row in conn.execute(
        "SELECT rowid, file, ocr_joined, ocr_ascii FROM frames "
        "WHERE COALESCE(ocr_joined,'')!='' OR COALESCE(ocr_ascii,'')!='' LIMIT 100"
    ):
        for source in (row[2], row[3]):
            tokens = [token for token in re.findall(r"\w+", str(source or "").casefold()) if token]
            for token in tokens:
                found = conn.execute(
                    "SELECT 1 FROM frames_fts "
                    "WHERE frames_fts MATCH ? AND rowid=? LIMIT 1",
                    (f'"{token}"', int(row[0])),
                ).fetchone()
                if found:
                    return {"term": token, "rowid": int(row[0]), "file": str(row[1])}
    return None


def _build_asr_fts(
    conn: sqlite3.Connection,
    cache_path: Path,
    *,
    batch_size: int,
    optimize: bool,
) -> dict:
    """Materialize portable 2-second ASR windows into local SQLite FTS5."""
    conn.executescript(
        """
        DROP TABLE IF EXISTS asr_segments_fts;
        DROP TABLE IF EXISTS asr_segments;
        CREATE TABLE asr_segments (
            id            INTEGER PRIMARY KEY,
            video_name    TEXT NOT NULL,
            video_id      TEXT NOT NULL COLLATE NOCASE,
            segment_index INTEGER NOT NULL,
            start         REAL NOT NULL,
            end           REAL NOT NULL,
            text          TEXT NOT NULL,
            search_text   TEXT NOT NULL,
            UNIQUE(video_name, segment_index)
        );
        CREATE INDEX idx_asr_segments_video_start
            ON asr_segments(video_id, start, id);
        CREATE VIRTUAL TABLE asr_segments_fts USING fts5(
            search_text,
            content='asr_segments',
            content_rowid='id',
            tokenize='unicode61 remove_diacritics 2'
        );
        """
    )
    if not cache_path.is_file():
        return {"rows": 0, "videos": 0}

    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PrepareError(f"Cannot read ASR cache {cache_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise PrepareError(f"ASR cache must contain an object: {cache_path}")

    from asr_index import (
        asr_cache_video_name,
        is_asr_excluded_video,
        is_current_asr_segments,
        normalize_asr_text,
    )

    pending: list[tuple[str, str, int, float, float, str, str]] = []
    videos: set[str] = set()
    row_count = 0
    for raw_key, segments in payload.items():
        if not is_current_asr_segments(segments):
            continue
        raw_name = asr_cache_video_name(raw_key).replace("\\", "/")
        video_name = raw_name.rsplit("/", 1)[-1].strip()
        if is_asr_excluded_video(video_name):
            continue
        if not video_name:
            continue
        video_id = Path(video_name).stem
        videos.add(video_id.casefold())
        for segment_index, segment in enumerate(segments):
            display_text = " ".join(str(segment.get("text", "")).split())
            search_text = normalize_asr_text(
                segment.get("search_text") or display_text
            )
            pending.append((
                video_name,
                video_id,
                segment_index,
                float(segment.get("start", 0.0)),
                float(segment.get("end", 0.0)),
                display_text,
                search_text,
            ))
            if len(pending) >= batch_size:
                conn.executemany(
                    "INSERT INTO asr_segments("
                    "video_name,video_id,segment_index,start,end,text,search_text"
                    ") VALUES (?,?,?,?,?,?,?)",
                    pending,
                )
                row_count += len(pending)
                pending.clear()
    if pending:
        conn.executemany(
            "INSERT INTO asr_segments("
            "video_name,video_id,segment_index,start,end,text,search_text"
            ") VALUES (?,?,?,?,?,?,?)",
            pending,
        )
        row_count += len(pending)

    conn.execute(
        "INSERT INTO asr_segments_fts(rowid,search_text) "
        "SELECT id,search_text FROM asr_segments"
    )
    conn.execute(
        "INSERT INTO asr_segments_fts(asr_segments_fts, rank) "
        "VALUES('integrity-check', 1)"
    )
    if optimize and row_count:
        conn.execute(
            "INSERT INTO asr_segments_fts(asr_segments_fts) VALUES('optimize')"
        )
    return {"rows": row_count, "videos": len(videos)}


def export_clip_text_artifact(root: Path, model_name: str) -> Path:
    """Export only the compatible CLIP text tower into the serving bundle."""
    target = root / "clip_text"
    if (target / "config.json").is_file():
        required = {
            "config.json",
            "tokenizer_config.json",
            "source_model.json",
        }
        missing = sorted(name for name in required if not (target / name).is_file())
        has_weights = any(
            (target / name).is_file()
            for name in ("model.safetensors", "pytorch_model.bin")
        )
        if missing or not has_weights:
            raise PrepareError(
                f"Incomplete CLIP text artifact {target}; missing={missing}, "
                f"weights={has_weights}"
            )
        source = json.loads((target / "source_model.json").read_text(encoding="utf-8"))
        if source.get("source_model") != model_name:
            raise PrepareError(
                f"CLIP text artifact model mismatch: "
                f"{source.get('source_model')} != {model_name}"
            )
        return target

    import torch
    from transformers import AutoTokenizer, CLIPTextModelWithProjection

    temporary = Path(tempfile.mkdtemp(prefix=".clip_text-", dir=root))
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = CLIPTextModelWithProjection.from_pretrained(
            model_name, torch_dtype=torch.float16
        )
        tokenizer.save_pretrained(temporary)
        model.save_pretrained(temporary, safe_serialization=True)
        (temporary / "source_model.json").write_text(
            json.dumps(
                {
                    "source_model": model_name,
                    "exported_at": datetime.now(timezone.utc).isoformat(
                        timespec="seconds"
                    ),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        if target.exists():
            raise PrepareError(
                f"Incomplete CLIP text artifact already exists: {target}"
            )
        temporary.replace(target)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return target


def prepare_dataset(
    root: Path,
    *,
    strict_ocr: bool = True,
    batch_size: int = 10_000,
    ocr_version: str = DEFAULT_OCR_PIPELINE_VERSION,
    optimize_fts: bool = True,
    export_clip_text: bool = True,
    clip_model: str = "openai/clip-vit-large-patch14",
    require_faiss: bool = True,
) -> dict:
    root = root.resolve()
    db_path = root / "meta.db"
    if not db_path.is_file():
        raise PrepareError(f"Missing metadata database: {db_path}")

    started = time.perf_counter()
    conn = sqlite3.connect(db_path, timeout=60.0)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA busy_timeout=60000")
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        if "frames" not in {
            row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }:
            raise PrepareError(f"Database has no frames table: {db_path}")

        _ensure_frame_columns(conn)
        mapped = _backfill_runtime_columns(
            conn, ocr_version=ocr_version, batch_size=batch_size
        )
        stats = _build_facets_and_stats(conn, batch_size=batch_size)
        fts_probe = _ensure_fts(conn, optimize=optimize_fts)
        asr_fts = _build_asr_fts(
            conn,
            root / "asr_cache.json",
            batch_size=batch_size,
            optimize=optimize_fts,
        )
        faiss_info = _validate_faiss_mapping(conn, root, required=require_faiss)
        faiss_count = faiss_info["count"]
        if strict_ocr and (
            stats["pending_ocr"]
            or stats["processing_ocr"]
            or stats["ocr_errors"]
        ):
            raise PrepareError(
                "Dataset is not ready to serve: "
                f"pending OCR={stats['pending_ocr']}, "
                f"processing OCR={stats['processing_ocr']}, "
                f"OCR errors={stats['ocr_errors']}"
            )

        conn.execute(
            "CREATE TABLE IF NOT EXISTS serve_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        observed_versions = sorted(stats["ocr_versions"])
        reported_ocr_version = (
            observed_versions[0]
            if len(observed_versions) == 1
            else ("mixed" if observed_versions else ocr_version)
        )
        serve_values = {
            "schema_version": str(SERVE_SCHEMA_VERSION),
            "generated_at": generated_at,
            "ocr_pipeline_version": reported_ocr_version,
            "frame_count": str(stats["frames"]),
            "asr_segment_count": str(asr_fts["rows"]),
            "asr_video_count": str(asr_fts["videos"]),
        }
        conn.executemany(
            "INSERT OR REPLACE INTO serve_meta(key,value) VALUES (?,?)",
            serve_values.items(),
        )
        conn.execute("PRAGMA optimize")
        conn.commit()
        try:
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except sqlite3.OperationalError:
            pass
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    clip_text_path = (
        export_clip_text_artifact(root, clip_model) if export_clip_text else None
    )
    elapsed = time.perf_counter() - started
    stats_payload = {
        "format_version": SERVE_SCHEMA_VERSION,
        "generated_at": generated_at,
        **stats,
    }
    manifest = {
        "format_version": SERVE_SCHEMA_VERSION,
        "generated_at": generated_at,
        "dataset_root": ".",
        "frame_count": stats["frames"],
        "faiss_count": faiss_count,
        "faiss_dimension": faiss_info["dimension"],
        "faiss_ids_verified": faiss_info["ids_verified"],
        "pending_ocr": stats["pending_ocr"],
        "processing_ocr": stats["processing_ocr"],
        "ocr_errors": stats["ocr_errors"],
        "ocr_pipeline_version": reported_ocr_version,
        "ocr_versions": stats["ocr_versions"],
        "ocr_cached_count": stats["ocr_cached_count"],
        "clip_model": clip_model,
        "clip_text_artifact": (
            str(clip_text_path.relative_to(root)) if clip_text_path else None
        ),
        "faiss_mapping": "meta.db:frames.faiss_id",
        "facet_index": "meta.db:frame_facets",
        "frame_locator": "meta.db:frame_locator",
        "asr_fts": "meta.db:asr_segments_fts",
        "asr_segment_count": asr_fts["rows"],
        "asr_video_count": asr_fts["videos"],
        "fts_probe": fts_probe,
        "elapsed_seconds": round(elapsed, 3),
        "mapped_rows": mapped,
    }
    _atomic_json(root / "serve_stats.json", stats_payload)
    _atomic_json(root / "serve_manifest.json", manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default=os.getenv("DATA_ROOT", "."))
    parser.add_argument("--batch-size", type=int, default=10_000)
    parser.add_argument(
        "--ocr-version",
        default=os.getenv("OCR_PIPELINE_VERSION", DEFAULT_OCR_PIPELINE_VERSION),
    )
    parser.add_argument(
        "--allow-pending-ocr",
        action="store_true",
        help="Build artifacts even when pending/error OCR rows remain",
    )
    parser.add_argument("--skip-fts-optimize", action="store_true")
    parser.add_argument("--skip-clip-text-export", action="store_true")
    parser.add_argument(
        "--clip-model",
        default=os.getenv("CLIP_MODEL", "openai/clip-vit-large-patch14"),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be greater than zero")
    try:
        result = prepare_dataset(
            Path(args.data_root),
            strict_ocr=not args.allow_pending_ocr,
            batch_size=args.batch_size,
            ocr_version=args.ocr_version,
            optimize_fts=not args.skip_fts_optimize,
            export_clip_text=not args.skip_clip_text_export,
            clip_model=args.clip_model,
        )
    except (PrepareError, sqlite3.Error) as exc:
        print(f"ERROR: {exc}")
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
