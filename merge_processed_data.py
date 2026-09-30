"""Merge preprocessing results produced by multiple machines.

Each source directory may be either a complete visual shard or an ASR-only
shard. A complete visual shard contains::

    keyframes/                 extracted images
    map-keyframes/             timestamp CSV files
    meta.db                    SQLite metadata/OCR
    faiss_index/frames.faiss   CLIP vectors
    faiss_index/frames_meta.json
    asr_cache.json             optional Whisper output

An ASR-only shard contains a non-empty ``asr_cache.json``. At least one
complete visual shard is required because the published dataset must always
have SQLite metadata, keyframes, timestamp maps, and a FAISS index.

The merger reconstructs the already-computed vectors from every FAISS shard
and inserts them into one new index.  It does not run CLIP, YOLO, OCR, or ASR
again.  The output is built in a temporary sibling directory and published
only after every validation succeeds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sqlite3
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable


FRAME_COLUMNS = (
    "id",
    "file",
    "mtime",
    "width",
    "height",
    "objects",
    "categories",
    "colors",
    "has_text",
    "ocr_texts",
    "ocr_joined",
    "ocr_ascii",
    "detections",
    "faiss_id",
    "ocr_status",
    "ocr_version",
    "ocr_source_file",
    "ocr_content_hash",
    "ocr_regions",
    "ocr_attempts",
    "ocr_error",
    "ocr_updated_at",
)
REQUIRED_FRAME_COLUMNS = {"id", "file", "mtime"}
ASR_KEY_PREFIX = "video-name-v1::"


class MergeError(RuntimeError):
    """A validation or merge failure that should be shown without traceback."""


@dataclass
class Shard:
    root: Path
    db_path: Path
    index_path: Path
    vector_meta_path: Path
    vector_meta: dict
    frame_rows: dict[str, dict]

    @property
    def vector_rows(self) -> dict[str, dict]:
        return self.vector_meta["fid_to_meta"]


@dataclass
class MergeSource:
    """One input folder, containing a complete visual shard and/or ASR."""

    root: Path
    visual: Shard | None
    asr_cache_path: Path | None

    @property
    def kind(self) -> str:
        return "visual+asr" if self.visual and self.asr_cache_path else (
            "visual" if self.visual else "asr-only"
        )

def normalize_rel_path(value: str) -> str:
    """Return a safe, platform-independent relative path."""
    raw = str(value or "").replace("\\", "/")
    path = PurePosixPath(raw)
    if not raw or path.is_absolute() or ".." in path.parts:
        raise MergeError(f"Invalid relative path in shard metadata: {value!r}")
    normalized = str(path)
    while normalized.startswith("./"):
        normalized = normalized[2:]
    if not normalized or normalized == ".":
        raise MergeError(f"Invalid relative path in shard metadata: {value!r}")
    return normalized


def stable_faiss_id(fid: str) -> int:
    digest = hashlib.blake2b(fid.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") & 0x7FFFFFFFFFFFFFFF


def frame_id(rel_path: str) -> str:
    return normalize_rel_path(rel_path).replace("/", "__").replace(".", "_")


def portable_asr_key(raw_key: str) -> str:
    """Convert old absolute-path ASR keys to a machine-independent key."""
    key = str(raw_key or "")
    if key.startswith(ASR_KEY_PREFIX):
        name = key[len(ASR_KEY_PREFIX):]
    else:
        # Legacy form: /absolute/path/video.mp4::<mtime>.  rsplit also works
        # when a Windows drive letter occurs in the path.
        path_part = key.rsplit("::", 1)[0].replace("\\", "/")
        name = PurePosixPath(path_part).name
    if not name:
        raise MergeError(f"Invalid ASR cache key: {raw_key!r}")
    return f"{ASR_KEY_PREFIX}{name}"


def _load_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MergeError(f"Cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise MergeError(f"Expected a JSON object in {path}")
    return value

def _load_asr_entries(cache_path: Path) -> dict[str, list]:
    """Load and normalize one portable ASR contribution."""
    cache = _load_json(cache_path)
    normalized: dict[str, list] = {}
    for raw_key, segments in cache.items():
        if not isinstance(segments, list):
            raise MergeError(
                f"{cache_path}: ASR value for {raw_key!r} is not a list"
            )
        key = portable_asr_key(raw_key)
        if key in normalized and normalized[key] != segments:
            raise MergeError(
                f"{cache_path}: conflicting ASR entries normalize to {key}"
            )
        normalized[key] = segments
    if not normalized:
        raise MergeError(f"ASR cache is empty: {cache_path}")
    return normalized


VISUAL_ARTIFACT_PATHS = (
    "meta.db", "faiss_index/frames.faiss", "faiss_index/frames_meta.json",
    "keyframes", "map-keyframes",
)

def _load_frame_rows(db_path: Path) -> dict[str, dict]:
    try:
        conn = sqlite3.connect(f"file:{db_path.resolve()}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        columns = {
            row[1] for row in conn.execute("PRAGMA table_info(frames)").fetchall()
        }
        missing = REQUIRED_FRAME_COLUMNS - columns
        if missing:
            raise MergeError(
                f"{db_path} is missing frames columns: {', '.join(sorted(missing))}"
            )

        select_parts = [
            name if name in columns else f"NULL AS {name}"
            for name in FRAME_COLUMNS
        ]
        rows: dict[str, dict] = {}
        files: dict[str, str] = {}
        for sqlite_row in conn.execute(
            f"SELECT {', '.join(select_parts)} FROM frames"
        ):
            row = dict(sqlite_row)
            rel = normalize_rel_path(row["file"])
            fid = str(row["id"] or frame_id(rel))
            expected_fid = frame_id(rel)
            if fid != expected_fid:
                raise MergeError(
                    f"{db_path}: id {fid!r} does not match file {rel!r} "
                    f"(expected {expected_fid!r})"
                )
            if fid in rows:
                raise MergeError(f"{db_path}: duplicate frame id {fid!r}")
            if rel in files:
                raise MergeError(f"{db_path}: duplicate frame file {rel!r}")
            row["id"] = fid
            row["file"] = rel
            valid_statuses = {
                "pending", "processing", "done", "no_text",
                "skipped", "error", "reused",
            }
            if row.get("ocr_status") not in valid_statuses:
                if str(row.get("ocr_joined") or ""):
                    row["ocr_status"] = "done"
                elif int(row.get("has_text") or 0):
                    row["ocr_status"] = "pending"
                else:
                    row["ocr_status"] = "skipped"
            if not row.get("ocr_version") and row["ocr_status"] in {
                "done", "no_text", "skipped", "reused",
            }:
                row["ocr_version"] = "legacy-v0"
            if not row.get("ocr_source_file") and row["ocr_status"] in {
                "done", "no_text", "skipped",
            }:
                row["ocr_source_file"] = rel
            row["faiss_id"] = int(row.get("faiss_id") or stable_faiss_id(fid))
            row["ocr_content_hash"] = row.get("ocr_content_hash") or ""
            row["ocr_regions"] = row.get("ocr_regions") or "[]"
            row["ocr_attempts"] = int(row.get("ocr_attempts") or 0)
            row["ocr_error"] = row.get("ocr_error") or ""
            rows[fid] = row
            files[rel] = fid
        return rows
    except sqlite3.Error as exc:
        raise MergeError(f"Cannot read SQLite database {db_path}: {exc}") from exc
    finally:
        if "conn" in locals():
            conn.close()


def load_shard(root: Path, *, validate_assets: bool = True) -> Shard:
    root = root.resolve()
    if not root.is_dir():
        raise MergeError(f"Shard directory does not exist: {root}")

    db_path = root / "meta.db"
    index_path = root / "faiss_index" / "frames.faiss"
    vector_meta_path = root / "faiss_index" / "frames_meta.json"
    keyframes_dir = root / "keyframes"
    for required in (db_path, index_path, vector_meta_path, keyframes_dir):
        if not required.exists():
            raise MergeError(f"Shard is missing required artifact: {required}")

    vector_meta = _load_json(vector_meta_path)
    vector_rows = vector_meta.get("fid_to_meta")
    dim = vector_meta.get("dim")
    if not isinstance(vector_rows, dict) or not vector_rows:
        raise MergeError(f"No vector metadata found in {vector_meta_path}")
    if not isinstance(dim, int) or dim <= 0:
        raise MergeError(f"Invalid vector dimension in {vector_meta_path}: {dim!r}")

    normalized_vectors: dict[str, dict] = {}
    vector_files: dict[str, str] = {}
    for fid, meta in vector_rows.items():
        if not isinstance(meta, dict):
            raise MergeError(f"{vector_meta_path}: invalid metadata for {fid!r}")
        rel = normalize_rel_path(meta.get("file", ""))
        expected_fid = frame_id(rel)
        if fid != expected_fid:
            raise MergeError(
                f"{vector_meta_path}: id {fid!r} does not match file {rel!r}"
            )
        try:
            faiss_id = int(meta["faiss_id"])
        except (KeyError, TypeError, ValueError) as exc:
            raise MergeError(
                f"{vector_meta_path}: invalid faiss_id for {fid!r}"
            ) from exc
        expected_faiss_id = stable_faiss_id(fid)
        if faiss_id != expected_faiss_id:
            raise MergeError(
                f"{vector_meta_path}: faiss_id mismatch for {fid!r}; "
                f"expected {expected_faiss_id}, got {faiss_id}"
            )
        if rel in vector_files:
            raise MergeError(f"{vector_meta_path}: duplicate vector file {rel!r}")
        normalized_vectors[fid] = {
            "faiss_id": faiss_id,
            "file": rel,
            "mtime": meta.get("mtime", 0),
        }
        vector_files[rel] = fid
    vector_meta["fid_to_meta"] = normalized_vectors

    frame_rows = _load_frame_rows(db_path)
    db_ids = set(frame_rows)
    vector_ids = set(normalized_vectors)
    if db_ids != vector_ids:
        only_db = sorted(db_ids - vector_ids)
        only_vector = sorted(vector_ids - db_ids)
        raise MergeError(
            f"Shard {root} is incomplete: SQLite={len(db_ids)}, "
            f"FAISS metadata={len(vector_ids)}; "
            f"only SQLite={only_db[:3]}, only FAISS={only_vector[:3]}"
        )
    mismatched_ids = [
        fid
        for fid, metadata in normalized_vectors.items()
        if int(frame_rows[fid]["faiss_id"]) != int(metadata["faiss_id"])
    ]
    if mismatched_ids:
        raise MergeError(
            f"Shard {root} has SQLite/sidecar faiss_id mismatches, "
            f"for example: {mismatched_ids[:3]}"
        )

    csv_dir = root / "map-keyframes"
    if not csv_dir.is_dir():
        raise MergeError(f"Shard {root} is missing map-keyframes directory")
    if validate_assets:
        missing_images = [
            meta["file"]
            for meta in normalized_vectors.values()
            if not (keyframes_dir / Path(meta["file"])).is_file()
        ]
        if missing_images:
            raise MergeError(
                f"Shard {root} is missing {len(missing_images)} keyframe(s), "
                f"for example: {missing_images[:3]}"
            )

        video_names = {
            PurePosixPath(meta["file"]).parts[0]
            for meta in normalized_vectors.values()
        }
        missing_csv = [
            name for name in sorted(video_names)
            if not (csv_dir / f"{name}.csv").is_file()
        ]
        if missing_csv:
            raise MergeError(
                f"Shard {root} is missing map-keyframes CSV for: {missing_csv[:10]}"
            )

    return Shard(
        root=root,
        db_path=db_path,
        index_path=index_path,
        vector_meta_path=vector_meta_path,
        vector_meta=vector_meta,
        frame_rows=frame_rows,
    )

def load_merge_source(root: Path, *, validate_assets: bool = True) -> MergeSource:
    """Classify one folder without weakening visual consistency checks."""
    root = root.resolve()
    if not root.is_dir():
        raise MergeError(f"Shard directory does not exist: {root}")

    artifact_paths = [root / relative for relative in VISUAL_ARTIFACT_PATHS]
    present = [path.exists() for path in artifact_paths]
    missing = [str(path.relative_to(root)) for path, exists in zip(artifact_paths, present) if not exists]
    if any(present) and missing:
        raise MergeError(
            f"Shard {root} has an incomplete visual artifact group; missing: "
            + ", ".join(missing)
        )

    asr_cache_path = root / "asr_cache.json"
    if not any(present) and not asr_cache_path.is_file():
        raise MergeError(
            f"Shard {root} contains neither a complete visual group nor asr_cache.json"
        )

    visual = load_shard(root, validate_assets=validate_assets) if all(present) else None
    if asr_cache_path.is_file():
        _load_asr_entries(asr_cache_path)
    return MergeSource(
        root=root,
        visual=visual,
        asr_cache_path=asr_cache_path if asr_cache_path.is_file() else None,
    )

def _validate_ocr_completion(
    shards: list[Shard], *, allow_incomplete: bool = False
) -> None:
    problems: list[str] = []
    versions: set[str] = set()
    for shard in shards:
        counts: dict[str, int] = {}
        for row in shard.frame_rows.values():
            status = str(row.get("ocr_status") or "pending")
            counts[status] = counts.get(status, 0) + 1
            if row.get("ocr_version"):
                versions.add(str(row["ocr_version"]))
        pending = counts.get("pending", 0) + counts.get("processing", 0)
        errors = counts.get("error", 0)
        if pending or errors:
            problems.append(
                f"{shard.root.name}: pending={pending}, errors={errors}"
            )
    if problems and not allow_incomplete:
        raise MergeError(
            "OCR shards are incomplete; finish OCR before merge or explicitly "
            "use --allow-incomplete-ocr: " + "; ".join(problems)
        )
    if versions:
        print("  OCR versions: " + ", ".join(sorted(versions)))


def _same_file(first: Path, second: Path, chunk_size: int = 4 * 1024 * 1024) -> bool:
    if first.stat().st_size != second.stat().st_size:
        return False
    with first.open("rb") as left, second.open("rb") as right:
        while True:
            left_chunk = left.read(chunk_size)
            right_chunk = right.read(chunk_size)
            if left_chunk != right_chunk:
                return False
            if not left_chunk:
                return True


def _copy_one(source: Path, destination: Path, mode: str) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if _same_file(source, destination):
            return "identical"
        raise MergeError(
            f"Conflicting files from different shards: {destination}"
        )
    if mode == "symlink":
        destination.symlink_to(source.resolve())
    elif mode == "hardlink":
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)
    else:
        shutil.copy2(source, destination)
    return "copied"


def _copy_tree_files(source_dir: Path, destination_dir: Path, mode: str) -> tuple[int, int]:
    copied = identical = 0
    if not source_dir.is_dir():
        return copied, identical
    for source in sorted(path for path in source_dir.rglob("*") if path.is_file()):
        destination = destination_dir / source.relative_to(source_dir)
        result = _copy_one(source, destination, mode)
        copied += result == "copied"
        identical += result == "identical"
    return copied, identical


def _create_output_db(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=DELETE")
    conn.execute("PRAGMA synchronous=FULL")
    conn.executescript(
        """
        CREATE TABLE frames (
            id          TEXT PRIMARY KEY,
            file        TEXT NOT NULL UNIQUE,
            mtime       REAL NOT NULL,
            width       INTEGER,
            height      INTEGER,
            objects     TEXT DEFAULT '[]',
            categories  TEXT DEFAULT '[]',
            colors      TEXT DEFAULT '[]',
            has_text    INTEGER DEFAULT 0,
            ocr_texts   TEXT DEFAULT '[]',
            ocr_joined  TEXT DEFAULT '',
            ocr_ascii   TEXT DEFAULT '',
            detections  TEXT DEFAULT '[]',
            faiss_id    INTEGER UNIQUE,
            ocr_status  TEXT DEFAULT 'legacy',
            ocr_version TEXT DEFAULT '',
            ocr_source_file TEXT DEFAULT '',
            ocr_content_hash TEXT DEFAULT '',
            ocr_regions TEXT DEFAULT '[]',
            ocr_attempts INTEGER DEFAULT 0,
            ocr_error TEXT DEFAULT '',
            ocr_updated_at REAL
        );
        CREATE INDEX idx_file ON frames(file);
        CREATE INDEX idx_has_text ON frames(has_text);
        """
    )
    return conn


def _merge_sqlite(
    shards: list[Shard], output_root: Path, *, external_assets: bool = False
) -> dict[str, dict]:
    output_rows: dict[str, dict] = {}
    conn = _create_output_db(output_root / "meta.db")
    placeholders = ",".join("?" for _ in FRAME_COLUMNS)
    sql = (
        f"INSERT INTO frames ({','.join(FRAME_COLUMNS)}) "
        f"VALUES ({placeholders})"
    )
    try:
        for shard in shards:
            for fid, source_row in shard.frame_rows.items():
                if fid in output_rows:
                    previous = output_rows[fid]
                    if previous["file"] != source_row["file"]:
                        raise MergeError(
                            f"Duplicate id {fid!r} points to two files: "
                            f"{previous['file']!r} and {source_row['file']!r}"
                        )
                    if any(previous.get(name) != source_row.get(name) for name in FRAME_COLUMNS if name != "mtime"):
                        raise MergeError(
                            f"Conflicting metadata for duplicate frame {source_row['file']!r}"
                        )
                    continue

                row = dict(source_row)
                if not external_assets:
                    image_path = output_root / "keyframes" / Path(row["file"])
                    row["mtime"] = image_path.stat().st_mtime
                for json_column in ("objects", "categories", "colors", "ocr_texts", "detections", "ocr_regions"):
                    if row.get(json_column) is None:
                        row[json_column] = "[]"
                for text_column in ("ocr_joined", "ocr_ascii"):
                    if row.get(text_column) is None:
                        row[text_column] = ""
                if row.get("has_text") is None:
                    row["has_text"] = 0
                conn.execute(sql, tuple(row.get(name) for name in FRAME_COLUMNS))
                output_rows[fid] = row

        try:
            conn.executescript(
                """
                CREATE VIRTUAL TABLE frames_fts USING fts5(
                    file UNINDEXED,
                    ocr_joined,
                    ocr_ascii,
                    content='frames',
                    content_rowid='rowid'
                );
                INSERT INTO frames_fts(rowid, file, ocr_joined, ocr_ascii)
                SELECT rowid, file, ocr_joined, ocr_ascii FROM frames;
                CREATE TRIGGER frames_ai AFTER INSERT ON frames BEGIN
                    INSERT INTO frames_fts(rowid, file, ocr_joined, ocr_ascii)
                    VALUES (new.rowid, new.file, new.ocr_joined, new.ocr_ascii);
                END;
                CREATE TRIGGER frames_ad AFTER DELETE ON frames BEGIN
                    INSERT INTO frames_fts(frames_fts, rowid, file, ocr_joined, ocr_ascii)
                    VALUES ('delete', old.rowid, old.file, old.ocr_joined, old.ocr_ascii);
                END;
                CREATE TRIGGER frames_au
                AFTER UPDATE OF file, ocr_joined, ocr_ascii ON frames BEGIN
                    INSERT INTO frames_fts(frames_fts, rowid, file, ocr_joined, ocr_ascii)
                    VALUES ('delete', old.rowid, old.file, old.ocr_joined, old.ocr_ascii);
                    INSERT INTO frames_fts(rowid, file, ocr_joined, ocr_ascii)
                    VALUES (new.rowid, new.file, new.ocr_joined, new.ocr_ascii);
                END;
                """
            )
        except sqlite3.OperationalError as exc:
            if "fts5" not in str(exc).lower():
                raise
            print("WARNING: SQLite has no FTS5; OCR search will use the slower fallback.")
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return output_rows


def _new_faiss_index(faiss, dim: int, index_type: str, hnsw_m: int):
    if index_type == "flat":
        base = faiss.IndexFlatIP(dim)
    else:
        base = faiss.IndexHNSWFlat(dim, hnsw_m, faiss.METRIC_INNER_PRODUCT)
    return faiss.IndexIDMap2(base)


def _preflight_faiss(shards: list[Shard]) -> None:
    """Fail before copying large assets when FAISS is unavailable or corrupt."""
    try:
        import faiss
        import numpy  # noqa: F401 - required later by the vector merge
    except ImportError as exc:
        raise MergeError(
            "FAISS/numpy is not installed. Activate the project's .venv first."
        ) from exc

    dimensions = {int(shard.vector_meta["dim"]) for shard in shards}
    if len(dimensions) != 1:
        raise MergeError(f"FAISS shards use different dimensions: {sorted(dimensions)}")

    clip_models = {
        str(shard.vector_meta["clip_model"])
        for shard in shards
        if shard.vector_meta.get("clip_model")
    }
    if len(clip_models) > 1:
        raise MergeError(
            f"FAISS shards were built with different CLIP models: {sorted(clip_models)}"
        )

    metrics = {
        str(shard.vector_meta.get("metric", "cosine_ip")) for shard in shards
    }
    if metrics != {"cosine_ip"}:
        raise MergeError(f"Unsupported or mixed FAISS metrics: {sorted(metrics)}")

    for shard in shards:
        try:
            index = faiss.read_index(str(shard.index_path))
        except Exception as exc:
            raise MergeError(f"Cannot read FAISS index {shard.index_path}: {exc}") from exc
        if index.d != int(shard.vector_meta["dim"]):
            raise MergeError(
                f"{shard.index_path}: index dim={index.d}, "
                f"metadata dim={shard.vector_meta['dim']}"
            )
        if index.ntotal != len(shard.vector_rows):
            raise MergeError(
                f"{shard.index_path}: index has {index.ntotal} vectors but "
                f"metadata has {len(shard.vector_rows)}"
            )


def _deduplicate_external_videos(shards: list[Shard]) -> tuple[int, int]:
    """In external mode, the first source owns every frame of a video."""
    owned: set[str] = set()
    duplicate_videos = 0
    duplicate_frames = 0
    for shard in shards:
        videos: dict[str, list[str]] = {}
        for fid, row in shard.frame_rows.items():
            video = normalize_rel_path(row["file"]).split("/", 1)[0].casefold()
            videos.setdefault(video, []).append(fid)
        rejected: set[str] = set()
        for video, ids in videos.items():
            if video in owned:
                duplicate_videos += 1
                duplicate_frames += len(ids)
                rejected.update(ids)
            else:
                owned.add(video)
        for fid in rejected:
            shard.frame_rows.pop(fid, None)
            shard.vector_rows.pop(fid, None)
    return duplicate_videos, duplicate_frames


def _reconstruct_vector(index, faiss_id: int):
    try:
        return index.reconstruct(faiss_id)
    except Exception:
        try:
            import faiss
            import numpy as np

            id_array = faiss.vector_to_array(index.id_map)
            positions = np.where(id_array == faiss_id)[0]
            if len(positions):
                return index.index.reconstruct(int(positions[0]))
        except Exception:
            pass
    raise MergeError(f"Could not reconstruct FAISS vector id {faiss_id}")


def _idmap_layout(index, vector_rows: dict[str, dict]):
    """Return a validated (base_index, external_ids) layout for an IDMap."""
    if not hasattr(index, "id_map") or not hasattr(index, "index"):
        return None
    try:
        import faiss
        import numpy as np

        external_ids = np.asarray(
            faiss.vector_to_array(index.id_map), dtype=np.int64
        )
    except Exception as exc:
        raise MergeError(f"Could not inspect FAISS ID map: {exc}") from exc

    expected_ids = [int(meta["faiss_id"]) for meta in vector_rows.values()]
    index_ids = [int(value) for value in external_ids.tolist()]
    if len(index_ids) != int(index.ntotal):
        raise MergeError(
            f"FAISS ID map has {len(index_ids)} ids but index has "
            f"{index.ntotal} vectors"
        )
    if len(set(expected_ids)) != len(expected_ids):
        raise MergeError("Duplicate integer faiss_id in vector metadata")
    if len(set(index_ids)) != len(index_ids):
        raise MergeError("Duplicate integer faiss_id in FAISS ID map")
    # External mode may select only a subset after whole-video deduplication.
    # Every selected metadata id must exist in the already preflighted source
    # index; additional source ids are intentionally ignored.
    missing_from_index = set(expected_ids) - set(index_ids)
    if missing_from_index:
        raise MergeError(
            "FAISS ID map and selected vector metadata disagree; missing ids: "
            f"{sorted(missing_from_index)[:3]}"
        )
    return index.index, external_ids


def _record_merged_items(merged_meta, output_rows, items) -> None:
    for fid, meta in items:
        merged_meta[fid] = {
            "faiss_id": int(meta["faiss_id"]),
            "file": meta["file"],
            "mtime": output_rows[fid]["mtime"],
        }


def _add_reconstructed_items(
    merged,
    source_index,
    vector_rows: dict[str, dict],
    new_items: list[tuple[str, dict]],
    merged_meta: dict[str, dict],
    output_rows: dict[str, dict],
    batch_size: int,
    dim: int,
) -> None:
    """Append a shard, preferring positional reconstruct_n batch reads."""
    import numpy as np

    layout = _idmap_layout(source_index, vector_rows)
    if layout is None:
        for start in range(0, len(new_items), batch_size):
            items = new_items[start:start + batch_size]
            vectors = np.asarray(
                [
                    _reconstruct_vector(source_index, int(meta["faiss_id"]))
                    for _, meta in items
                ],
                dtype=np.float32,
            )
            ids = np.asarray(
                [int(meta["faiss_id"]) for _, meta in items], dtype=np.int64
            )
            merged.add_with_ids(vectors, ids)
            _record_merged_items(merged_meta, output_rows, items)
        return

    base_index, external_ids = layout
    new_by_id = {
        int(meta["faiss_id"]): (fid, meta) for fid, meta in new_items
    }
    for start in range(0, int(source_index.ntotal), batch_size):
        count = min(batch_size, int(source_index.ntotal) - start)
        block_ids = external_ids[start:start + count]
        selected_offsets = []
        items = []
        for offset, faiss_id in enumerate(block_ids.tolist()):
            item = new_by_id.get(int(faiss_id))
            if item is not None:
                selected_offsets.append(offset)
                items.append(item)
        if not items:
            continue

        try:
            block = np.asarray(
                base_index.reconstruct_n(start, count), dtype=np.float32
            )
            if block.shape != (count, dim):
                raise ValueError(
                    f"expected {(count, dim)}, received {block.shape}"
                )
            vectors = np.ascontiguousarray(block[selected_offsets])
        except Exception:
            # Older/custom base indexes may expose id_map but no reconstruct_n.
            vectors = np.asarray(
                [
                    _reconstruct_vector(source_index, int(meta["faiss_id"]))
                    for _, meta in items
                ],
                dtype=np.float32,
            )

        ids = np.asarray(
            [int(meta["faiss_id"]) for _, meta in items], dtype=np.int64
        )
        merged.add_with_ids(vectors, ids)
        _record_merged_items(merged_meta, output_rows, items)


def _merge_faiss(
    shards: list[Shard],
    output_root: Path,
    output_rows: dict[str, dict],
    index_type: str,
    hnsw_m: int,
    batch_size: int,
) -> int:
    try:
        import faiss
        import numpy as np
    except ImportError as exc:
        raise MergeError(
            "FAISS/numpy is not installed. Activate the project's .venv first."
        ) from exc

    dimensions = {int(shard.vector_meta["dim"]) for shard in shards}
    if len(dimensions) != 1:
        raise MergeError(f"FAISS shards use different dimensions: {sorted(dimensions)}")
    dim = dimensions.pop()
    clip_models = {
        str(shard.vector_meta["clip_model"])
        for shard in shards
        if shard.vector_meta.get("clip_model")
    }
    if len(clip_models) > 1:
        raise MergeError(
            f"FAISS shards were built with different CLIP models: {sorted(clip_models)}"
        )
    clip_model = next(iter(clip_models), None)
    metrics = {
        str(shard.vector_meta.get("metric", "cosine_ip")) for shard in shards
    }
    if metrics != {"cosine_ip"}:
        raise MergeError(f"Unsupported or mixed FAISS metrics: {sorted(metrics)}")
    merged = _new_faiss_index(faiss, dim, index_type, hnsw_m)
    merged_meta: dict[str, dict] = {}

    for shard_number, shard in enumerate(shards, start=1):
        source_index = faiss.read_index(str(shard.index_path))
        if source_index.d != dim:
            raise MergeError(
                f"{shard.index_path}: index dim={source_index.d}, metadata dim={dim}"
            )
        if source_index.ntotal < len(shard.vector_rows):
            raise MergeError(
                f"{shard.index_path}: index has fewer vectors "
                f"({source_index.ntotal}) than selected metadata "
                f"({len(shard.vector_rows)})"
            )

        new_items = []
        for fid, meta in sorted(shard.vector_rows.items()):
            if fid in merged_meta:
                if merged_meta[fid]["file"] != meta["file"]:
                    raise MergeError(f"Conflicting duplicate FAISS id: {fid}")
                faiss_id = int(meta["faiss_id"])
                existing_vector = np.asarray(
                    _reconstruct_vector(merged, faiss_id), dtype=np.float32
                )
                incoming_vector = np.asarray(
                    _reconstruct_vector(source_index, faiss_id), dtype=np.float32
                )
                if not np.allclose(
                    existing_vector, incoming_vector, rtol=1e-4, atol=1e-5
                ):
                    raise MergeError(
                        f"Conflicting duplicate FAISS vector for {fid!r}"
                    )
                continue
            new_items.append((fid, meta))

        print(
            f"  FAISS shard {shard_number}/{len(shards)}: "
            f"adding {len(new_items)} vector(s)"
        )
        _add_reconstructed_items(
            merged,
            source_index,
            shard.vector_rows,
            new_items,
            merged_meta,
            output_rows,
            batch_size,
            dim,
        )

    if merged.ntotal != len(output_rows) or set(merged_meta) != set(output_rows):
        raise MergeError(
            f"Merged outputs disagree: SQLite={len(output_rows)}, "
            f"FAISS={merged.ntotal}, sidecar={len(merged_meta)}"
        )

    faiss_dir = output_root / "faiss_index"
    faiss_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(merged, str(faiss_dir / "frames.faiss"))
    (faiss_dir / "frames_meta.json").write_text(
        json.dumps(
            {
                "format_version": 2,
                "dim": dim,
                "index_type": index_type,
                "clip_model": clip_model,
                "metric": "cosine_ip",
                "fid_to_meta": merged_meta,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return int(merged.ntotal)


def _merge_asr(
    sources: Iterable[MergeSource],
    output_root: Path,
    *,
    prefer_first: bool = False,
) -> int:
    merged: dict[str, list] = {}
    for source in sources:
        if source.asr_cache_path is None:
            continue
        for key, segments in _load_asr_entries(source.asr_cache_path).items():
            if key in merged:
                if prefer_first:
                    # Source order is the ownership priority for duplicate videos.
                    continue
                if merged[key] != segments:
                    raise MergeError(
                        f"Conflicting ASR transcripts for {key}: "
                        f"{source.asr_cache_path}. Video filenames must be unique "
                        "or transcripts must be identical across shards."
                    )
            merged[key] = segments
    if merged:
        (output_root / "asr_cache.json").write_text(
            json.dumps(merged, ensure_ascii=False), encoding="utf-8"
        )
    return len(merged)


def merge(args: argparse.Namespace) -> None:
    sources = [Path(value).resolve() for value in args.sources]
    if len(set(sources)) != len(sources):
        raise MergeError("The same shard directory was supplied more than once")
    output = Path(args.output).resolve()
    if output.exists():
        if any(output.iterdir()):
            raise MergeError(
                f"Output directory must be absent or empty: {output}. "
                "This guard prevents accidental overwrite of processed data."
            )
        output.rmdir()
    if output in sources:
        raise MergeError("Output directory cannot also be an input shard")

    print("Validating shards...")
    merge_sources = [
        load_merge_source(
            source, validate_assets=args.asset_mode != "external"
        )
        for source in sources
    ]
    shards = [
        source.visual for source in merge_sources if source.visual is not None
    ]
    if not shards:
        raise MergeError(
            "At least one complete visual shard is required for the final merge"
        )
    _validate_ocr_completion(
        shards,
        allow_incomplete=getattr(args, "allow_incomplete_ocr", False),
    )
    for source in merge_sources:
        detail = (
            f"{len(source.visual.frame_rows)} frame(s)"
            if source.visual is not None else
            f"{len(_load_asr_entries(source.asr_cache_path))} ASR video(s)"
        )
        print(f"  {source.root.name}: {source.kind}, {detail}")
    _preflight_faiss(shards)
    if args.asset_mode == "external":
        duplicate_videos, duplicate_frames = _deduplicate_external_videos(shards)
        if duplicate_videos:
            print(
                f"  External priority ignored {duplicate_videos} duplicate video(s), "
                f"{duplicate_frames} frame(s) from later sources"
            )

    staging = output.with_name(f".{output.name}.merge-{uuid.uuid4().hex[:8]}")
    staging.mkdir(parents=True)
    try:
        external_assets = args.asset_mode == "external"
        if external_assets:
            print("Keeping keyframes, timestamp maps, and videos on source storage...")
            (staging / "keyframes").mkdir()
            (staging / "map-keyframes").mkdir()
        else:
            print("Copying keyframes and timestamp maps...")
            for shard in shards:
                _copy_tree_files(shard.root / "keyframes", staging / "keyframes", args.asset_mode)
                _copy_tree_files(
                    shard.root / "map-keyframes", staging / "map-keyframes", args.asset_mode
                )
                if args.include_videos:
                    _copy_tree_files(shard.root / "videos", staging / "videos", args.asset_mode)

        print("Merging SQLite metadata and rebuilding OCR FTS...")
        output_rows = _merge_sqlite(
            shards, staging, external_assets=external_assets
        )

        selected_type = args.index_type
        if selected_type == "auto":
            source_types = {
                str(shard.vector_meta.get("index_type", "hnsw")).lower()
                for shard in shards
            }
            selected_type = source_types.pop() if len(source_types) == 1 else "hnsw"
            if selected_type not in {"flat", "hnsw"}:
                selected_type = "hnsw"

        print(f"Merging existing CLIP vectors into one {selected_type.upper()} FAISS index...")
        vector_count = _merge_faiss(
            shards,
            staging,
            output_rows,
            selected_type,
            args.hnsw_m,
            args.batch_size,
        )
        asr_count = _merge_asr(
            merge_sources, staging, prefer_first=external_assets
        )

        print("Preparing read-only serving indexes, stats, and CLIP text artifact...")
        from prepare_serve_bundle import prepare_dataset
        clip_model = next(
            (
                str(shard.vector_meta.get("clip_model"))
                for shard in shards
                if shard.vector_meta.get("clip_model")
            ),
            "openai/clip-vit-large-patch14",
        )
        prepare_dataset(
            staging,
            strict_ocr=not getattr(args, "allow_incomplete_ocr", False),
            export_clip_text=not getattr(args, "skip_clip_text_export", True),
            clip_model=clip_model,
        )

        if external_assets:
            video_sources: dict[str, str] = {}
            source_rows = []
            for number, shard in enumerate(shards):
                source_id = f"source_{number + 1}_{shard.root.name.lower()}"
                source_rows.append({
                    "id": source_id,
                    "data_root": str(shard.root),
                    "videos_dir": str(shard.root / "videos"),
                    "require_mount": False,
                })
                for row in shard.frame_rows.values():
                    video = normalize_rel_path(row["file"]).split("/", 1)[0]
                    video_sources.setdefault(video.casefold(), source_id)
            (staging / "external_storage.json").write_text(
                json.dumps({
                    "format_version": 2,
                    "asset_mode": "external",
                    "sources": source_rows,
                    "video_sources": video_sources,
                }, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

        external_manifest = getattr(args, "external_storage_manifest", None)
        if external_manifest:
            source_manifest = Path(external_manifest).resolve()
            if not source_manifest.is_file():
                raise MergeError(
                    f"External storage manifest does not exist: {source_manifest}"
                )
            try:
                payload = json.loads(source_manifest.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise MergeError(
                    f"Cannot read external storage manifest {source_manifest}: {exc}"
                ) from exc
            if not isinstance(payload, dict) or not isinstance(payload.get("sources"), list):
                raise MergeError(
                    f"Invalid external storage manifest: {source_manifest}"
                )
            (staging / "external_storage.json").write_text(
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )

        os.replace(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    print("\nMERGE COMPLETE")
    print(f"  Output:       {output}")
    print(f"  Frames/rows:  {len(output_rows)}")
    print(f"  FAISS vectors:{vector_count}")
    print(f"  ASR videos:   {asr_count}")
    if not args.include_videos:
        print("  Videos:       not copied (use --include-videos if needed)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Merge keyframes, timestamp maps, SQLite/OCR, FAISS vectors, and "
            "ASR caches produced by multiple preprocessing machines."
        )
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        required=True,
        metavar="DIR",
        help="Visual or ASR-only shard roots (one directory per machine)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="New empty/nonexistent directory that will receive merged data",
    )
    parser.add_argument(
        "--asset-mode",
        choices=("copy", "hardlink", "symlink", "external"),
        default="copy",
        help=(
            "Copy assets, hardlink/symlink them, or use external to keep "
            "keyframes/CSVs/videos on their source volumes without links"
        ),
    )
    parser.add_argument(
        "--include-videos",
        action="store_true",
        help="Also merge videos/ (off by default because original videos are large)",
    )
    parser.add_argument(
        "--index-type",
        choices=("auto", "hnsw", "flat"),
        default="auto",
        help="Output FAISS type; auto preserves the common shard type",
    )
    parser.add_argument(
        "--hnsw-m",
        type=int,
        default=32,
        help="HNSW graph degree when the output index is HNSW (default: 32)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4096,
        help="Number of reconstructed vectors added per batch (default: 4096)",
    )
    parser.add_argument(
        "--allow-incomplete-ocr",
        action="store_true",
        help="Allow pending/error OCR rows (not recommended for a ready server)",
    )
    parser.add_argument(
        "--skip-clip-text-export",
        action="store_true",
        help="Do not export the local CLIP text-only serving artifact",
    )
    parser.add_argument(
        "--external-storage-manifest",
        help="Validated JSON manifest copied atomically into the output bundle",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.hnsw_m <= 0:
        parser.error("--hnsw-m must be greater than zero")
    if args.batch_size <= 0:
        parser.error("--batch-size must be greater than zero")
    try:
        merge(args)
    except MergeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
