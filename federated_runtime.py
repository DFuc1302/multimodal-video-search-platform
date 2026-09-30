"""Read-only search federation across independent local/removable datasets."""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from dataset_layout import DatasetShard, load_dataset_config
from folder_filter import matches_folder


class FederationError(RuntimeError):
    pass


def _chunks(values: list[str], size: int = 800):
    for start in range(0, len(values), size):
        yield values[start:start + size]


def _slim_row(row: sqlite3.Row) -> dict:
    return {
        "file": row["file"],
        "mtime": row["mtime"],
        "width": row["width"],
        "height": row["height"],
        "objects": json.loads(row["objects"] or "[]"),
        "categories": json.loads(row["categories"] or "[]"),
        "colors": json.loads(row["colors"] or "[]"),
        "has_text": bool(row["has_text"]),
        "ocr_joined": row["ocr_joined"] or "",
        "ocr_ascii": row["ocr_ascii"] or "",
        "detections": json.loads(row["detections"] or "[]"),
    }


@dataclass
class DatasetReader:
    shard: DatasetShard
    manifest: dict
    index: object
    faiss_ids: dict[int, dict]

    @property
    def storage_id(self) -> str:
        return self.shard.shard_id

    @property
    def root(self) -> Path:
        return self.shard.data_root

    @property
    def db_path(self) -> Path:
        return self.root / "meta.db"

    def connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            f"file:{self.db_path.resolve()}?mode=ro", uri=True, timeout=30.0
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA query_only=ON")
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    def metadata_for_ids(self, ids: Iterable[int]) -> dict[int, dict]:
        wanted = [int(value) for value in ids if int(value) >= 0]
        if not wanted:
            return {}
        if self.faiss_ids:
            return {value: self.faiss_ids[value] for value in wanted if value in self.faiss_ids}
        found: dict[int, dict] = {}
        with self.connect() as conn:
            for chunk_start in range(0, len(wanted), 800):
                chunk = wanted[chunk_start:chunk_start + 800]
                placeholders = ",".join("?" for _ in chunk)
                rows = conn.execute(
                    f"SELECT faiss_id, file, mtime FROM frames WHERE faiss_id IN ({placeholders})",
                    chunk,
                ).fetchall()
                found.update({
                    int(row["faiss_id"]): {"file": row["file"], "mtime": row["mtime"]}
                    for row in rows
                })
        return found


class FederatedCollection:
    def __init__(self, runtime: "FederatedRuntime"):
        self.runtime = runtime

    def count(self) -> int:
        return self.runtime._unique_frame_count

    def dimension(self) -> int | None:
        dimensions = {int(reader.index.d) for reader in self.runtime.readers}
        if not dimensions:
            return None
        if len(dimensions) != 1:
            raise FederationError(f"FAISS dimensions differ: {sorted(dimensions)}")
        return dimensions.pop()

    def query(self, query_embeddings: list, n_results: int) -> dict:
        query = np.asarray(query_embeddings, dtype=np.float32)
        if query.ndim == 1:
            query = query.reshape(1, -1)
        query /= np.linalg.norm(query, axis=1, keepdims=True) + 1e-8
        all_metas = []
        all_distances = []
        for i in range(len(query)):
            q_row = query[i : i + 1]
            candidates: dict[str, tuple[float, dict]] = {}
            for reader in self.runtime.readers:
                count = int(reader.index.ntotal)
                if not count:
                    continue
                k = min(max(1, int(n_results)), count)
                scores, ids = reader.index.search(q_row, k)
                metadata = reader.metadata_for_ids(ids[0])
                for score, faiss_id in zip(scores[0], ids[0]):
                    if faiss_id < 0:
                        continue
                    meta = metadata.get(int(faiss_id))
                    if not meta or not meta.get("file"):
                        continue
                    filename = str(meta["file"])
                    if self.runtime.owner(filename) is not reader:
                        continue
                    decorated = {
                        "file": filename,
                        "mtime": meta.get("mtime", 0),
                        "storage_id": reader.storage_id,
                    }
                    previous = candidates.get(filename)
                    if previous is None or float(score) > previous[0]:
                        candidates[filename] = (float(score), decorated)
            ranked = sorted(candidates.values(), key=lambda item: item[0], reverse=True)
            ranked = ranked[: max(1, int(n_results))]
            all_metas.append([item[1] for item in ranked])
            all_distances.append([float(1.0 - item[0]) for item in ranked])
        return {"metadatas": all_metas, "distances": all_distances}

    def reconstruct_by_files(self, files: list[str]) -> dict:
        # KIS calls this only for a narrowed pool. Query the owning reader and
        # reconstruct by its persisted FAISS id.
        result = {}
        for reader in self.runtime.readers:
            owned = [name for name in files if self.runtime.owner(name) is reader]
            if not owned:
                continue
            with reader.connect() as conn:
                for chunk in _chunks(owned):
                    placeholders = ",".join("?" for _ in chunk)
                    rows = conn.execute(
                        f"SELECT file, faiss_id FROM frames WHERE file IN ({placeholders})",
                        chunk,
                    ).fetchall()
                    for row in rows:
                        try:
                            result[row["file"]] = np.asarray(
                                reader.index.reconstruct(int(row["faiss_id"])),
                                dtype=np.float32,
                            )
                        except Exception:
                            continue
        return result


class FederatedRuntime:
    def __init__(self, config_path: str | Path, *, allow_offline: bool = True):
        import faiss

        self.config = load_dataset_config(config_path)
        self.allow_offline = allow_offline
        self.readers: list[DatasetReader] = []
        self.offline: dict[str, str] = {}
        self._video_owner: dict[str, DatasetReader] = {}
        self._unique_frame_count = 0
        self._duplicate_frame_count = 0
        self.duplicates: dict[str, list[str]] = {}
        self._lock = threading.RLock()

        clip_models: set[str] = set()
        dimensions: set[int] = set()
        for shard in self.config.select():
            try:
                shard.preflight(write=False)
                manifest_path = shard.data_root / "serve_manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                index_path = shard.data_root / "faiss_index" / "frames.faiss"
                if not index_path.is_file() or not shard.meta_db.is_file():
                    raise FederationError(f"Dataset chưa có SQLite/FAISS hoàn chỉnh: {shard.data_root}")
                index = faiss.read_index(
                    str(index_path), faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY
                )
                # Map only the small result set returned by each search via
                # indexed SQLite faiss_id lookups. Loading the JSON sidecars
                # would expand >100 MB into a very large Python object.
                reader = DatasetReader(shard, manifest, index, {})
                self._validate_reader(reader)
                self.readers.append(reader)
                clip_models.add(str(manifest.get("clip_model", "")))
                dimensions.add(int(index.d))
            except Exception as exc:
                self.offline[shard.shard_id] = str(exc)
                if not allow_offline:
                    raise
        if not self.readers:
            raise FederationError(
                "Không có dataset search-ready; hãy preprocess và bundle local hoặc USB"
            )
        if len(clip_models) != 1:
            raise FederationError(f"Các dataset dùng CLIP model khác nhau: {sorted(clip_models)}")
        if len(dimensions) != 1:
            raise FederationError(f"Các dataset có FAISS dimension khác nhau: {sorted(dimensions)}")
        self.clip_model = next(iter(clip_models))
        self.dimension = next(iter(dimensions))
        self.collection = FederatedCollection(self)
        self._build_catalog()

    def _validate_reader(self, reader: DatasetReader) -> None:
        manifest = reader.manifest
        frame_count = int(manifest.get("frame_count", -1))
        if frame_count <= 0 or frame_count != int(reader.index.ntotal):
            raise FederationError(
                f"Manifest/FAISS mismatch tại {reader.root}: {frame_count} != {reader.index.ntotal}"
            )
        with reader.connect() as conn:
            db_count = int(conn.execute("SELECT COUNT(*) FROM frames").fetchone()[0])
            if db_count != frame_count:
                raise FederationError(
                    f"SQLite/FAISS mismatch tại {reader.root}: {db_count} != {frame_count}"
                )

    def _build_catalog(self) -> None:
        for reader in self.readers:
            with reader.connect() as conn:
                rows = conn.execute(
                    "WITH normalized AS ("
                    " SELECT replace(file, char(92), '/') AS name FROM frames"
                    "), videos AS ("
                    " SELECT CASE WHEN instr(name, '/') > 0"
                    " THEN substr(name, 1, instr(name, '/') - 1) ELSE name END AS video"
                    " FROM normalized"
                    ") SELECT video, COUNT(*) AS frame_count"
                    " FROM videos WHERE video != '' GROUP BY video ORDER BY video"
                ).fetchall()
            for row in rows:
                video = str(row["video"])
                frame_count = int(row["frame_count"])
                key = Path(video).stem.casefold()
                old_video = self._video_owner.get(key)
                if old_video is not None and old_video is not reader:
                    self.duplicates.setdefault(video, [old_video.storage_id]).append(
                        reader.storage_id
                    )
                    self._duplicate_frame_count += frame_count
                    continue
                self._video_owner[key] = reader
                self._unique_frame_count += frame_count

    def owner(self, filename: str) -> DatasetReader | None:
        normalized = str(filename).replace("\\", "/")
        return self.owner_for_video(normalized.split("/", 1)[0])

    def owner_for_video(self, video: str) -> DatasetReader | None:
        return self._video_owner.get(Path(video).stem.casefold())

    def resolve_image(self, filename: str) -> Path | None:
        reader = self.owner(filename)
        if reader is not None:
            path = reader.root / "keyframes" / filename
            return path if path.is_file() else None
        return None

    def resolve_map(self, video: str) -> Path | None:
        reader = self.owner_for_video(video)
        if reader is not None:
            path = reader.root / "map-keyframes" / f"{Path(video).stem}.csv"
            return path if path.is_file() else None
        return None

    def get_meta_by_files(self, files: list[str], *, full: bool = False) -> list:
        ordered = list(dict.fromkeys(str(value) for value in files if value))
        found: dict[str, object] = {}
        per_reader: dict[int, tuple[DatasetReader, list[str]]] = {}
        for filename in ordered:
            reader = self.owner(filename)
            if reader is not None:
                per_reader.setdefault(id(reader), (reader, []))[1].append(filename)
        for reader, names in per_reader.values():
            with reader.connect() as conn:
                for chunk in _chunks(names):
                    placeholders = ",".join("?" for _ in chunk)
                    columns = "*" if full else (
                        "file, mtime, width, height, objects, categories, colors, "
                        "has_text, ocr_joined, ocr_ascii, detections"
                    )
                    rows = conn.execute(
                        f"SELECT {columns} FROM frames WHERE file IN ({placeholders})",
                        chunk,
                    ).fetchall()
                    for row in rows:
                        if full:
                            found[row["file"]] = row
                        else:
                            item = _slim_row(row)
                            item["storage_id"] = reader.storage_id
                            found[row["file"]] = item
        return [found[name] for name in ordered if name in found]

    def get_frame_meta(self, filename: str) -> dict | None:
        rows = self.get_meta_by_files([filename], full=True)
        if not rows:
            return None
        row = rows[0]
        return {
            "file": row["file"],
            "storage_id": self.owner(filename).storage_id,
            "width": row["width"], "height": row["height"],
            "objects": json.loads(row["objects"] or "[]"),
            "categories": json.loads(row["categories"] or "[]"),
            "colors": json.loads(row["colors"] or "[]"),
            "has_text": bool(row["has_text"]),
            "ocr_texts": json.loads(row["ocr_texts"] or "[]"),
            "ocr_joined": row["ocr_joined"] or "",
            "ocr_ascii": row["ocr_ascii"] or "",
            "detections": json.loads(row["detections"] or "[]"),
        }

    def get_all_meta(self) -> list[dict]:
        files = []
        for reader in self.readers:
            with reader.connect() as conn:
                files.extend(row['file'] for row in conn.execute('SELECT file FROM frames ORDER BY file')
                             if self.owner(row['file']) is reader)
        return self.get_meta_by_files(files)

    def query_frame_files(self, *, objects=None, categories=None, colors=None, video=None, exclude_video=False, limit=10000):
        gathered: set[str] = set()
        filters = [
            ("object", {value.casefold() for value in (objects or set())}),
            ("category", {value.casefold() for value in (categories or set())}),
            ("color", {value.casefold() for value in (colors or set())}),
        ]
        for reader in self.readers:
            subqueries: list[str] = []
            params: list[object] = []
            for kind, values in filters:
                if values:
                    placeholders = ",".join("?" for _ in values)
                    subqueries.append(
                        f"SELECT file FROM frame_facets WHERE kind=? AND value IN ({placeholders})"
                    )
                    params.extend([kind, *sorted(values)])
            if video:
                subqueries.append(
                    "SELECT file FROM frames WHERE "
                    f"{'NOT ' if exclude_video else ''}matches_folder(file, ?)"
                )
                params.append(video)
            if not subqueries:
                continue
            params.append(max(1, int(limit)))
            with reader.connect() as conn:
                conn.create_function('matches_folder', 2, matches_folder, deterministic=True)
                rows = conn.execute(
                    f"SELECT file FROM ({' INTERSECT '.join(subqueries)}) ORDER BY file LIMIT ?",
                    params,
                ).fetchall()
            gathered.update(
                str(row["file"])
                for row in rows
                if self.owner(str(row["file"])) is reader
            )
        return sorted(gathered)[: max(1, int(limit))]

    def ocr_match_files(self, match: str, limit: int) -> tuple[list[str], bool]:
        ranked: list[str] = []
        for reader in self.readers:
            try:
                with reader.connect() as conn:
                    rows = conn.execute(
                        "SELECT file FROM frames_fts WHERE frames_fts MATCH ? ORDER BY rank LIMIT ?",
                        (match, limit),
                    ).fetchall()
                ranked.extend(
                    str(row["file"])
                    for row in rows
                    if self.owner(str(row["file"])) is reader
                )
            except sqlite3.OperationalError:
                continue
        return list(dict.fromkeys(ranked))[:limit], True

    def frame_locator(self, video: str) -> list[sqlite3.Row]:
        reader = self.owner_for_video(video)
        if reader is None:
            return []
        with reader.connect() as conn:
            return conn.execute(
                "SELECT n, file FROM frame_locator WHERE video=? COLLATE NOCASE ORDER BY n",
                (video,),
            ).fetchall()

    def asr_caches(self) -> list[tuple[DatasetReader, dict]]:
        caches = []
        for reader in self.readers:
            path = reader.root / "asr_cache.json"
            if path.is_file():
                try:
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    if isinstance(payload, dict):
                        caches.append((reader, payload))
                except (OSError, json.JSONDecodeError):
                    continue
        return caches

    def status(self) -> dict:
        counters = {
            "categories": {}, "objects": {}, "colors": {},
            "has_text_count": 0, "ocr_cached_count": 0,
            "pending_ocr": 0, "processing_ocr": 0, "ocr_errors": 0,
        }
        for reader in self.readers:
            stats_path = reader.root / "serve_stats.json"
            try:
                stats = json.loads(stats_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            for group in ("categories", "objects", "colors"):
                for key, value in stats.get(group, {}).items():
                    counters[group][key] = counters[group].get(key, 0) + int(value)
            for key in (
                "has_text_count", "ocr_cached_count", "pending_ocr",
                "processing_ocr", "ocr_errors",
            ):
                counters[key] += int(stats.get(key, 0))
        return {
            "mode": "federated",
            "online": [
                {"id": reader.storage_id, "root": str(reader.root), "frames": int(reader.index.ntotal)}
                for reader in self.readers
            ],
            "offline": self.offline,
            "frames": self.collection.count(),
            "videos": len(self._video_owner),
            "duplicate_videos_ignored": len(self.duplicates),
            "duplicate_frames_ignored": self._duplicate_frame_count,
            **counters,
        }
