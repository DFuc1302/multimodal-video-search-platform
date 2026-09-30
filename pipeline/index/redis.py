"""Import existing AIC indexes into Redis without recomputing OCR or CLIP."""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

import faiss
import redis


JSON_FIELDS = ("objects", "categories", "colors", "detections", "ocr_texts", "ocr_regions")


def args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-index", type=Path, default=Path("../aic_local_prep/search_index_ocr"))
    parser.add_argument("--summary-index", type=Path, default=Path("../aic_local_prep/summary_index"))
    parser.add_argument("--media-root", type=Path, default=Path("../filtered_data_all_873_moved"))
    parser.add_argument("--redis-url", default=os.getenv("REDIS_URL", "redis://127.0.0.1:16379/0"))
    parser.add_argument("--prefix", default=os.getenv("REDIS_PREFIX", "aic:v2:"))
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def dumps(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def load_rows(path: Path):
    import sqlite3

    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
        conn.row_factory = sqlite3.Row
        return [dict(row) for row in conn.execute("SELECT * FROM frames ORDER BY file")]


def normalize_row(row):
    for field in JSON_FIELDS:
        value = row.get(field)
        if isinstance(value, str):
            try:
                row[field] = json.loads(value or "[]")
            except json.JSONDecodeError:
                row[field] = []
    return row


def key(prefix, kind, identity):
    import hashlib
    return prefix + kind + ":" + hashlib.sha256(str(identity).encode()).hexdigest()


def create_index(client, prefix, kind, dimension=None):
    schema = ["text", "TEXT", "NOSTEM", "file", "TAG"]
    if dimension:
        schema += ["vector", "VECTOR", "FLAT", 6, "TYPE", "FLOAT32", "DIM", dimension,
                   "DISTANCE_METRIC", "COSINE"]
    try:
        client.execute_command("FT.CREATE", prefix + kind + "_idx", "ON", "HASH",
                               "PREFIX", 1, prefix + kind + ":", "STOPWORDS", 0,
                               "SCHEMA", *schema)
    except redis.ResponseError as exc:
        if "Index already exists" not in str(exc):
            raise


def wait_index(client, prefix, kind, expected):
    for _ in range(180):
        raw = client.execute_command("FT.INFO", prefix + kind + "_idx")
        info = raw if isinstance(raw, dict) else dict(zip(raw[::2], raw[1::2]))
        docs = info.get(b"num_docs", info.get("num_docs", 0))
        indexing = info.get(b"indexing", info.get("indexing", 0))
        if int(docs) == expected and not int(indexing):
            return
        time.sleep(1)
    raise RuntimeError(f"Redis index {kind} chưa hoàn tất: expected={expected}")


def import_frames(client, prefix, source: Path, rows, media: Path):
    index = faiss.read_index(str(source / "faiss_index" / "frames.faiss"))
    if index.ntotal != len(rows):
        raise RuntimeError(f"FAISS={index.ntotal}, SQLite={len(rows)}: dữ liệu không khớp")
    files, timelines, statuses = [], {}, {}
    pipe = client.pipeline(transaction=False)
    for number, raw in enumerate(rows, 1):
        row = normalize_row(raw)
        filename = row["file"]
        files.append(filename)
        video = Path(filename).parent.name.upper()
        timelines.setdefault(video, {})[int(Path(filename).stem)] = filename
        status = row.get("ocr_status", "")
        statuses[status] = statuses.get(status, 0) + 1
        vector = index.reconstruct(int(row["faiss_id"]))
        text = str(row.get("ocr_joined") or "")
        pipe.hset(key(prefix, "frame", filename), mapping={
            "file": filename, "payload": dumps(row), "text": text,
            "vector": vector.tobytes(),
        })
        if number % 500 == 0:
            pipe.execute()
            print(f"Frames migrated: {number}/{len(rows)}", flush=True)
    pipe.execute()
    client.delete(prefix + "files")
    for start in range(0, len(files), 1000):
        client.rpush(prefix + "files", *files[start:start + 1000])
    client.set(prefix + "frame_count", len(files))

    map_root = media / "map-keyframes"
    for video, lookup in timelines.items():
        timeline = []
        csv_path = map_root / f"{video}.csv"
        if csv_path.is_file():
            with csv_path.open(encoding="utf-8-sig", newline="") as handle:
                for mapped in csv.DictReader(handle):
                    number = int(mapped["n"])
                    if number in lookup:
                        timeline.append({"file": lookup[number], "n": number,
                                         "timestamp": float(mapped["pts_time"]),
                                         "frame_idx": int(mapped["frame_idx"])})
        client.set(prefix + "timeline:" + video, dumps(sorted(timeline, key=lambda x: (x["timestamp"], x["file"]))))
    return files, timelines, statuses


def import_asr(client, prefix, source: Path):
    cache_path = source / "asr_cache.json"
    if not cache_path.is_file():
        client.set(prefix + "asr_count", 0)
        client.set(prefix + "asr_cache", dumps({}))
        return 0, 0
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    cache_prefix = "video-name-v1::"
    client.set(prefix + "asr_cache", dumps(cache))
    count = 0
    pipe = client.pipeline(transaction=False)
    for cache_key, segments in cache.items():
        raw_key = str(cache_key)
        if raw_key.startswith(cache_prefix):
            video_name = raw_key[len(cache_prefix):]
        else:
            video_name = raw_key.rsplit("::", 1)[0].replace("\\", "/").rsplit("/", 1)[-1]
        video_id = Path(video_name).stem
        if video_id.casefold() == "l30_v029":
            continue
        for number, segment in enumerate(segments):
            payload = {**segment, "video_id": video_id, "video_name": video_name}
            pipe.hset(key(prefix, "asr", f"{video_id}:{number}"),
                      mapping={"payload": dumps(payload), "text": str(segment.get("text", ""))})
            count += 1
            if count % 1000 == 0:
                pipe.execute()
    pipe.execute()
    client.set(prefix + "asr_count", count)
    return count, len(cache)


def import_summaries(client, prefix, summary_root: Path):
    db_path, faiss_path = summary_root / "summaries.db", summary_root / "summaries.faiss"
    if not db_path.is_file() or not faiss_path.is_file():
        client.set(prefix + "summary_count", 0)
        client.set(prefix + "summary_docs", dumps([]))
        return 0
    import sqlite3
    index = faiss.read_index(str(faiss_path))
    with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
        conn.row_factory = sqlite3.Row
        docs = [dict(row) for row in conn.execute("SELECT * FROM docs ORDER BY id")]
    if index.ntotal != len(docs):
        raise RuntimeError(f"Summary FAISS={index.ntotal}, DB={len(docs)}: dữ liệu không khớp")
    pipe = client.pipeline(transaction=False)
    for doc in docs:
        identity = str(doc["id"])
        pipe.hset(key(prefix, "summary", identity), mapping={
            "file": identity, "payload": dumps(doc), "text": str(doc.get("text", "")),
            "vector": index.reconstruct(int(doc["id"])).tobytes(),
        })
    pipe.execute()
    client.set(prefix + "summary_docs", dumps(docs))
    client.set(prefix + "summary_count", len(docs))
    return len(docs)


def verify(client, prefix):
    result = {}
    for kind in ("frame", "asr", "summary"):
        expected = int(client.get(prefix + kind + "_count") or 0)
        raw = client.execute_command("FT.INFO", prefix + kind + "_idx")
        info = raw if isinstance(raw, dict) else dict(zip(raw[::2], raw[1::2]))
        result[kind] = {"expected": expected,
                        "indexed": int(info.get(b"num_docs", info.get("num_docs", 0))),
                        "indexing": int(info.get(b"indexing", info.get("indexing", 0)))}
    if result["frame"]["expected"] != result["frame"]["indexed"]:
        raise RuntimeError(f"Redis frame index chưa đủ: {result}")
    return result


def wait_for_redis(client):
    timeout = max(30, int(os.getenv("REDIS_READY_TIMEOUT", "900")))
    deadline = time.monotonic() + timeout
    announced = False
    while True:
        try:
            client.ping()
            return
        except redis.exceptions.BusyLoadingError:
            if not announced:
                print("Redis dang nap snapshot, dang cho den khi san sang...", flush=True)
                announced = True
        except redis.exceptions.ConnectionError:
            if not announced:
                print("Dang cho Redis khoi dong...", flush=True)
                announced = True
        if time.monotonic() >= deadline:
            raise RuntimeError(f"Redis chua san sang sau {timeout} giay")
        time.sleep(2)


def main():
    options = args()
    client = redis.Redis.from_url(options.redis_url, socket_timeout=120, protocol=2)
    wait_for_redis(client)
    prefix = options.prefix if options.prefix.endswith(":") else options.prefix + ":"
    create_index(client, prefix, "frame", 768)
    create_index(client, prefix, "asr")
    create_index(client, prefix, "summary", 384)
    source = options.source_index.resolve()
    if options.verify_only:
        print(json.dumps(verify(client, prefix), indent=2))
        return
    rows = load_rows(source / "meta.db")
    files, timelines, statuses = import_frames(client, prefix, source, rows, options.media_root.resolve())
    asr_count, asr_videos = import_asr(client, prefix, source)
    summary_count = import_summaries(client, prefix, options.summary_index.resolve())
    wait_index(client, prefix, "frame", len(files))
    wait_index(client, prefix, "asr", asr_count)
    wait_index(client, prefix, "summary", summary_count)
    report = {"ready": True, "backend": "redis", "prefix": prefix,
              "frame_count": len(files), "asr_count": asr_count,
              "asr_videos": asr_videos, "summary_count": summary_count,
              "video_count": len(timelines), "ocr_statuses": statuses,
              "source_index": str(source), "media_root": str(options.media_root.resolve()),
              "migrated_at": time.time()}
    client.set(prefix + "manifest", dumps(report))
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(verify(client, prefix), indent=2))


if __name__ == "__main__":
    main()
