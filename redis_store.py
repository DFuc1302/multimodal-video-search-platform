"""Redis is the source of truth for searchable data in Redis serving mode."""
import hashlib
from folder_filter import matches_folder
import json
import os
import re
from functools import lru_cache

import numpy as np
import redis

BACKEND_MODE = os.getenv("BACKEND_MODE", os.getenv("STORAGE_BACKEND", "sqlite")).strip().lower()
if BACKEND_MODE not in {"sqlite", "redis", "hybrid"}:
    raise RuntimeError(
        f"Unsupported BACKEND_MODE={BACKEND_MODE!r}; use sqlite, hybrid, or redis"
    )

# In hybrid mode, SQLite/FAISS remains the source for visual retrieval and
# frame metadata. Redis is used only for OCR, ASR, and summary indexes.
ENABLED = BACKEND_MODE == "redis"
REDIS_TEXT_ENABLED = BACKEND_MODE in {"redis", "hybrid"}


def dumps(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


@lru_cache(maxsize=1)
def store():
    return RedisStore()


class RedisStore:
    def __init__(self):
        self.r = redis.Redis.from_url(os.getenv("REDIS_URL", "redis://127.0.0.1:16379/0"), socket_timeout=120, protocol=2)
        self.prefix = os.getenv("REDIS_PREFIX", "aic:v1:")

    def key(self, kind, identity):
        return self.prefix + kind + ":" + hashlib.sha256(str(identity).encode()).hexdigest()

    def put_json(self, key, value, ttl=None):
        self.r.set(self.prefix + key, dumps(value), ex=ttl)

    def json(self, key, default=None):
        raw = self.r.get(self.prefix + key)
        return json.loads(raw) if raw else default

    def manifest(self):
        value = self.json("manifest")
        if not value or not value.get("ready"):
            raise RuntimeError("Redis dataset has not passed migration checks")
        return value

    def rows(self, files):
        files = list(dict.fromkeys(files))
        result = []
        for start in range(0, len(files), 1000):
            pipe = self.r.pipeline(transaction=False)
            for file in files[start:start+1000]:
                pipe.hget(self.key("frame", file), "payload")
            result.extend(json.loads(raw) for raw in pipe.execute() if raw)
        return result

    def all_rows(self):
        return self.rows([x.decode() for x in self.r.lrange(self.prefix + "files", 0, -1)])

    def timeline(self, video):
        return self.json("timeline:" + video.upper(), [])

    def text_search(self, kind, phrase, limit=1000000):
        from shared import normalize_ascii_text
        tokens = re.findall(r"\w+", normalize_ascii_text(phrase))
        if not tokens:
            return []
        expression = '@text:"' + ' '.join(tokens) + '"'
        response = self.r.execute_command("FT.SEARCH", self.prefix+kind+"_idx", expression,
            "WITHSCORES", "SCORER", "BM25STD", "RETURN", 1, "payload", "LIMIT", 0, limit, "DIALECT", 2)
        result = []
        for i in range(1, len(response), 3):
            fields = dict(zip(response[i+2][::2], response[i+2][1::2]))
            result.append((json.loads(fields[b"payload"]), float(response[i+1])))
        return result

    def asr_matches(self, phrase, offset, limit, operator):
        from asr_index import is_asr_excluded_video
        terms = [x.strip() for x in phrase.split(",") if x.strip()]
        rankings = [self.text_search("asr", term) for term in terms]
        allowed = None
        if operator == "and" and rankings:
            allowed = set.intersection(*({x[0]["video_id"] for x in ranking} for ranking in rankings))
        matches = {}
        for ranking in rankings:
            for item, score in ranking:
                vid = item["video_id"]
                if is_asr_excluded_video(vid) or (allowed is not None and vid not in allowed):
                    continue
                key = (vid, item["start"], item["end"])
                prev = matches.get(key)
                if prev is None or score > prev[1]:
                    matches[key] = (item, score)
        ordered = sorted(matches.values(), key=lambda x: (-x[1], x[0]["video_id"], x[0]["start"]))
        return dict(total=len(ordered), videos=len({x[0]["video_id"] for x in ordered}),
            matches=[({"video_id": x["video_id"], "video_name": x["video_name"]}, {**x, "score": score})
                     for x, score in ordered[offset:offset+limit]])

    def query_files(self, objects=None, categories=None, colors=None, video=None, exclude_video=False, limit=1000000):
        if not any((objects, categories, colors, video)):
            return []
        objects, categories, colors = ({value.casefold() for value in (values or [])}
                                      for values in (objects, categories, colors))
        found = []
        for row in self.all_rows():
            if any(wanted and not {x.casefold() for x in row.get(field, [])} & wanted
                   for field, wanted in (("objects", objects), ("categories", categories), ("colors", colors))):
                continue
            if video and (matches_folder(row["file"], video) == exclude_video):
                continue
            found.append(row["file"])
        return sorted(found)[:limit]

    def update_frame(self, row):
        from shared import normalize_ascii_text
        self.r.hset(self.key("frame", row["file"]), mapping={"payload": dumps(row),
                    "text": normalize_ascii_text(row.get("ocr_joined", ""))})


class RedisCollection:
    def __init__(self, kind="frame", dim=768):
        self.s = store()
        self.kind, self.dim = kind, dim

    def count(self):
        return int(self.s.r.get(self.s.prefix + self.kind + "_count") or 0)

    def dimension(self):
        return self.dim

    def query(self, query_embeddings, n_results):
        metas, distances = [], []
        for vector in query_embeddings:
            q = np.asarray(vector, dtype=np.float32)
            q /= max(float(np.linalg.norm(q)), 1e-12)
            k = min(max(1, int(n_results)), self.count())
            if not k:
                metas.append([]); distances.append([]); continue
            raw = self.s.r.execute_command("FT.SEARCH", self.s.prefix+self.kind+"_idx",
                ("@vector:[VECTOR_RANGE 2.01 $q]=>{$YIELD_DISTANCE_AS: distance}" if k == self.count()
                 else f"*=>[KNN {k} @vector $q AS distance]"), "PARAMS", 2, "q", q.tobytes(),
                "SORTBY", "distance", "ASC", "RETURN", 2, "file", "distance", "LIMIT", 0, k, "DIALECT", 2)
            fields = [dict(zip(raw[i+1][::2], raw[i+1][1::2])) for i in range(1, len(raw), 2)]
            metas.append([{"file": f[b"file"].decode()} for f in fields])
            distances.append([float(f[b"distance"]) for f in fields])
        return {"metadatas": metas, "distances": distances}

    def reconstruct_by_files(self, files):
        result = {}
        for start in range(0, len(files), 1000):
            batch = files[start:start+1000]
            pipe = self.s.r.pipeline(transaction=False)
            for file in batch:
                pipe.hget(self.s.key(self.kind, file), "vector")
            for file, raw in zip(batch, pipe.execute()):
                if raw:
                    result[file] = np.frombuffer(raw, dtype=np.float32).copy()
        return result
