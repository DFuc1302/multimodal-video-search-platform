"""Image-based pagination for audio, video matching, and conversation results."""
from redis_store import REDIS_TEXT_ENABLED, store as redis_store
from collections import OrderedDict
import threading
import time

from fastapi.responses import JSONResponse
from query_hints import keyword_hints

PAGE_SIZE = 1000
_cache = OrderedDict()
_lock = threading.Lock()


def snapshot(key, build):
    if REDIS_TEXT_ENABLED:
        import hashlib, json, zlib
        db=redis_store()
        cache_key=db.prefix+'cache:'+hashlib.sha256(repr(key).encode()).hexdigest()
        raw=db.r.get(cache_key)
        if raw:return json.loads(zlib.decompress(raw))
        with db.r.lock(cache_key+':lock',timeout=180,blocking_timeout=180):
            raw=db.r.get(cache_key)
            if raw:return json.loads(zlib.decompress(raw))
            value=build()
            db.r.set(cache_key,zlib.compress(json.dumps(value).encode()),ex=600)
            return value
    # Freeze rankings during navigation; retain only two recent result sets.
    with _lock:
        entry = _cache.get(key)
        if entry and time.monotonic() - entry[0] < 600:
            _cache.move_to_end(key)
            return entry[1]
        value = build()
        _cache[key] = (time.monotonic(), value)
        while len(_cache) > 2:
            _cache.popitem(last=False)
        return value


def slice_page(items, page):
    page = max(1, min(int(page), max(1, (len(items) + PAGE_SIZE - 1) // PAGE_SIZE)))
    return items[(page-1)*PAGE_SIZE:page*PAGE_SIZE], page


def asr_page(body):
    import main as api
    phrase = str(body.get("phrase") or "").strip()
    operator = str(body.get("operator", "and")).strip().lower()
    try:
        page_number = int(body.get('page', 1))
    except (ValueError, TypeError, OverflowError):
        return JSONResponse({'error': 'page must be an integer'}, status_code=400)
    if not phrase or operator not in {"and", "or"}:
        return JSONResponse({"error": "Provide phrase and valid operator"}, status_code=400)
    def build():
        data = api._search_asr_fts(phrase, offset=0, limit=2147483647, operator=operator)
        if data is None:
            # The legacy cache search is unbounded when its limit covers all windows.
            cache = api.load_asr_cache()
            matches = []
            terms = [x.strip() for x in phrase.split(",") if x.strip()]
            for video in api._asr_cache_inventory(cache)["entries"]:
                if not video['current']:
                    continue
                groups = [list(api.find_phrase_segments(video["segments"], term)) for term in terms]
                if operator == "and" and not all(groups):
                    continue
                matches.extend((video, match) for group in groups for match in group)
            unique_matches = {(video['video_id'], match['start'], match['end'], match['text']): (video, match)
                              for video, match in matches}
            matches = list(unique_matches.values())
            data = {"matches": matches, "total": len(matches),
                    "videos": len({video['video_id'] for video, _ in matches})}
        timelines, unique = {}, {}
        unmapped = 0
        for video, match in data["matches"]:
            vid = video["video_id"]
            if api.is_asr_excluded_video(vid):
                continue
            if vid not in timelines:
                timelines[vid] = api._load_asr_keyframe_timeline(vid)
            timeline, mapping = timelines[vid]
            frames = api._frames_in_asr_interval(timeline, match["start"], match["end"])
            if not frames:
                unmapped += 1
            score = match.get("score", 0.0)
            for frame in frames:
                item = dict(video=video["video_name"], video_id=vid,
                            start=match["start"], end=match["end"],
                            duration=round(match["end"]-match["start"], 2),
                            text=match["text"], score=score, frames=[frame], frame_count=1,
                            mapping_mode=mapping.get("mapping_mode"))
                previous = unique.get(frame["file"])
                if previous is None or score > previous["score"]:
                    unique[frame["file"]] = item
        items = sorted(unique.values(), key=lambda x: (-x["score"], x["frames"][0]["file"]))
        return items, data["total"], data["videos"], unmapped
    # Cache invalidates when transcript artifacts change.
    signature = redis_store().manifest()['migrated_at'] if REDIS_TEXT_ENABLED else tuple((str(p), p.stat().st_mtime_ns) for p in api._asr_cache_files() if p.exists())
    items, intervals, videos, unmapped = snapshot(("asr", phrase, operator, signature), build)
    page_items, page = slice_page(items, page_number)
    return dict(results=page_items, total=len(items), page=page, total_frames=len(page_items),
                phrase=phrase, operator=operator, videos_scanned=videos,
                matched_intervals=intervals, unmapped_intervals=unmapped, score_type="bm25")


def video_page(query, page):
    import main as api
    if not query.strip():
        return JSONResponse({"error": "Provide query"}, status_code=400)
    def build():
        col = api.get_collection()
        vector = api._embed_query_terms([query])[0]
        result = col.query(query_embeddings=[vector.tolist()], n_results=max(1, col.count()))
        scores = {meta["file"]: 1-float(d) for meta, d in zip(result["metadatas"][0], result["distances"][0])}
        colors = keyword_hints(query, api.COLOR_ALIASES)
        for row in api.get_meta_by_files(list(scores)):
            scores[row["file"]] += 0.03 * len(colors & set(row.get("colors", [])))
        return sorted(scores.items(), key=lambda x: (-x[1], x[0]))
    items = snapshot(("video", query, api.get_collection().count()), build)
    rows, page = slice_page(items, page)
    scores = dict(rows)
    entries = {r["file"]: {**r, "_score": scores[r["file"]]} for r in api.get_meta_by_files(list(scores))}
    frames = api._public_frame_results([entries[file] for file, _ in rows if file in entries])
    # Keep adjacent video runs together without changing global frame ranking.
    videos = []
    for frame in frames:
        video = api._video_id_from_frame(frame["file"])
        if not videos or videos[-1]["video"] != video:
            videos.append(dict(video=video, score=frame["score"], candidate_frame_count=0, frames=[]))
        videos[-1]["frames"].append(frame)
        videos[-1]["candidate_frame_count"] += 1
    return dict(videos=videos, total=len(items), page=page, total_videos=len({api._video_id_from_frame(f) for f, _ in items}),
                candidate_frames=len(items), returned_frames=len(frames), score_type="clip")


def kis_page(session_id, page):
    import main as api
    session = api.get_session(session_id)
    if session is None:
        return JSONResponse({"error": "Session expired; start a new conversation"}, status_code=404)
    rows, page = slice_page(session.pool or [], page)
    return dict(results=api._public_frame_results(rows), total=len(session.pool or []),
                page=page, turn=len(session.turns), session_id=session_id)
