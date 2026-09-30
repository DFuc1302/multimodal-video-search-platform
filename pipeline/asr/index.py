"""
asr_index.py — fixed 2-second ASR indexing
=============================================
Transcribes videos with Whisper and stores speech in fixed 2-second windows.

Each cached window has:
  - text: original text with Vietnamese accents, used for display
  - search_text: lowercase accent-free text, used for matching
  - start/end: exact half-open two-second interval [start, end)

ASR search never extracts images. main.py maps transcript intervals to images
that already exist under keyframes/, using map-keyframes/*.csv.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from redis_store import REDIS_TEXT_ENABLED, store as redis_store
import re
import threading
from pathlib import Path

from tqdm import tqdm

from shared import (
    VIDEO_EXTS,
    ASR_CACHE_FILE,
    log,
    get_whisper,
    normalize_ascii_text,
    unload_whisper,
)
from video_manifest import VideoManifestError, filter_assigned_videos

ASR_CHUNK_SECONDS = float(os.getenv("ASR_CHUNK_SECONDS", "2.0"))
ASR_CACHE_VERSION = 2
ASR_MAX_QUERY_WINDOWS = int(os.getenv("ASR_MAX_QUERY_WINDOWS", "6"))
ASR_CACHE_KEY_PREFIX = "video-name-v1::"
# Unusable transcript; visual retrieval remains available.
ASR_EXCLUDED_VIDEO_IDS = frozenset({"l30_v029"})
_asr_cache_lock = threading.RLock()
_asr_cache_memory: dict | None = None
_asr_cache_signature: tuple[int, int] | None = None


def _cache_signature() -> tuple[int, int] | None:
    try:
        stat = ASR_CACHE_FILE.stat()
        return stat.st_mtime_ns, stat.st_size
    except OSError:
        return None


def normalize_asr_text(text: str) -> str:
    """Canonical ASR search form: lowercase, no accents, normalized spaces."""
    return " ".join(normalize_ascii_text(text or "").lower().split())


def _clean_display_text(text: str) -> str:
    text = " ".join(str(text or "").split())
    return re.sub(r"\s+([,.;:!?%\)\]])", r"\1", text).strip()


def load_asr_cache() -> dict:
    if REDIS_TEXT_ENABLED:
        return {k: v for k, v in redis_store().json('asr_cache', {}).items()
                if not is_asr_excluded_video(asr_cache_video_name(k))}
    global _asr_cache_memory, _asr_cache_signature
    signature = _cache_signature()
    with _asr_cache_lock:
        if _asr_cache_memory is not None and signature == _asr_cache_signature:
            return _asr_cache_memory
        if signature is not None:
            try:
                data = json.loads(ASR_CACHE_FILE.read_text(encoding="utf-8"))
                if not isinstance(data, dict):
                    data = {}

                # Older cache keys contained an absolute path and mtime, so a
                # cache created on one preprocessing machine could not be used
                # on another. Normalize to portable video-name keys in memory.
                normalized = {}
                for raw_key, segments in data.items():
                    name = asr_cache_video_name(raw_key)
                    if is_asr_excluded_video(name):
                        continue
                    if not name:
                        log.warning(f"Ignoring invalid ASR cache key: {raw_key!r}")
                        continue
                    key = f"{ASR_CACHE_KEY_PREFIX}{name}"
                    if key in normalized and normalized[key] != segments:
                        log.warning(
                            f"Conflicting ASR cache entries for {name}; keeping the first one. "
                            "Video filenames must be unique."
                        )
                        continue
                    normalized[key] = segments
                _asr_cache_memory = normalized
            except Exception as exc:
                log.warning(f"Could not read ASR cache {ASR_CACHE_FILE}: {exc}")
                _asr_cache_memory = {}
        else:
            _asr_cache_memory = {}
        _asr_cache_signature = signature
        return _asr_cache_memory


def save_asr_cache(cache: dict) -> None:
    if REDIS_TEXT_ENABLED:
        redis_store().put_json('asr_cache', {k: v for k, v in cache.items()
            if not is_asr_excluded_video(asr_cache_video_name(k))})
        return
    global _asr_cache_memory, _asr_cache_signature
    cache = {key: segments for key, segments in cache.items()
             if not is_asr_excluded_video(asr_cache_video_name(key))}
    ASR_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = ASR_CACHE_FILE.with_suffix(ASR_CACHE_FILE.suffix + ".tmp")
    tmp.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    tmp.replace(ASR_CACHE_FILE)
    with _asr_cache_lock:
        _asr_cache_memory = cache
        _asr_cache_signature = _cache_signature()


def asr_cache_key(video_path: Path) -> str:
    """Return a cache key that remains valid after copying to another machine."""
    return f"{ASR_CACHE_KEY_PREFIX}{video_path.name}"


def asr_cache_video_name(cache_key: str) -> str:
    """Extract the filename from portable and legacy absolute-path keys."""
    key = str(cache_key or "")
    if key.startswith(ASR_CACHE_KEY_PREFIX):
        return key[len(ASR_CACHE_KEY_PREFIX):]
    legacy_path = key.rsplit("::", 1)[0].replace("\\", "/")
    return legacy_path.rsplit("/", 1)[-1]


def is_asr_excluded_video(video: str | Path) -> bool:
    name = str(video).replace("\\", "/").rsplit("/", 1)[-1]
    return Path(name).stem.casefold() in ASR_EXCLUDED_VIDEO_IDS


def is_current_asr_segments(segments: object) -> bool:
    """True only for the fixed-window v2 cache format."""
    if not isinstance(segments, list):
        return False
    if not segments:
        return True

    tolerance = 0.02
    for segment in segments:
        if not isinstance(segment, dict):
            return False
        if segment.get("cache_version") != ASR_CACHE_VERSION:
            return False
        if not isinstance(segment.get("text"), str):
            return False
        if not isinstance(segment.get("search_text"), str):
            return False
        try:
            start = float(segment["start"])
            end = float(segment["end"])
        except (KeyError, TypeError, ValueError):
            return False
        if not math.isfinite(start) or not math.isfinite(end) or start < 0:
            return False
        if abs((end - start) - ASR_CHUNK_SECONDS) > tolerance:
            return False
        aligned = start / ASR_CHUNK_SECONDS
        if abs(aligned - round(aligned)) > tolerance:
            return False
    return True


def _iter_timed_tokens(result: dict):
    """Yield (token, start, end), preferring real Whisper word timestamps."""
    for segment in result.get("segments", []) or []:
        words = segment.get("words") or []
        yielded_real_word = False

        for word in words:
            token = str(word.get("word") or word.get("text") or "")
            if not token.strip():
                continue
            try:
                start = float(word.get("start", segment.get("start", 0.0)))
                end = float(word.get("end", start))
            except (TypeError, ValueError):
                continue
            yielded_real_word = True
            yield token, max(0.0, start), max(start, end)

        if yielded_real_word:
            continue

        display = _clean_display_text(segment.get("text", ""))
        tokens = display.split()
        if not tokens:
            continue
        try:
            seg_start = max(0.0, float(segment.get("start", 0.0)))
            seg_end = max(seg_start, float(segment.get("end", seg_start)))
        except (TypeError, ValueError):
            continue

        duration = max(seg_end - seg_start, 0.001)
        step = duration / len(tokens)
        for index, token in enumerate(tokens):
            yield (
                token,
                seg_start + index * step,
                seg_start + (index + 1) * step,
            )


def build_fixed_2s_segments(result: dict) -> list[dict]:
    """Bin transcribed words into timeline-aligned two-second windows."""
    buckets: dict[int, list[str]] = {}

    for token, start, end in _iter_timed_tokens(result):
        midpoint = max(0.0, (start + end) / 2.0)
        bucket_index = int(midpoint // ASR_CHUNK_SECONDS)
        buckets.setdefault(bucket_index, []).append(token)

    segments: list[dict] = []
    for bucket_index in sorted(buckets):
        display_text = _clean_display_text(" ".join(buckets[bucket_index]))
        search_text = normalize_asr_text(display_text)
        if not search_text:
            continue
        start = round(bucket_index * ASR_CHUNK_SECONDS, 3)
        end = round(start + ASR_CHUNK_SECONDS, 3)
        segments.append({
            "cache_version": ASR_CACHE_VERSION,
            "text": display_text,
            "search_text": search_text,
            "start": start,
            "end": end,
        })
    return segments


def _remove_old_cache_entries_for_video(
    cache: dict,
    video_path: Path,
    keep_key: str,
) -> None:
    for old_key in list(cache):
        if old_key != keep_key and asr_cache_video_name(old_key) == video_path.name:
            cache.pop(old_key, None)


def transcribe_video(video_path: Path, cache: dict) -> list:
    if is_asr_excluded_video(video_path):
        log.info(f"Skipping excluded ASR video: {video_path.name}")
        return []
    key = asr_cache_key(video_path)
    cached = cache.get(key)
    if cached is not None and is_current_asr_segments(cached):
        return cached

    print(
        f"  Transcribing {video_path.name} into {ASR_CHUNK_SECONDS:g}s windows...",
        flush=True,
    )

    try:
        import torch
        use_fp16 = torch.cuda.is_available()
    except Exception:
        use_fp16 = False

    model = get_whisper()
    common_args = {
        "verbose": False,
        "fp16": use_fp16,
        "condition_on_previous_text": False,
    }

    try:
        result = model.transcribe(
            str(video_path),
            word_timestamps=True,
            **common_args,
        )
    except Exception as exc:
        log.warning(
            f"Word timestamps failed for {video_path.name}: {exc} — "
            "retrying without word timestamps"
        )
        result = model.transcribe(
            str(video_path),
            word_timestamps=False,
            **common_args,
        )

    segments = build_fixed_2s_segments(result)
    _remove_old_cache_entries_for_video(cache, video_path, key)
    cache[key] = segments
    save_asr_cache(cache)
    return segments


def _join_segment_text(segments: list[dict]) -> tuple[str, str]:
    display = _clean_display_text(
        " ".join(segment.get("text", "") for segment in segments)
    )
    search = normalize_asr_text(
        " ".join(segment.get("search_text", "") for segment in segments)
    )
    return display, search


def find_phrase_segments(segments: list, phrase: str) -> list:
    """
    Search accent-insensitively. Direct two-second hits are preferred; a
    rolling-window fallback handles phrases crossing a two-second boundary.
    """
    needle = normalize_asr_text(phrase)
    if not needle or not is_current_asr_segments(segments):
        return []

    ordered = sorted(segments, key=lambda item: float(item.get("start", 0.0)))
    direct = [
        dict(segment)
        for segment in ordered
        if needle in normalize_asr_text(
            segment.get("search_text") or segment.get("text", "")
        )
    ]

    if direct:
        merged: list[dict] = []
        for segment in direct:
            if not merged:
                merged.append(segment)
                continue

            previous = merged[-1]
            if float(segment["start"]) <= float(previous["end"]) + 0.02:
                previous["end"] = max(
                    float(previous["end"]),
                    float(segment["end"]),
                )
                previous["text"] = _clean_display_text(
                    f"{previous.get('text', '')} {segment.get('text', '')}"
                )
                previous["search_text"] = normalize_asr_text(
                    f"{previous.get('search_text', '')} "
                    f"{segment.get('search_text', '')}"
                )
            else:
                merged.append(segment)
        return merged

    rolling_hits: list[dict] = []
    for start_index in range(len(ordered)):
        span: list[dict] = []
        for end_index in range(
            start_index,
            min(len(ordered), start_index + ASR_MAX_QUERY_WINDOWS),
        ):
            current = ordered[end_index]
            if span and float(current["start"]) > float(span[-1]["end"]) + 0.02:
                break

            span.append(current)
            display, search = _join_segment_text(span)
            if needle in search:
                rolling_hits.append({
                    "cache_version": ASR_CACHE_VERSION,
                    "text": display,
                    "search_text": search,
                    "start": float(span[0]["start"]),
                    "end": float(span[-1]["end"]),
                })
                break

    deduped: list[dict] = []
    for hit in rolling_hits:
        if any(
            float(existing["start"]) <= float(hit["start"])
            and float(existing["end"]) >= float(hit["end"])
            for existing in deduped
        ):
            continue
        deduped.append(hit)
    return deduped


def transcribe_folder(
    folder: str,
    video_list: str | Path | None = None,
    shard_id: str | None = None,
) -> None:
    folder_path = Path(folder)
    if not folder_path.exists():
        log.error(f"Folder not found: {folder}")
        return

    candidates = (
        path
        for path in folder_path.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTS
    )
    try:
        videos = filter_assigned_videos(
            candidates, folder_path, video_list, shard_id
        )
    except VideoManifestError as exc:
        raise RuntimeError(str(exc)) from exc
    cache = load_asr_cache()
    done = 0
    skipped = 0
    failed = 0

    for video in tqdm(videos, desc="Transcribing", unit="video"):
        if is_asr_excluded_video(video):
            skipped += 1
            continue
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
    log.info(
        f"=== ASR indexing complete — {done} transcribed, {skipped} cached/excluded, "
        f"{failed} failed, {len(videos)} total; "
        f"window={ASR_CHUNK_SECONDS:g}s ==="
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Transcribe videos into fixed two-second searchable windows."
    )
    parser.add_argument("--folder", required=True)
    parser.add_argument("--video-list")
    parser.add_argument("--shard-id")
    arguments = parser.parse_args(argv)
    transcribe_folder(
        arguments.folder,
        video_list=arguments.video_list,
        shard_id=arguments.shard_id,
    )


if __name__ == "__main__":
    main()
