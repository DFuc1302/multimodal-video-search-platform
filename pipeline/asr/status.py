#!/usr/bin/env python3
"""Report which configured videos have current, outdated, or missing ASR cache."""

from __future__ import annotations

import json
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CONFIG_FILE = REPOSITORY_ROOT / "dataset_roots.json"
REPORT_DIR = REPOSITORY_ROOT / "asr_reports"

VIDEO_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".webm",
    ".m4v",
    ".mpg",
    ".mpeg",
}

ASR_CACHE_VERSION = 2
ASR_CHUNK_SECONDS = 2.0


def resolve_path(value: str, base: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def cache_video_name(raw_key: object) -> str:
    """Extract a filename from portable and legacy ASR cache keys."""
    key = str(raw_key or "")
    prefix = "video-name-v1::"

    if key.startswith(prefix):
        return key[len(prefix) :]

    legacy_path = key.rsplit("::", 1)[0].replace("\\", "/")
    return legacy_path.rsplit("/", 1)[-1]


def is_current_asr(segments: object) -> bool:
    """Match the ASR v2 validity rules used by asr_index.py."""
    if not isinstance(segments, list):
        return False

    # An empty list is valid: Whisper ran but found no speech.
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

        if abs((end - start) - ASR_CHUNK_SECONDS) > tolerance:
            return False

        aligned = start / ASR_CHUNK_SECONDS
        if abs(aligned - round(aligned)) > tolerance:
            return False

    return True


def load_configuration() -> list[dict[str, object]]:
    config = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    base = CONFIG_FILE.parent
    shards: list[dict[str, object]] = []

    for shard in config.get("shards", []):
        if not shard.get("enabled", True):
            continue

        shards.append(
            {
                "id": shard.get("id", "unknown"),
                "videos_dir": resolve_path(shard["videos_dir"], base),
                "data_root": resolve_path(shard["data_root"], base),
            }
        )

    return shards


def find_cache_files(data_root: Path) -> list[Path]:
    candidates = [
        data_root / "asr_cache.json",
        data_root / "search_index" / "asr_cache.json",
    ]
    candidates.extend(data_root.glob("processed/*/asr_cache.json"))

    return sorted(
        {
            path.resolve()
            for path in candidates
            if path.is_file()
        }
    )


def write_report(filename: str, videos: list[Path]) -> Path:
    output_file = REPORT_DIR / filename
    lines = [str(video) for video in videos]
    output_file.write_text(
        "\n".join(lines) + ("\n" if lines else ""),
        encoding="utf-8",
    )
    return output_file


def main() -> None:
    shards = load_configuration()
    videos_by_name: dict[str, list[Path]] = {}
    cache_files: set[Path] = set()

    for shard in shards:
        shard_id = str(shard["id"])
        videos_dir = Path(shard["videos_dir"])
        data_root = Path(shard["data_root"])

        if not videos_dir.is_dir():
            print(f"CẢNH BÁO: Không truy cập được shard {shard_id}: {videos_dir}")
            continue

        for video in videos_dir.rglob("*"):
            if video.is_file() and video.suffix.lower() in VIDEO_EXTENSIONS:
                videos_by_name.setdefault(video.name, []).append(video.resolve())

        cache_files.update(find_cache_files(data_root))

    valid_cache_names: set[str] = set()
    invalid_cache_names: set[str] = set()

    for cache_file in sorted(cache_files):
        try:
            cache = json.loads(cache_file.read_text(encoding="utf-8"))
        except Exception as error:
            print(f"CẢNH BÁO: Không đọc được {cache_file}: {error}")
            continue

        if not isinstance(cache, dict):
            print(f"CẢNH BÁO: Cache không hợp lệ: {cache_file}")
            continue

        for raw_key, segments in cache.items():
            video_name = cache_video_name(raw_key)
            if is_current_asr(segments):
                valid_cache_names.add(video_name)
            else:
                invalid_cache_names.add(video_name)

    all_videos = sorted(
        video
        for paths in videos_by_name.values()
        for video in paths
    )
    asr_done = sorted(
        video
        for name, paths in videos_by_name.items()
        if name in valid_cache_names
        for video in paths
    )
    asr_outdated = sorted(
        video
        for name, paths in videos_by_name.items()
        if name not in valid_cache_names and name in invalid_cache_names
        for video in paths
    )
    asr_missing = sorted(
        video
        for name, paths in videos_by_name.items()
        if name not in valid_cache_names and name not in invalid_cache_names
        for video in paths
    )
    duplicate_names = {
        name: paths
        for name, paths in videos_by_name.items()
        if len(paths) > 1
    }

    print("\n========== THỐNG KÊ ASR ==========")
    print(f"Tổng số file video:          {len(all_videos)}")
    print(f"Số tên video duy nhất:       {len(videos_by_name)}")
    print(f"Đã ASR hợp lệ:               {len(asr_done)}")
    print(f"Cache cũ/lỗi, cần ASR lại:   {len(asr_outdated)}")
    print(f"Chưa có ASR:                 {len(asr_missing)}")
    print(f"Tên video bị trùng:          {len(duplicate_names)}")
    print(f"Số tệp cache đã kiểm tra:    {len(cache_files)}")

    print("\n--- ĐÃ ASR ---")
    for video in asr_done:
        print(video)

    print("\n--- CACHE CŨ/LỖI, CẦN ASR LẠI ---")
    for video in asr_outdated:
        print(video)

    print("\n--- CHƯA ASR ---")
    for video in asr_missing:
        print(video)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    all_file = write_report("videos_all.txt", all_videos)
    done_file = write_report("videos_asr_done.txt", asr_done)
    outdated_file = write_report("videos_asr_outdated.txt", asr_outdated)
    missing_file = write_report("videos_asr_missing.txt", asr_missing)

    duplicate_file = REPORT_DIR / "duplicate_video_names.txt"
    duplicate_lines = [
        f"{name}\n" + "\n".join(f"  {path}" for path in paths)
        for name, paths in sorted(duplicate_names.items())
    ]
    duplicate_file.write_text(
        "\n".join(duplicate_lines) + ("\n" if duplicate_lines else ""),
        encoding="utf-8",
    )

    print("\n========== FILE BÁO CÁO ==========")
    print(f"Tất cả video:       {all_file}")
    print(f"Đã ASR:             {done_file}")
    print(f"Cần ASR lại:        {outdated_file}")
    print(f"Chưa ASR:           {missing_file}")
    print(f"Tên bị trùng:       {duplicate_file}")


if __name__ == "__main__":
    main()

