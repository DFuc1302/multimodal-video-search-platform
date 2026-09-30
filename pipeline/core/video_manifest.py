"""Portable video-to-shard assignment manifests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable
from asset_paths import relative_asset_path


class VideoManifestError(RuntimeError):
    pass


def _normalize_relative(value: str) -> str:
    normalized = relative_asset_path(value)
    if normalized is None:
        raise VideoManifestError(f"Đường dẫn video không hợp lệ trong manifest: {value!r}")
    return normalized


def load_assignments(path: str | Path) -> dict[str, str]:
    manifest_path = Path(path)
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VideoManifestError(f"Không đọc được video manifest {manifest_path}: {exc}") from exc
    raw = payload.get("assignments") if isinstance(payload, dict) else None
    if not isinstance(raw, dict):
        raise VideoManifestError("Video manifest phải có object assignments")
    result: dict[str, str] = {}
    for filename, shard_id in raw.items():
        relative = _normalize_relative(filename)
        if not isinstance(shard_id, str) or not shard_id.strip():
            raise VideoManifestError(f"Shard id không hợp lệ cho {relative}")
        result[relative] = shard_id.strip()
    return result


def filter_assigned_videos(
    candidates: Iterable[Path],
    root: Path,
    manifest_path: str | Path | None,
    shard_id: str | None,
) -> list[Path]:
    videos = sorted(candidates)
    if manifest_path is not None:
        if not shard_id:
            raise VideoManifestError("--shard-id là bắt buộc khi dùng --video-list")
        assignments = load_assignments(manifest_path)
        wanted = {name for name, assigned in assignments.items() if assigned == shard_id}
        by_relative = {
            path.relative_to(root).as_posix(): path
            for path in videos
        }
        missing = sorted(wanted - set(by_relative))
        if missing:
            raise VideoManifestError(
                f"Shard {shard_id} được gán {len(missing)} video không tồn tại dưới {root}: "
                + ", ".join(missing[:10])
            )
        videos = [by_relative[name] for name in sorted(wanted)]

    by_stem: dict[str, list[Path]] = {}
    for video in videos:
        by_stem.setdefault(video.stem, []).append(video)
    duplicates = {stem: paths for stem, paths in by_stem.items() if len(paths) > 1}
    if duplicates:
        stem, paths = next(iter(sorted(duplicates.items())))
        raise VideoManifestError(
            f"Tên video không duy nhất {stem!r}; output keyframe/ASR sẽ bị trùng: "
            + ", ".join(str(path) for path in paths[:3])
        )
    return videos
