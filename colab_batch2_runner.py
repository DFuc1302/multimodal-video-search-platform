#!/usr/bin/env python3
"""Drive-backed Colab runner for Batch 2 frame extraction and optional ASR.

The heavy work always happens on the Colab VM.  Only validated, complete
outputs are published to Drive.  Frame extraction and ASR deliberately reuse
the project's existing ``re_extract.py`` and ``asr_index.py`` logic.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
import uuid
from fractions import Fraction
from pathlib import Path

from re_extract import (
    VIDEO_EXTENSIONS,
    build_frame_indices,
    extract_video,
    fps_for_video,
    numbered_images,
    validate_numbering,
)
from video_manifest import VideoManifestError, filter_assigned_videos


DEFAULT_INTERVAL = Fraction(1, 2)
DEFAULT_QUALITY = 2
DEFAULT_DIGITS = 4
DEFAULT_THREADS = 8
STATE_DIRNAME = ".colab-state"
PUBLISH_MARKER = ".colab-publish.json"


class ColabPipelineError(RuntimeError):
    pass


def _json_read(path: Path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default
    except (OSError, json.JSONDecodeError) as exc:
        raise ColabPipelineError(f"Không đọc được JSON {path}: {exc}") from exc


def _json_atomic_write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex[:8]}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _safe_id(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in value)
    cleaned = cleaned.strip("_")
    if not cleaned:
        raise ColabPipelineError("SHARD_ID không hợp lệ")
    return cleaned


def _source_fingerprint(video: Path, video_root: Path) -> dict[str, object]:
    stat = video.stat()
    return {
        "relative_path": video.relative_to(video_root).as_posix(),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def discover_videos(
    video_root: Path,
    manifest_path: Path | None = None,
    shard_id: str | None = None,
) -> list[Path]:
    if not video_root.is_dir():
        raise ColabPipelineError(f"Không tìm thấy thư mục video: {video_root}")
    candidates = sorted(
        path
        for path in video_root.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )
    if not candidates:
        raise ColabPipelineError(f"Không tìm thấy video dưới {video_root}")
    try:
        # Validate globally before selecting one worker's assignment.  This
        # prevents two different source paths from publishing the same stem.
        filter_assigned_videos(candidates, video_root, None, None)
        selected = filter_assigned_videos(
            candidates,
            video_root,
            manifest_path,
            shard_id,
        )
    except VideoManifestError as exc:
        raise ColabPipelineError(str(exc)) from exc
    if not selected:
        raise ColabPipelineError(
            f"Manifest không gán video nào cho shard {shard_id!r}"
        )
    return selected


def validate_frame_output(
    stem: str,
    frames_dir: Path,
    csv_path: Path,
    *,
    expected_count: int | None = None,
) -> int:
    if not frames_dir.is_dir():
        raise ColabPipelineError(f"Thiếu thư mục keyframe: {frames_dir}")
    if not csv_path.is_file():
        raise ColabPipelineError(f"Thiếu map CSV: {csv_path}")
    images = numbered_images(frames_dir, stem)
    try:
        validate_numbering(images)
    except RuntimeError as exc:
        raise ColabPipelineError(f"{frames_dir}: {exc}") from exc
    if expected_count is not None and len(images) != expected_count:
        raise ColabPipelineError(
            f"{stem}: keyframe={len(images)} nhưng receipt={expected_count}"
        )
    try:
        with csv_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            if reader.fieldnames != ["n", "pts_time", "fps", "frame_idx"]:
                raise ColabPipelineError(
                    f"Map CSV sai cột tại {csv_path}: {reader.fieldnames}"
                )
    except (OSError, csv.Error) as exc:
        raise ColabPipelineError(f"Không đọc được map CSV {csv_path}: {exc}") from exc
    if len(rows) != len(images):
        raise ColabPipelineError(
            f"{stem}: keyframe={len(images)} nhưng map={len(rows)}"
        )
    fps = fps_for_video(stem)
    expected_indices = build_frame_indices(len(images), DEFAULT_INTERVAL, fps)
    for position, (row, expected_frame_idx) in enumerate(
        zip(rows, expected_indices), start=1
    ):
        try:
            actual_n = int(row["n"])
            actual_frame_idx = int(row["frame_idx"])
            actual_fps = Fraction(str(row["fps"]))
        except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
            raise ColabPipelineError(
                f"Map CSV không hợp lệ tại dòng {position + 1}: {csv_path}"
            ) from exc
        if (
            actual_n != position
            or actual_frame_idx != expected_frame_idx
            or actual_fps != fps
        ):
            raise ColabPipelineError(
                f"Map CSV không khớp logic laptop tại dòng {position + 1}: {csv_path}"
            )
    return len(images)


def _receipt_path(batch_root: Path, shard_id: str, stem: str) -> Path:
    return batch_root / STATE_DIRNAME / _safe_id(shard_id) / f"{stem}.json"


def _published_is_complete(
    video: Path,
    video_root: Path,
    batch_root: Path,
    shard_id: str,
) -> tuple[bool, int]:
    receipt_path = _receipt_path(batch_root, shard_id, video.stem)
    receipt = _json_read(receipt_path)
    if not isinstance(receipt, dict):
        return False, 0
    if receipt.get("source") != _source_fingerprint(video, video_root):
        return False, 0
    expected_count = receipt.get("frame_count")
    if not isinstance(expected_count, int) or expected_count <= 0:
        return False, 0
    count = validate_frame_output(
        video.stem,
        batch_root / "keyframes" / video.stem,
        batch_root / "map-keyframes" / f"{video.stem}.csv",
        expected_count=expected_count,
    )
    return True, count


def _adopt_existing_output(
    video: Path,
    video_root: Path,
    batch_root: Path,
    shard_id: str,
) -> int:
    count = validate_frame_output(
        video.stem,
        batch_root / "keyframes" / video.stem,
        batch_root / "map-keyframes" / f"{video.stem}.csv",
    )
    _json_atomic_write(
        _receipt_path(batch_root, shard_id, video.stem),
        {
            "format_version": 1,
            "stage": "extract",
            "source": _source_fingerprint(video, video_root),
            "frame_count": count,
            "interval_seconds": 0.5,
            "jpeg_quality": DEFAULT_QUALITY,
            "digits": DEFAULT_DIGITS,
            "adopted": True,
            "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
    )
    return count


def _estimate_required_bytes(video: Path) -> int:
    """Conservative per-video workspace estimate using ffprobe duration."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video),
        ],
        capture_output=True,
        text=True,
    )
    try:
        duration = max(0.0, float(result.stdout.strip())) if result.returncode == 0 else 0.0
    except ValueError:
        duration = 0.0
    estimated_frames = math.ceil(duration / float(DEFAULT_INTERVAL))
    estimated_jpeg_bytes = estimated_frames * 500_000
    return int(video.stat().st_size + estimated_jpeg_bytes + 2 * 1024**3)


def _ensure_workspace_capacity(work_root: Path, video: Path) -> None:
    work_root.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(work_root).free
    required = _estimate_required_bytes(video)
    if free < required:
        raise ColabPipelineError(
            f"Không đủ ổ tạm cho {video.name}: cần ước tính "
            f"{required / 1024**3:.1f} GiB, còn {free / 1024**3:.1f} GiB"
        )


def _copy_video_to_workspace(video: Path, work_root: Path) -> Path:
    _ensure_workspace_capacity(work_root, video)
    input_dir = work_root / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    local_video = input_dir / video.name
    shutil.copy2(video, local_video)
    if local_video.stat().st_size != video.stat().st_size:
        raise ColabPipelineError(f"Copy video không đủ byte: {video}")
    return local_video


def _publish_frame_output(
    *,
    video: Path,
    video_root: Path,
    local_frames: Path,
    local_csv: Path,
    batch_root: Path,
    shard_id: str,
    frame_count: int,
) -> None:
    keyframes_root = batch_root / "keyframes"
    maps_root = batch_root / "map-keyframes"
    partial_root = batch_root / STATE_DIRNAME / _safe_id(shard_id) / "partial"
    keyframes_root.mkdir(parents=True, exist_ok=True)
    maps_root.mkdir(parents=True, exist_ok=True)
    partial_root.mkdir(parents=True, exist_ok=True)

    final_frames = keyframes_root / video.stem
    final_csv = maps_root / f"{video.stem}.csv"
    if final_frames.exists() or final_csv.exists():
        raise ColabPipelineError(
            f"Output {video.stem} đã tồn tại nhưng chưa có receipt hợp lệ; "
            "không ghi đè. Hãy kiểm tra hoặc dùng ADOPT_EXISTING=True."
        )

    token = uuid.uuid4().hex[:8]
    partial_frames = partial_root / f"{video.stem}-{token}"
    partial_csv = partial_root / f"{video.stem}-{token}.csv"
    try:
        shutil.copytree(local_frames, partial_frames)
        shutil.copy2(local_csv, partial_csv)
        _json_atomic_write(
            partial_frames / PUBLISH_MARKER,
            {
                "source": _source_fingerprint(video, video_root),
                "frame_count": frame_count,
            },
        )
        validate_frame_output(
            video.stem,
            partial_frames,
            partial_csv,
            expected_count=frame_count,
        )
        os.replace(partial_csv, final_csv)
        os.replace(partial_frames, final_frames)
        _json_atomic_write(
            _receipt_path(batch_root, shard_id, video.stem),
            {
                "format_version": 1,
                "stage": "extract",
                "source": _source_fingerprint(video, video_root),
                "frame_count": frame_count,
                "interval_seconds": 0.5,
                "jpeg_quality": DEFAULT_QUALITY,
                "digits": DEFAULT_DIGITS,
                "completed_at": time.strftime(
                    "%Y-%m-%dT%H:%M:%SZ", time.gmtime()
                ),
            },
        )
        (final_frames / PUBLISH_MARKER).unlink(missing_ok=True)
    finally:
        if partial_frames.exists():
            shutil.rmtree(partial_frames, ignore_errors=True)
        partial_csv.unlink(missing_ok=True)


def preflight(
    *,
    batch_root: Path,
    video_root: Path,
    manifest_path: Path | None,
    shard_id: str,
    adopt_existing: bool = False,
) -> dict[str, int]:
    if "drive" not in {part.casefold() for part in batch_root.parts}:
        print(
            "[CẢNH BÁO] BATCH_ROOT không có thành phần 'drive'; "
            "hãy chắc chắn đây là Google Drive đã mount."
        )
    if manifest_path and not manifest_path.is_file():
        raise ColabPipelineError(f"Không tìm thấy manifest: {manifest_path}")
    videos = discover_videos(video_root, manifest_path, shard_id if manifest_path else None)
    (batch_root / "keyframes").mkdir(parents=True, exist_ok=True)
    (batch_root / "map-keyframes").mkdir(parents=True, exist_ok=True)
    stats = {"selected": len(videos), "complete": 0, "pending": 0, "conflict": 0}
    for video in videos:
        try:
            complete, _ = _published_is_complete(
                video, video_root, batch_root, shard_id
            )
        except ColabPipelineError as exc:
            print(f"[XUNG ĐỘT] {video.name}: {exc}")
            stats["conflict"] += 1
            continue
        if complete:
            stats["complete"] += 1
            continue
        final_frames = batch_root / "keyframes" / video.stem
        final_csv = batch_root / "map-keyframes" / f"{video.stem}.csv"
        if final_frames.exists() or final_csv.exists():
            if adopt_existing and final_frames.is_dir() and final_csv.is_file():
                _adopt_existing_output(
                    video, video_root, batch_root, shard_id
                )
                stats["complete"] += 1
            else:
                print(
                    f"[XUNG ĐỘT] {video.name}: output có sẵn nhưng thiếu receipt"
                )
                stats["conflict"] += 1
        else:
            stats["pending"] += 1
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    if stats["conflict"]:
        raise ColabPipelineError(
            "Preflight phát hiện output xung đột; không bắt đầu xử lý."
        )
    return stats


def run_extract(
    *,
    batch_root: Path,
    video_root: Path,
    work_root: Path,
    manifest_path: Path | None,
    shard_id: str,
    adopt_existing: bool = False,
    threads: int = DEFAULT_THREADS,
) -> dict[str, int]:
    work_root.mkdir(parents=True, exist_ok=True)
    preflight(
        batch_root=batch_root,
        video_root=video_root,
        manifest_path=manifest_path,
        shard_id=shard_id,
        adopt_existing=adopt_existing,
    )
    videos = discover_videos(video_root, manifest_path, shard_id if manifest_path else None)
    stats = {"processed": 0, "skipped": 0, "failed": 0, "frames": 0}
    for position, video in enumerate(videos, start=1):
        try:
            complete, count = _published_is_complete(
                video, video_root, batch_root, shard_id
            )
            if complete:
                stats["skipped"] += 1
                stats["frames"] += count
                print(f"[{position}/{len(videos)}] BỎ QUA {video.name}: đã hoàn tất")
                continue
            print(f"[{position}/{len(videos)}] CẮT {video.name}", flush=True)
            with tempfile.TemporaryDirectory(prefix="aic-batch2-", dir=work_root) as tmp:
                tmp_root = Path(tmp)
                local_video = _copy_video_to_workspace(video, tmp_root)
                local_keyframes_root = tmp_root / "keyframes"
                local_maps_root = tmp_root / "map-keyframes"
                frames_dir, csv_path, frame_count = extract_video(
                    video=local_video,
                    fps=fps_for_video(video.stem),
                    interval=DEFAULT_INTERVAL,
                    keyframes_root=local_keyframes_root,
                    maps_root=local_maps_root,
                    digits=DEFAULT_DIGITS,
                    quality=DEFAULT_QUALITY,
                    threads=threads,
                    force=False,
                )
                validate_frame_output(
                    video.stem, frames_dir, csv_path, expected_count=frame_count
                )
                _publish_frame_output(
                    video=video,
                    video_root=video_root,
                    local_frames=frames_dir,
                    local_csv=csv_path,
                    batch_root=batch_root,
                    shard_id=shard_id,
                    frame_count=frame_count,
                )
            stats["processed"] += 1
            stats["frames"] += frame_count
            print(f"[XONG] {video.name}: {frame_count} frame", flush=True)
        except Exception as exc:
            stats["failed"] += 1
            print(f"[LỖI] {video.name}: {exc}", file=sys.stderr, flush=True)
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    if stats["failed"]:
        raise ColabPipelineError(
            f"Có {stats['failed']} video cắt frame thất bại; chạy lại để resume."
        )
    return stats


def _publish_asr_cache(local_cache: Path, drive_cache: Path) -> None:
    payload = _json_read(local_cache)
    if not isinstance(payload, dict):
        raise ColabPipelineError(f"ASR cache không hợp lệ: {local_cache}")
    drive_cache.parent.mkdir(parents=True, exist_ok=True)
    temporary = drive_cache.with_name(
        f".{drive_cache.name}.tmp-{uuid.uuid4().hex[:8]}"
    )
    shutil.copy2(local_cache, temporary)
    if _json_read(temporary) != payload:
        temporary.unlink(missing_ok=True)
        raise ColabPipelineError("ASR cache copy lên Drive không khớp")
    os.replace(temporary, drive_cache)


def run_asr(
    *,
    batch_root: Path,
    video_root: Path,
    work_root: Path,
    manifest_path: Path | None,
    shard_id: str,
) -> dict[str, int]:
    videos = discover_videos(video_root, manifest_path, shard_id if manifest_path else None)
    safe_shard = _safe_id(shard_id)
    drive_cache = batch_root / "asr-shards" / safe_shard / "asr_cache.json"
    work_root.mkdir(parents=True, exist_ok=True)
    local_cache = work_root / f"asr_cache_{safe_shard}.json"
    if drive_cache.is_file():
        shutil.copy2(drive_cache, local_cache)
    elif not local_cache.exists():
        _json_atomic_write(local_cache, {})

    # shared.py reads these paths at import time, so set them first.
    os.environ["DATA_ROOT"] = str(work_root)
    os.environ["ASR_CACHE_FILE"] = str(local_cache)
    os.environ.setdefault("WHISPER_MODEL", "small")
    import asr_index

    cache = asr_index.load_asr_cache()
    stats = {"processed": 0, "skipped": 0, "failed": 0}
    for position, video in enumerate(videos, start=1):
        key = asr_index.asr_cache_key(video)
        if key in cache and asr_index.is_current_asr_segments(cache[key]):
            stats["skipped"] += 1
            print(f"[{position}/{len(videos)}] BỎ QUA ASR {video.name}: đã cache")
            continue
        try:
            print(f"[{position}/{len(videos)}] ASR {video.name}", flush=True)
            with tempfile.TemporaryDirectory(prefix="aic-asr-", dir=work_root) as tmp:
                local_video = _copy_video_to_workspace(video, Path(tmp))
                asr_index.transcribe_video(local_video, cache)
            _publish_asr_cache(local_cache, drive_cache)
            stats["processed"] += 1
        except Exception as exc:
            stats["failed"] += 1
            print(f"[LỖI ASR] {video.name}: {exc}", file=sys.stderr, flush=True)
    asr_index.unload_whisper()
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    if stats["failed"]:
        raise ColabPipelineError(
            f"Có {stats['failed']} video ASR thất bại; chạy lại để resume."
        )
    return stats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "extract", "asr"))
    parser.add_argument("--batch-root", type=Path, required=True)
    parser.add_argument("--video-dir", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, default=Path("/content/aic_batch2"))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--shard-id", default="batch2_all")
    parser.add_argument("--adopt-existing", action="store_true")
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    batch_root = args.batch_root.expanduser().resolve()
    video_root = args.video_dir.expanduser().resolve()
    work_root = args.work_root.expanduser().resolve()
    manifest = args.manifest.expanduser().resolve() if args.manifest else None
    common = {
        "batch_root": batch_root,
        "video_root": video_root,
        "manifest_path": manifest,
        "shard_id": args.shard_id,
    }
    try:
        if args.command == "preflight":
            preflight(**common, adopt_existing=args.adopt_existing)
        elif args.command == "extract":
            run_extract(
                **common,
                work_root=work_root,
                adopt_existing=args.adopt_existing,
                threads=args.threads,
            )
        else:
            run_asr(**common, work_root=work_root)
        return 0
    except (ColabPipelineError, OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
