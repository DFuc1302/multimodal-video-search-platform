#!/usr/bin/env python3
"""Validate and import Colab Batch 2 outputs into a separate USB data root.

The command is dry-run by default.  Pass ``--apply`` only after reviewing the
plan.  Existing video IDs are never overwritten or mixed with the new batch.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import uuid
from pathlib import Path

from colab_batch2_runner import ColabPipelineError, validate_frame_output


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


class ImportError(RuntimeError):
    pass


def video_stems(root: Path) -> set[str]:
    keyframes = root / "keyframes"
    if not keyframes.is_dir():
        return set()
    return {
        path.name.casefold()
        for path in keyframes.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    }


def validate_source(source: Path) -> list[tuple[str, Path, Path, int]]:
    keyframes = source / "keyframes"
    maps = source / "map-keyframes"
    if not keyframes.is_dir() or not maps.is_dir():
        raise ImportError(
            f"Source phải có keyframes/ và map-keyframes/: {source}"
        )
    rows: list[tuple[str, Path, Path, int]] = []
    for frame_dir in sorted(keyframes.iterdir()):
        if not frame_dir.is_dir() or frame_dir.name.startswith("."):
            continue
        stem = frame_dir.name
        csv_path = maps / f"{stem}.csv"
        try:
            count = validate_frame_output(stem, frame_dir, csv_path)
        except ColabPipelineError as exc:
            raise ImportError(str(exc)) from exc
        rows.append((stem, frame_dir, csv_path, count))
    if not rows:
        raise ImportError(f"Không có video hợp lệ trong {keyframes}")
    map_stems = {
        path.stem.casefold()
        for path in maps.glob("*.csv")
        if path.is_file()
    }
    frame_stems = {stem.casefold() for stem, *_ in rows}
    orphan_maps = sorted(map_stems - frame_stems)
    if orphan_maps:
        raise ImportError(
            "Có map CSV không có thư mục keyframe: " + ", ".join(orphan_maps[:10])
        )
    return rows


def merge_asr_shards(source: Path) -> dict:
    from asr_index import asr_cache_video_name, is_current_asr_segments

    merged: dict = {}
    candidates: list[Path] = []
    direct = source / "asr_cache.json"
    if direct.is_file():
        candidates.append(direct)
    shard_root = source / "asr-shards"
    if shard_root.is_dir():
        candidates.extend(sorted(shard_root.glob("*/asr_cache.json")))
    for path in candidates:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ImportError(f"Không đọc được ASR cache {path}: {exc}") from exc
        if not isinstance(payload, dict):
            raise ImportError(f"ASR cache phải là object JSON: {path}")
        for key, segments in payload.items():
            if not asr_cache_video_name(str(key)):
                raise ImportError(f"ASR cache có key video rỗng trong {path}")
            if not is_current_asr_segments(segments):
                raise ImportError(
                    f"ASR cache không đúng định dạng cửa sổ 2 giây cho {key!r}: {path}"
                )
            if key in merged and merged[key] != segments:
                raise ImportError(
                    f"ASR xung đột cho {key!r} giữa các cache Colab"
                )
            merged[key] = segments
    return merged


def _write_json_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex[:8]}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def import_outputs(
    *,
    source: Path,
    destination: Path,
    existing_roots: list[Path],
    apply: bool,
    merge_asr: bool,
) -> dict[str, int]:
    rows = validate_source(source)
    source_stems = {stem.casefold() for stem, *_ in rows}
    duplicates: dict[str, list[str]] = {}
    for root in existing_roots:
        for stem in sorted(source_stems & video_stems(root)):
            duplicates.setdefault(stem, []).append(str(root))
    destination_existing = video_stems(destination)
    for stem in sorted(source_stems & destination_existing):
        duplicates.setdefault(stem, []).append(str(destination))
    if duplicates:
        sample = "; ".join(
            f"{stem}: {', '.join(roots)}"
            for stem, roots in list(sorted(duplicates.items()))[:10]
        )
        raise ImportError(f"Video ID đã tồn tại; không ghi đè: {sample}")

    total_frames = sum(count for *_, count in rows)
    total_bytes = sum(
        path.stat().st_size
        for _, frame_dir, csv_path, _ in rows
        for path in [*frame_dir.iterdir(), csv_path]
        if path.is_file()
    )
    asr_payload = merge_asr_shards(source) if merge_asr else {}
    destination_cache = destination / "asr_cache.json"
    existing = {}
    if merge_asr and asr_payload:
        if destination_cache.is_file():
            try:
                existing = json.loads(destination_cache.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ImportError(
                    f"Không đọc được ASR cache đích {destination_cache}: {exc}"
                ) from exc
            if not isinstance(existing, dict):
                raise ImportError(f"ASR cache đích không hợp lệ: {destination_cache}")
        for key, segments in asr_payload.items():
            if key in existing and existing[key] != segments:
                raise ImportError(f"ASR đích xung đột cho {key!r}")
            existing[key] = segments
    plan = {
        "videos": len(rows),
        "frames": total_frames,
        "bytes": total_bytes,
        "asr_videos": len(asr_payload),
    }
    print(json.dumps(plan, ensure_ascii=False, indent=2))
    if not apply:
        print("DRY-RUN: chưa chép file; thêm --apply sau khi kiểm tra kế hoạch.")
        return plan

    keyframes_dest = destination / "keyframes"
    maps_dest = destination / "map-keyframes"
    staging = destination / f".batch2-import-{uuid.uuid4().hex[:8]}"
    staging_keyframes = staging / "keyframes"
    staging_maps = staging / "map-keyframes"
    try:
        staging_keyframes.mkdir(parents=True, exist_ok=False)
        staging_maps.mkdir(parents=True, exist_ok=False)
        for stem, frame_dir, csv_path, count in rows:
            target_frames = staging_keyframes / stem
            target_csv = staging_maps / f"{stem}.csv"
            shutil.copytree(frame_dir, target_frames)
            shutil.copy2(csv_path, target_csv)
            try:
                validate_frame_output(
                    stem, target_frames, target_csv, expected_count=count
                )
            except ColabPipelineError as exc:
                raise ImportError(str(exc)) from exc

        keyframes_dest.mkdir(parents=True, exist_ok=True)
        maps_dest.mkdir(parents=True, exist_ok=True)
        for stem, _, _, _ in rows:
            os.replace(staging_keyframes / stem, keyframes_dest / stem)
            os.replace(
                staging_maps / f"{stem}.csv", maps_dest / f"{stem}.csv"
            )
        if merge_asr and asr_payload:
            _write_json_atomic(destination_cache, existing)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    print(f"Đã import an toàn vào {destination}")
    return plan


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path("/mnt/d/AIC/data/processed/BATCH2"),
    )
    parser.add_argument(
        "--existing-root",
        type=Path,
        action="append",
        default=[],
        help="Dataset cũ cần kiểm tra trùng video ID; có thể lặp lại.",
    )
    parser.add_argument("--merge-asr", action="store_true")
    parser.add_argument("--apply", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        import_outputs(
            source=args.source.expanduser().resolve(),
            destination=args.destination.expanduser().resolve(strict=False),
            existing_roots=[path.expanduser().resolve() for path in args.existing_root],
            apply=args.apply,
            merge_asr=args.merge_asr,
        )
        return 0
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
