#!/usr/bin/env python3
"""Safely extend an existing merged dataset with additional team shards."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from pipeline.preprocess.merge import MergeError, merge


class ExtendError(RuntimeError):
    """A safe incremental-merge preflight or publication failure."""


def _same_filesystem(paths: list[Path], output_parent: Path) -> None:
    output_device = output_parent.stat().st_dev
    different = [path for path in paths if path.stat().st_dev != output_device]
    if different:
        raise ExtendError(
            "Hardlink mode requires target and every added shard on the same "
            "filesystem. Different filesystem: "
            + ", ".join(str(path) for path in different)
        )


def _hardlink_probe(source: Path, output_parent: Path) -> None:
    keyframes = source / "keyframes"
    if not keyframes.is_dir():
        return
    sample = next((path for path in keyframes.rglob("*") if path.is_file()), None)
    if sample is None:
        raise ExtendError(f"Visual source has no keyframe files: {source}")
    probe = output_parent / f".extend-hardlink-probe-{uuid.uuid4().hex}"
    try:
        os.link(sample, probe)
    except OSError as exc:
        raise ExtendError(
            f"Hardlink is unavailable from {source} to {output_parent}: {exc}"
        ) from exc
    finally:
        if probe.exists():
            probe.unlink()


def _publish(target: Path, candidate: Path, backup: Path) -> None:
    os.replace(target, backup)
    try:
        os.replace(candidate, target)
    except Exception:
        os.replace(backup, target)
        raise


def extend(args: argparse.Namespace) -> Path:
    target = Path(args.target).resolve()
    additions = [Path(value).resolve() for value in args.add]
    if not target.is_dir():
        raise ExtendError(f"Existing merged target does not exist: {target}")
    if not additions:
        raise ExtendError("At least one added shard is required")
    if any(not path.is_dir() for path in additions):
        missing = [str(path) for path in additions if not path.is_dir()]
        raise ExtendError("Added shard does not exist: " + ", ".join(missing))
    if target in additions:
        raise ExtendError("Do not repeat the target in --add")
    if len(set(additions)) != len(additions):
        raise ExtendError("The same added shard was supplied more than once")

    output_parent = target.parent
    _same_filesystem([target, *additions], output_parent)
    for source in [target, *additions]:
        _hardlink_probe(source, output_parent)

    candidate = output_parent / (
        f".{target.name}.extend-candidate-{uuid.uuid4().hex[:8]}"
    )
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    backup = output_parent / (
        f".{target.name}.backup-before-extend-{timestamp}-{uuid.uuid4().hex[:6]}"
    )
    if candidate.exists() or backup.exists():
        raise ExtendError("Unexpected candidate/backup path collision")

    merge_args = argparse.Namespace(
        sources=[str(target), *(str(path) for path in additions)],
        output=str(candidate),
        asset_mode="hardlink",
        include_videos=bool(args.include_videos),
        index_type=args.index_type,
        hnsw_m=args.hnsw_m,
        batch_size=args.batch_size,
        allow_incomplete_ocr=False,
        skip_clip_text_export=False,
    )

    try:
        merge(merge_args)
        report = {
            "format_version": 1,
            "extended_at_utc": datetime.now(timezone.utc).isoformat(),
            "target": str(target),
            "base_source": str(target),
            "added_sources": [str(path) for path in additions],
            "asset_mode": "hardlink",
            "backup": str(backup),
        }
        (candidate / "extend_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        _publish(target, candidate, backup)
    except Exception:
        if candidate.exists():
            shutil.rmtree(candidate, ignore_errors=True)
        raise

    if args.remove_backup:
        shutil.rmtree(backup)
        report["backup"] = None
        report["backup_removed"] = True
        (target / "extend_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print("\nEXTEND COMPLETE")
    print(f"  Active dataset: {target}")
    print("  Added shards:   " + ", ".join(str(path) for path in additions))
    if args.remove_backup:
        print("  Old dataset:    removed by explicit --remove-backup")
    else:
        print(f"  Rollback backup:{backup}")
        print("  Keep this backup until the server has been tested.")
    return backup


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        required=True,
        help="Existing merged_data directory to extend and keep as the active name",
    )
    parser.add_argument(
        "--add",
        nargs="+",
        required=True,
        metavar="SHARD",
        help="New shard directories only; do not repeat shards already in target",
    )
    parser.add_argument(
        "--index-type",
        choices=("auto", "hnsw", "flat"),
        default="auto",
    )
    parser.add_argument("--hnsw-m", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument(
        "--include-videos",
        action="store_true",
        help="Also include videos from added shards",
    )
    parser.add_argument(
        "--remove-backup",
        action="store_true",
        help="Remove the old dataset immediately after verified publication",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        extend(args)
    except (ExtendError, MergeError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
