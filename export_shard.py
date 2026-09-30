"""Export processed project artifacts into one portable team shard."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path


VISUAL_ARTIFACTS = (
    Path("meta.db"),
    Path("faiss_index/frames.faiss"),
    Path("faiss_index/frames_meta.json"),
    Path("keyframes"),
    Path("map-keyframes"),
)


class ExportError(RuntimeError):
    pass


def _copy_file(source: Path, destination: Path, mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if mode == "hardlink":
        try:
            os.link(source, destination)
            return
        except OSError:
            pass
    shutil.copy2(source, destination)


def _copy_tree(source: Path, destination: Path, mode: str) -> None:
    for path in sorted(item for item in source.rglob("*") if item.is_file()):
        _copy_file(path, destination / path.relative_to(source), mode)

def _copy_database(source: Path, destination: Path) -> None:
    """Create a consistent SQLite snapshot, including committed WAL data."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    source_db = destination_db = None
    try:
        source_db = sqlite3.connect(
            f"file:{source.resolve()}?mode=ro", uri=True
        )
        destination_db = sqlite3.connect(destination)
        source_db.backup(destination_db)
        destination_db.commit()
    except sqlite3.Error as exc:
        raise ExportError(f"Cannot snapshot SQLite database {source}: {exc}") from exc
    finally:
        if destination_db is not None:
            destination_db.close()
        if source_db is not None:
            source_db.close()
    shutil.copystat(source, destination)



def _validate_asr(path: Path) -> int:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ExportError(f"Cannot read ASR cache {path}: {exc}") from exc
    if not isinstance(data, dict) or not data:
        raise ExportError(f"ASR cache must be a non-empty JSON object: {path}")
    invalid = [key for key, value in data.items() if not isinstance(value, list)]
    if invalid:
        raise ExportError(f"ASR cache contains invalid entries: {invalid[:3]}")
    return len(data)


def _frame_count(db_path: Path) -> int:
    try:
        with sqlite3.connect(
            f"file:{db_path.resolve()}?mode=ro", uri=True
        ) as connection:
            row = connection.execute("SELECT COUNT(*) FROM frames").fetchone()
            return int(row[0])
    except sqlite3.Error as exc:
        raise ExportError(f"Cannot count frames in {db_path}: {exc}") from exc


def _visual_state(project: Path) -> tuple[bool, list[str]]:
    states = [(relative, (project / relative).exists()) for relative in VISUAL_ARTIFACTS]
    missing = [str(relative) for relative, exists in states if not exists]
    present_count = sum(exists for _, exists in states)
    if present_count and missing:
        raise ExportError(
            "Visual artifact group is incomplete; missing: " + ", ".join(missing)
        )
    return present_count == len(states), missing


def export_shard(args: argparse.Namespace) -> Path:
    project = Path(args.project).resolve()
    output = Path(args.output).resolve()
    if not project.is_dir():
        raise ExportError(f"Project directory does not exist: {project}")
    if output.exists():
        if not output.is_dir() or any(output.iterdir()):
            raise ExportError(f"Output must be absent or empty: {output}")
        output.rmdir()

    asr_path = project / "asr_cache.json"
    visual_complete = False
    if args.mode in {"auto", "visual"}:
        visual_complete, _ = _visual_state(project)

    if args.mode == "visual" and not visual_complete:
        raise ExportError("Visual mode requires the complete visual artifact group")
    if args.mode == "asr":
        shard_type = "asr-only"
    elif visual_complete:
        shard_type = "visual+asr" if asr_path.is_file() and not args.skip_asr else "visual"
    elif asr_path.is_file():
        shard_type = "asr-only"
    else:
        raise ExportError("No complete visual group or ASR cache is available to export")

    include_visual = shard_type.startswith("visual")
    include_asr = shard_type in {"asr-only", "visual+asr"}
    asr_count = _validate_asr(asr_path) if include_asr else 0
    frame_count = _frame_count(project / "meta.db") if include_visual else 0

    staging = output.with_name(f".{output.name}.export-{uuid.uuid4().hex[:8]}")
    staging.mkdir(parents=True)
    try:
        if include_visual:
            _copy_database(project / "meta.db", staging / "meta.db")
            _copy_tree(project / "faiss_index", staging / "faiss_index", args.asset_mode)
            _copy_tree(project / "keyframes", staging / "keyframes", args.asset_mode)
            _copy_tree(
                project / "map-keyframes",
                staging / "map-keyframes",
                args.asset_mode,
            )
        if include_asr:
            _copy_file(asr_path, staging / "asr_cache.json", args.asset_mode)

        artifacts = sorted(
            str(path.relative_to(staging))
            for path in staging.rglob("*")
            if path.is_file()
        )
        manifest = {
            "format_version": 1,
            "member": args.name,
            "shard_type": shard_type,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "frame_count": frame_count,
            "asr_video_count": asr_count,
            "artifacts": artifacts,
        }
        (staging / "shard_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        os.replace(staging, output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    print("SHARD EXPORT COMPLETE")
    print(f"  Member: {args.name}")
    print(f"  Type:   {shard_type}")
    print(f"  Output: {output}")
    print(f"  Frames: {frame_count}")
    print(f"  ASR:    {asr_count} video(s)")
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--mode", choices=("auto", "visual", "asr"), default="auto")
    parser.add_argument("--asset-mode", choices=("copy", "hardlink"), default="copy")
    parser.add_argument("--skip-asr", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        export_shard(args)
    except ExportError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

