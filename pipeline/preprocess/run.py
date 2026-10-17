#!/usr/bin/env python3
"""Run the preprocessing pipeline safely across local and removable shards."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from pathlib import Path

from dataset_layout import (
    DatasetConfigError,
    DatasetShard,
    StorageUnavailableError,
    load_dataset_config,
)
from video_manifest import filter_assigned_videos, load_assignments


PROJECT_ROOT = Path(__file__).resolve().parents[2]
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".flv", ".webm", ".m4v"}
DEFAULT_PROCESSED_CONFIG = PROJECT_ROOT / "federated_roots.json"


class PreprocessError(RuntimeError):
    pass


def video_files(shard: DatasetShard) -> list[Path]:
    if not shard.videos_dir.is_dir():
        return []
    return sorted(
        path for path in shard.videos_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )


def has_visual_artifacts(shard: DatasetShard) -> bool:
    return (
        shard.meta_db.is_file()
        or (shard.faiss_dir / "frames.faiss").is_file()
        or any(shard.keyframes_dir.rglob("*"))
    )


def python_runtime() -> Path:
    preferred = PROJECT_ROOT / ".venv_rebuild" / "bin" / "python"
    return preferred if preferred.is_file() else Path(sys.executable)


def selection_args(parser: argparse.ArgumentParser, *, optional: bool = False) -> None:
    parser.add_argument("--shard", action="append", dest="shards", help="Shard id; lặp lại để chọn nhiều shard")
    parser.add_argument("--all", action="store_true", help="Chọn mọi shard đang enabled")
    parser.set_defaults(selection_optional=optional)


def selected_shards(config, args) -> list[DatasetShard]:
    if getattr(args, "all", False) and getattr(args, "shards", None):
        raise PreprocessError("Không dùng đồng thời --all và --shard")
    if not getattr(args, "all", False) and not getattr(args, "shards", None):
        if not getattr(args, "selection_optional", False):
            raise PreprocessError("Hãy chọn --shard ID hoặc --all")
        return config.select()
    return config.select(None if args.all else args.shards)


def processed_video_owners(config_path: Path) -> dict[str, list[str]]:
    """Return case-insensitive video stem -> ready visual dataset ids."""
    if not config_path.is_file():
        return {}
    owners: dict[str, list[str]] = {}
    config = load_dataset_config(config_path)
    for dataset in config.select():
        try:
            dataset.preflight(write=False)
            if not dataset.meta_db.is_file():
                continue
            with sqlite3.connect(
                f"file:{dataset.meta_db.resolve()}?mode=ro", uri=True
            ) as db:
                has_locator = db.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='table' AND name='frame_locator'"
                ).fetchone()
                locator_complete = False
                if has_locator:
                    locator_count = int(
                        db.execute("SELECT COUNT(*) FROM frame_locator").fetchone()[0]
                    )
                    frame_count = int(db.execute("SELECT COUNT(*) FROM frames").fetchone()[0])
                    locator_complete = locator_count == frame_count and frame_count > 0
                if locator_complete:
                    rows = db.execute(
                        "SELECT DISTINCT video FROM frame_locator WHERE video != ''"
                    ).fetchall()
                if not locator_complete:
                    rows = db.execute(
                        "WITH normalized AS ("
                        " SELECT replace(file, char(92), '/') AS name FROM frames"
                        "), videos AS ("
                        " SELECT CASE WHEN instr(name, '/') > 0"
                        " THEN substr(name, 1, instr(name, '/') - 1) ELSE name END AS video"
                        " FROM normalized"
                        ") SELECT DISTINCT video FROM videos WHERE video != ''"
                    ).fetchall()
            for (video,) in rows:
                stem = Path(str(video)).stem.casefold()
                if stem:
                    owners.setdefault(stem, []).append(dataset.shard_id)
        except (DatasetConfigError, StorageUnavailableError, OSError, sqlite3.Error):
            # An offline removable root must not prevent processing on another
            # available storage. It will rejoin the catalog when plugged in.
            continue
    return owners


def processed_asr_owners(config_path: Path) -> dict[str, list[str]]:
    """Return case-insensitive video stem -> dataset ids with cached ASR."""
    if not config_path.is_file():
        return {}
    owners: dict[str, list[str]] = {}
    config = load_dataset_config(config_path)
    for dataset in config.select():
        try:
            dataset.preflight(write=False)
            if not dataset.asr_cache.is_file():
                continue
            payload = json.loads(dataset.asr_cache.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                continue
            for raw_key in payload:
                name = str(raw_key).replace("\\", "/").rsplit("::", 1)[-1]
                stem = Path(name).stem.casefold()
                if stem:
                    owners.setdefault(stem, []).append(dataset.shard_id)
        except (
            DatasetConfigError,
            StorageUnavailableError,
            OSError,
            json.JSONDecodeError,
        ):
            continue
    return owners


def unprocessed_videos(
    shard: DatasetShard,
    processed_config: Path,
    *,
    kind: str = "visual",
) -> tuple[list[Path], dict[str, list[str]]]:
    candidates = filter_assigned_videos(
        video_files(shard),
        shard.videos_dir,
        shard.video_manifest,
        shard.shard_id if shard.video_manifest else None,
    )
    owners = (
        processed_asr_owners(processed_config)
        if kind == "asr"
        else processed_video_owners(processed_config)
    )
    return [path for path in candidates if path.stem.casefold() not in owners], owners


def manifest_arguments(
    shard: DatasetShard,
    override: Path | None = None,
) -> list[str]:
    manifest = override or shard.video_manifest
    if not manifest:
        return []
    return ["--video-list", str(manifest), "--shard-id", shard.shard_id]


def command_for(
    stage: str,
    shard: DatasetShard,
    args,
    *,
    video_manifest: Path | None = None,
) -> tuple[list[str], dict[str, str]]:
    runtime = str(python_runtime())
    environment: dict[str, str] = {}
    if stage == "extract":
        command = [
            runtime,
            str(PROJECT_ROOT / "re_extract.py"),
            "--input-dir", str(shard.videos_dir),
            "--keyframes-dir", str(shard.keyframes_dir),
            "--map-dir", str(shard.map_dir),
            "--skip-existing",
            *manifest_arguments(shard, video_manifest),
        ]
    elif stage in {"index-fast", "index-full"}:
        command = [runtime, str(PROJECT_ROOT / "ocr_index.py")]
        environment.update({
            "AUTO_INDEX_ON_STARTUP": "0",
            "CLIP_BATCH_SIZE": os.getenv("CLIP_BATCH_SIZE", "8"),
            "FAISS_INDEX_TYPE": os.getenv("FAISS_INDEX_TYPE", "hnsw"),
            "FAISS_HNSW_M": os.getenv("FAISS_HNSW_M", "32"),
            "FAISS_EF_SEARCH": os.getenv("FAISS_EF_SEARCH", "96"),
            "FAISS_SAVE_EVERY": os.getenv("FAISS_SAVE_EVERY", "50000"),
            "YOLO_MODEL": os.getenv(
                "YOLO_MODEL",
                str(
                    PROJECT_ROOT
                    / "models"
                    / ("yolov8n.pt" if stage == "index-fast" else "yolov8m.pt")
                ),
            ),
            "YOLO_IMGSZ": os.getenv("YOLO_IMGSZ", "640"),
            "OCR_ENGINE": "none" if stage == "index-fast" else os.getenv("OCR_ENGINE", "easyocr"),
            "OCR_GPU": "0" if stage == "index-fast" else os.getenv("OCR_GPU", "1"),
            "OCR_LANGS": os.getenv("OCR_LANGS", "en,vi"),
            "OCR_BATCH_SIZE": os.getenv("OCR_BATCH_SIZE", "8"),
            "OCR_USE_ANGLE_CLS": "0",
        })
    elif stage == "ocr":
        command = ["bash", str(PROJECT_ROOT / "run_ocr_only_wsl.sh")]
    elif stage == "asr":
        command = [
            runtime, str(PROJECT_ROOT / "asr_index.py"),
            "--folder", str(shard.videos_dir),
            *manifest_arguments(shard, video_manifest),
        ]
    elif stage == "bundle":
        command = [
            runtime, str(PROJECT_ROOT / "prepare_serve_bundle.py"),
            "--data-root", str(shard.data_root),
        ]
        if getattr(args, "allow_incomplete_ocr", False):
            command.append("--allow-pending-ocr")
    elif stage == "export":
        output = Path(args.output.format(shard=shard.shard_id)).expanduser()
        command = [
            runtime, str(PROJECT_ROOT / "export_shard.py"),
            "--project", str(shard.data_root),
            "--output", str(output),
            "--name", shard.shard_id,
            "--mode", args.mode,
            "--asset-mode", args.asset_mode,
        ]
    else:
        raise PreprocessError(f"Stage chưa được hỗ trợ: {stage}")
    return command, environment


def run_external_stage(stage: str, shard: DatasetShard, args) -> None:
    videos = video_files(shard)
    if stage in {"extract", "asr"} and not videos:
        print(f"\n=== {shard.shard_id}: {stage} bỏ qua — không có video ===")
        return
    if stage in {"index-fast", "index-full", "ocr", "bundle"} and not (
        has_visual_artifacts(shard)
    ):
        print(f"\n=== {shard.shard_id}: {stage} bỏ qua — dataset rỗng ===")
        return
    selected_videos = videos
    if stage in {"extract", "asr"}:
        selected_videos, _ = unprocessed_videos(
            shard,
            Path(args.processed_config).expanduser().resolve(strict=False),
            kind="asr" if stage == "asr" else "visual",
        )
        skipped = len(videos) - len(selected_videos)
        if skipped:
            print(
                f"\n=== {shard.shard_id}: {stage} inventory bỏ qua "
                f"{skipped} video đã xử lý ==="
            )
        if not selected_videos:
            print(f"=== {shard.shard_id}: {stage} bỏ qua — không có video mới ===")
            return
    snapshot = shard.preflight(
        write=stage not in {"export"},
        require_videos=stage in {"extract", "asr"},
    )
    temporary_manifest: Path | None = None
    if stage in {"extract", "asr"}:
        handle = tempfile.NamedTemporaryFile(
            mode="w",
            prefix=f"aic-{shard.shard_id}-{stage}-unprocessed-",
            suffix=".json",
            encoding="utf-8",
            delete=False,
        )
        temporary_manifest = Path(handle.name)
        with handle:
            json.dump(
                {
                    "format_version": 1,
                    "assignments": {
                        path.relative_to(shard.videos_dir).as_posix(): shard.shard_id
                        for path in selected_videos
                    },
                },
                handle,
                ensure_ascii=False,
                indent=2,
            )
            handle.write("\n")
    command, overrides = command_for(
        stage, shard, args, video_manifest=temporary_manifest
    )
    environment = os.environ.copy()
    environment.update(shard.environment())
    environment.update(overrides)
    print(f"\n=== {shard.shard_id}: {stage} ===", flush=True)
    print("Data root:", shard.data_root, flush=True)
    try:
        result = subprocess.run(command, cwd=PROJECT_ROOT, env=environment)
        shard.verify_unchanged(snapshot)
    finally:
        if temporary_manifest:
            temporary_manifest.unlink(missing_ok=True)
    if result.returncode:
        raise PreprocessError(
            f"Shard {shard.shard_id}, stage {stage} thất bại (exit {result.returncode})"
        )


def verify_shard(shard: DatasetShard) -> dict[str, int]:
    shard.preflight(write=False)
    required = [
        shard.meta_db,
        shard.faiss_dir / "frames.faiss",
        shard.faiss_dir / "frames_meta.json",
        shard.keyframes_dir,
        shard.map_dir,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise PreprocessError(
            f"Shard {shard.shard_id} thiếu artifact: " + ", ".join(missing)
        )
    try:
        with sqlite3.connect(f"file:{shard.meta_db.resolve()}?mode=ro", uri=True) as db:
            frame_count = int(db.execute("SELECT COUNT(*) FROM frames").fetchone()[0])
        sidecar = json.loads(
            (shard.faiss_dir / "frames_meta.json").read_text(encoding="utf-8")
        )
        vector_count = len(sidecar.get("fid_to_meta", sidecar.get("items", {})))
    except (OSError, sqlite3.Error, json.JSONDecodeError, AttributeError) as exc:
        raise PreprocessError(f"Shard {shard.shard_id} không hợp lệ: {exc}") from exc
    image_count = sum(
        1 for path in shard.keyframes_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    )
    if not (frame_count == vector_count == image_count):
        raise PreprocessError(
            f"Shard {shard.shard_id} lệch số lượng: SQLite={frame_count}, "
            f"FAISS metadata={vector_count}, images={image_count}"
        )
    return {"frames": frame_count, "vectors": vector_count, "images": image_count}


def status(shards: list[DatasetShard]) -> int:
    failed = False
    for shard in shards:
        print(f"\n[{shard.shard_id}] {'enabled' if shard.enabled else 'disabled'}")
        print(f"  data_root:  {shard.data_root}")
        print(f"  videos_dir: {shard.videos_dir}")
        try:
            shard.preflight(write=False)
            usage = shutil.disk_usage(shard.data_root)
            videos = len(video_files(shard))
            print("  status:     ONLINE")
            print(f"  free:       {usage.free / 1024**3:.2f} GiB")
            print(f"  videos:     {videos}")
            if shard.video_manifest and shard.video_manifest.is_file():
                assignments = load_assignments(shard.video_manifest)
                assigned = sum(
                    value == shard.shard_id for value in assignments.values()
                )
                print(f"  assigned:   {assigned}")
            if not shard.videos_dir.is_dir():
                print("  video src:  ABSENT (index/verify vẫn dùng được; extract/ASR không dùng được)")
        except (DatasetConfigError, StorageUnavailableError) as exc:
            failed = True
            print(f"  status:     OFFLINE — {exc}")
    return 1 if failed else 0


def plan_split(source: Path, output: Path, shards: list[DatasetShard], force: bool) -> None:
    source = source.expanduser().resolve()
    if not source.is_dir():
        raise PreprocessError(f"Source video directory không tồn tại: {source}")
    if output.exists() and not force:
        raise PreprocessError(f"Manifest đã tồn tại: {output}; dùng --force để thay thế")
    videos = sorted(
        (path for path in source.rglob("*") if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS),
        key=lambda path: (-path.stat().st_size, path.as_posix()),
    )
    if not videos:
        raise PreprocessError(f"Không tìm thấy video dưới {source}")
    totals = {shard.shard_id: 0 for shard in shards}
    assignments: dict[str, str] = {}
    for video in videos:
        shard_id = min(totals, key=lambda key: (totals[key], key))
        relative = video.relative_to(source).as_posix()
        assignments[relative] = shard_id
        totals[shard_id] += video.stat().st_size
    payload = {
        "format_version": 1,
        "source_dir": str(source),
        "strategy": "greedy-by-file-size",
        "assignments": dict(sorted(assignments.items())),
        "shard_bytes": totals,
    }
    output = output.expanduser().resolve(strict=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(f"Đã gán {len(videos)} video: {output}")
    for shard_id, total in totals.items():
        count = sum(1 for value in assignments.values() if value == shard_id)
        print(f"  {shard_id}: {count} video, {total / 1024**3:.2f} GiB")


def merge_shards(shards: list[DatasetShard], args) -> None:
    snapshots = [shard.preflight(write=False) for shard in shards]
    output = args.output.expanduser().resolve(strict=False)
    manifest_path: Path | None = None
    if args.asset_mode == "symlink":
        payload = {
            "format_version": 1,
            "asset_mode": "symlink",
            "sources": [
                {
                    "id": shard.shard_id,
                    "data_root": str(shard.data_root),
                    "videos_dir": str(shard.videos_dir),
                    "mount_root": str(shard.mount_root) if shard.mount_root else None,
                    "volume_marker": shard.volume_marker,
                    "require_mount": shard.require_mount,
                }
                for shard in shards
            ],
        }
        manifest_path = output.parent / f".{output.name}.external-storage-input.json"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    command = [
        str(python_runtime()),
        str(PROJECT_ROOT / "merge_processed_data.py"),
        "--sources", *[str(shard.data_root) for shard in shards],
        "--output", str(output),
        "--asset-mode", args.asset_mode,
    ]
    if manifest_path:
        command.extend(["--external-storage-manifest", str(manifest_path)])
    if args.include_videos:
        command.append("--include-videos")
    if args.allow_incomplete_ocr:
        command.append("--allow-incomplete-ocr")
    if args.skip_clip_text_export:
        command.append("--skip-clip-text-export")
    try:
        result = subprocess.run(command, cwd=PROJECT_ROOT, env=os.environ.copy())
    finally:
        if manifest_path:
            manifest_path.unlink(missing_ok=True)
    for shard, snapshot in zip(shards, snapshots):
        shard.verify_unchanged(snapshot)
    if result.returncode:
        raise PreprocessError(f"Merge thất bại (exit {result.returncode})")
    if args.asset_mode == "symlink":
        print(f"Đã ghi external storage manifest: {output / 'external_storage.json'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "dataset_roots.json"))
    parser.add_argument(
        "--processed-config",
        default=os.getenv("FEDERATED_DATASET_CONFIG", str(DEFAULT_PROCESSED_CONFIG)),
        help="Các root read-only dùng để bỏ qua video đã có visual artifact",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    status_parser = commands.add_parser("status", help="Kiểm tra mount, marker và dung lượng")
    selection_args(status_parser, optional=True)

    init_parser = commands.add_parser("init", help="Khởi tạo cấu trúc thư mục shard")
    selection_args(init_parser)
    init_parser.add_argument("--write-marker", action="store_true", help="Tạo marker mới; không ghi đè marker khác")

    for name in ("extract", "index-fast", "index-full", "ocr", "asr", "verify"):
        child = commands.add_parser(name)
        selection_args(child)

    bundle = commands.add_parser("bundle")
    selection_args(bundle)
    bundle.add_argument("--allow-incomplete-ocr", action="store_true")

    export = commands.add_parser("export")
    selection_args(export)
    export.add_argument("--output", required=True, help="Đường dẫn output; dùng {shard} cho shard id")
    export.add_argument("--mode", choices=("auto", "visual", "asr"), default="auto")
    export.add_argument("--asset-mode", choices=("copy", "hardlink"), default="copy")

    merge = commands.add_parser("merge", help="Gộp SQLite/FAISS; có thể giữ asset trên local/USB")
    selection_args(merge)
    merge.add_argument("--output", type=Path, required=True)
    merge.add_argument("--asset-mode", choices=("copy", "hardlink", "symlink"), default="symlink")
    merge.add_argument("--include-videos", action="store_true")
    merge.add_argument("--allow-incomplete-ocr", action="store_true")
    merge.add_argument("--skip-clip-text-export", action="store_true")

    run = commands.add_parser("run", help="Chạy extract + index, và các bước tùy chọn")
    selection_args(run)
    run.add_argument("--full-index", action="store_true", help="Bật OCR trong index thay vì fast mode")
    run.add_argument("--ocr", action="store_true", help="Chạy thêm OCR-only sau index")
    run.add_argument("--asr", action="store_true")
    run.add_argument("--bundle", action="store_true")
    run.add_argument("--allow-incomplete-ocr", action="store_true")

    split = commands.add_parser("plan-split", help="Lập manifest chia video cân bằng theo dung lượng")
    selection_args(split, optional=True)
    split.add_argument("--source-dir", type=Path, required=True)
    split.add_argument("--output", type=Path, required=True)
    split.add_argument("--force", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        config = load_dataset_config(args.config)
        shards = selected_shards(config, args)
        if args.command == "status":
            return status(shards)
        if args.command == "init":
            for shard in shards:
                shard.initialize(write_marker=args.write_marker)
                print(f"Khởi tạo xong {shard.shard_id}: {shard.data_root}")
            return 0
        if args.command == "plan-split":
            plan_split(args.source_dir, args.output, shards, args.force)
            return 0
        if args.command == "merge":
            merge_shards(shards, args)
            return 0
        if args.command == "verify":
            for shard in shards:
                result = verify_shard(shard)
                print(f"{shard.shard_id}: OK — {result['frames']} frames")
            return 0
        stages = (
            ["extract", "index-full" if args.full_index else "index-fast"]
            + (["ocr"] if args.ocr else [])
            + (["asr"] if args.asr else [])
            + (["bundle"] if args.bundle else [])
        ) if args.command == "run" else [args.command]
        for shard in shards:
            for stage in stages:
                run_external_stage(stage, shard, args)
        return 0
    except (
        DatasetConfigError,
        StorageUnavailableError,
        PreprocessError,
        ValueError,
        RuntimeError,
    ) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nĐã dừng theo yêu cầu; có thể chạy lại để resume.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
