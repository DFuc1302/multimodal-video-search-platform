#!/usr/bin/env python3
"""Build one merged old+new search index while keeping all media at its source.

Only user-invoked execution modifies data. audit/--dry-run never delete images.
Dedup keeps the first frame of each consecutive similar run of length >= 2.
ASR is produced externally; this pipeline never transcribes videos. Existing old vectors/OCR/ASR are reused, not recomputed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import sqlite3
import sys
import uuid
from pathlib import Path
from tqdm import tqdm

from pipeline.run import save_json, run
from external_assets import ExternalAssetResolver, video_id_from_path
from server_data_health import record_issue

# Pipeline modules remain at repository root while the package migration is in
# progress. Resolve the repository root explicitly from the new location.
CODE = Path(__file__).resolve().parents[1]
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".m4v", ".ts", ".m2ts"}
STAGES = ("audit", "dedup", "index", "bundle", "merge", "verify", "redis", "activate")
MAP_FIELDS = ["n", "pts_time", "fps", "frame_idx"]


def progress_items(items, description, unit="video", total=None, stages=False):
    """Advance only after the item finishes; close the display on failure."""
    options = {"desc": description, "unit": unit, "dynamic_ncols": True,
               "total": len(items) if total is None and hasattr(items, "__len__") else total}
    if stages:
        # Stages differ greatly in duration: do not show a misleading global ETA.
        options["bar_format"] = "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}] {postfix}"
    with tqdm(**options) as bar:
        for item in items:
            if stages:
                bar.set_postfix_str(str(item).upper())
            yield item
            bar.update(1)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def connect_ro(path):
    connection = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    return connection


def file_digest(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def preflight(stage):
    modules = {"imagehash", "PIL", "numpy", "scipy", "torch", "tqdm"}
    if stage == "all":
        modules.update({"faiss", "easyocr", "transformers", "ultralytics", "redis", "cv2"})
    missing = sorted(name for name in modules if importlib.util.find_spec(name) is None)
    if missing:
        raise RuntimeError(f"Missing Python modules in {sys.executable}: {', '.join(missing)}. See docs/archive/aic-2026/README_NEW_DATASET.md")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot access CUDA in this environment")


def image_files(folder):
    files = sorted(folder.glob("*.jpg"), key=lambda p: int(p.stem) if p.stem.isdigit() else -1)
    if not files or any(not p.stem.isdigit() or p.is_symlink() for p in files):
        raise RuntimeError(f"Empty/non-numeric/symlink image folder: {folder}")
    if len({int(p.stem) for p in files}) != len(files):
        raise RuntimeError(f"Duplicate numeric frame IDs: {folder}")
    return files


def image_signature(files):
    digest = hashlib.sha256()
    for image in files:
        stat = image.stat()
        digest.update(f"{image.name}:{stat.st_size}:{stat.st_mtime_ns}\n".encode())
    return digest.hexdigest()


def map_rows(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != MAP_FIELDS:
            raise RuntimeError(f"Wrong map columns: {path}")
        rows = list(reader)
    previous = (-1, -1, -1.0)
    for row in rows:
        n, idx, timestamp = int(row["n"]), int(row["frame_idx"]), float(row["pts_time"])
        if n <= previous[0] or idx <= previous[1] or timestamp < previous[2] or not math.isfinite(timestamp):
            raise RuntimeError(f"Unordered/invalid frame map: {path}: {row}")
        previous = (n, idx, timestamp)
    if not rows:
        raise RuntimeError(f"Empty map: {path}")
    return rows


def filter_map(path, files):
    rows = map_rows(path)
    ids = {int(p.stem) for p in files}
    retained = [row for row in rows if int(row["n"]) in ids]
    if len(retained) != len(ids):
        raise RuntimeError(f"Some images have no map entry: {path}")
    temp = path.with_suffix(".csv.first-only.tmp")
    with temp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MAP_FIELDS)
        writer.writeheader()
        writer.writerows(retained)
    temp.replace(path)


def first_only_deletions(hashes, threshold):
    # Compare adjacent ORIGINAL frames, not every frame against the retained first.
    return [i for i in range(1, len(hashes))
            if hashes[i - 1] is not None and hashes[i] is not None
            and hashes[i - 1] - hashes[i] <= threshold]


def inventory(extracted, videos_dir):
    videos = {}
    for video in sorted(videos_dir.rglob("*")):
        if video.is_file() and video.suffix.lower() in VIDEO_EXTS:
            key = video.stem.casefold()
            if key in videos:
                raise RuntimeError(f"Duplicate source video ID: {video.stem}")
            videos[key] = video
    folders = {p.name.casefold(): p for p in (extracted / "keyframes").iterdir()
               if p.is_dir() and not p.name.startswith(".")}
    if not videos or set(videos) != set(folders):
        raise RuntimeError(f"Video/frame folder mismatch: missing={sorted(set(videos)-set(folders))}; "
                           f"unknown={sorted(set(folders)-set(videos))}")
    result = []
    for key, video in progress_items(sorted(videos.items()), "Inventory / maps"):
        folder = folders[key]
        mapping = extracted / "map-keyframes" / (folder.name + ".csv")
        files, rows = image_files(folder), map_rows(mapping)
        row_ids = {int(row["n"]) for row in rows}
        if any(int(p.stem) not in row_ids for p in files):
            raise RuntimeError(f"Images without source-frame mapping: {folder}")
        result.append({"video": video, "folder": folder, "map": mapping, "files": files,
                       "signature": image_signature(files), "map_rows": len(rows),
                       "asr": bool(re.fullmatch(r"S01[-_]V\d+", video.stem, re.I))})
    return result


def old_video_ids(root):
    with connect_ro(root / "meta.db") as connection:
        return {video_id_from_path(row[0]).casefold() for row in connection.execute("SELECT file FROM frames")}


def verify_asr(path, records):
    cache = read_json(path)
    expected = {f"video-name-v1::{r['video'].name}" for r in records if r["asr"]}
    if set(cache) != expected:
        raise RuntimeError(f"ASR must cover S01 only; missing={expected-set(cache)}, extra={set(cache)-expected}")
    for key, segments in cache.items():
        if not isinstance(segments, list):
            raise RuntimeError(f"Invalid ASR cache for {key}")
        for segment in segments:
            if (segment.get("cache_version") != 2 or not isinstance(segment.get("text"), str)
                    or not isinstance(segment.get("search_text"), str)
                    or abs(float(segment["end"])-float(segment["start"])-2) > 0.02):
                raise RuntimeError(f"Incomplete ASR windows for {key}")
    return len(expected)


def verify_bundle(root, expected_count=None, check_assets=True):
    from prepare_serve_bundle import _validate_faiss_mapping
    manifest = read_json(root / "serve_manifest.json")
    resolver = ExternalAssetResolver.load(root)
    with connect_ro(root / "meta.db") as connection:
        count = connection.execute("SELECT COUNT(*) FROM frames").fetchone()[0]
        faiss_info = _validate_faiss_mapping(connection, root, required=True)
        bad_ocr = connection.execute("SELECT COUNT(*) FROM frames WHERE ocr_status IS NULL "
                                     "OR ocr_status NOT IN ('done','no_text','skipped','reused')").fetchone()[0]
        if bad_ocr:
            record_issue('pipeline.ocr', 'OCR is incomplete; completed rows remain searchable', path=root, count=bad_ocr)
        if expected_count is not None and count != expected_count:
            record_issue('pipeline.metadata', 'Some expected frames have not been indexed', path=root,
                         count=abs(expected_count - count))
        if manifest["frame_count"] != count:
            raise RuntimeError(f"Incomplete bundle {root}: rows={count}, expected={expected_count}, bad_ocr={bad_ocr}")
        if check_assets:
            cached_maps = {}
            for row in progress_items(connection.execute("SELECT file FROM frames"),
                                      f"Verify {root.name}", unit="frame", total=count):
                filename = row[0]
                image = resolver.resolve_image(filename) if resolver else root / "keyframes" / filename
                video = video_id_from_path(filename)
                if image is None or not image.is_file():
                    record_issue('pipeline.keyframe', 'Unresolved keyframe', path=filename)
                if video not in cached_maps:
                    mapping = resolver.resolve_map(video) if resolver else root / "map-keyframes" / f"{video}.csv"
                    if mapping is None or not mapping.is_file():
                        record_issue('pipeline.map', 'Unresolved map', path=video)
                        cached_maps[video] = set()
                        continue
                    # Old competition maps may repeat frame_idx; only their n lookup is required.
                    with mapping.open(encoding="utf-8-sig", newline="") as handle:
                        cached_maps[video] = {int(r["n"]) for r in csv.DictReader(handle)}
                if int(Path(filename).stem) not in cached_maps[video]:
                    record_issue('pipeline.map', 'No original frame index', path=filename)
    return {"frames": count, "vectors": faiss_info["count"], "assets_checked": check_assets}


def dedup_records(records, work, threshold, workers, batch_size, dry_run=False):
    from dedup_keyframes import GPUHasher
    plans, backups = work / "dedup_plans", work / "maps_before_dedup"
    plans.mkdir(exist_ok=True)
    backups.mkdir(exist_ok=True)
    gpu = GPUHasher(workers, batch_size)
    total = 0
    try:
        for record in progress_items(records, "Dedup videos"):
            path = plans / (record["folder"].name + ".json")
            plan = read_json(path) if path.exists() else None
            if plan is not None and (plan["threshold"] != threshold or plan["policy"] != "first-only-v1"):
                raise RuntimeError(f"Dedup recipe changed: {path}; use a fresh work directory")
            if plan and plan.get("complete"):
                if record["signature"] != plan["result_signature"]:
                    raise RuntimeError(f"Images changed since dedup: {record['folder']}")
                if plan.get("map_signature") != file_digest(record["map"]):
                    raise RuntimeError(f"Map changed since dedup: {record['map']}")
                tqdm.write(f"Dedup already complete: {record['video'].stem}")
                continue
            if plan is None:
                with tqdm(total=len(record["files"]), desc=f"pHash GPU {record['video'].stem}",
                          unit="frame", dynamic_ncols=True, leave=False) as frames_bar:
                    hashes, errors = gpu.hashes(record["files"], "phash", progress=frames_bar.update)
                if errors:
                    raise RuntimeError(f"Unreadable images; no deletion for this video: {errors[:5]}")
                plan = {"policy": "first-only-v1", "threshold": threshold, "complete": False,
                        "input": [p.name for p in record["files"]],
                        "remove": [record["files"][i].name for i in first_only_deletions(hashes, threshold)]}
                if not dry_run:
                    shutil.copy2(record["map"], backups / record["map"].name)
                    save_json(path, plan)  # Persist the exact deletion list BEFORE deleting anything.
            present = {p.name for p in record["files"]}
            removable = set(plan["remove"])
            if present-set(plan["input"]) or (set(plan["input"])-removable)-present:
                raise RuntimeError(f"Images changed during interrupted dedup: {record['folder']}")
            total += len(removable & present)
            tqdm.write(f"{'Would delete' if dry_run else 'Deleting'} {len(removable & present)}: {record['video'].stem}")
            if dry_run:
                continue
            if (work / "new_index" / "meta.db").exists():
                raise RuntimeError("New index already exists; do not change its image inventory. Use a fresh work directory")
            for name in progress_items(plan["remove"], f"Delete {record['video'].stem}", unit="frame"):
                if Path(name).name != name:
                    raise RuntimeError("Invalid filename in saved deletion plan")
                (record["folder"] / name).unlink(missing_ok=True)
            remaining = image_files(record["folder"])
            filter_map(record["map"], remaining)
            plan.update(complete=True, result_signature=image_signature(remaining), remaining=len(remaining),
                        map_signature=file_digest(record["map"]))
            save_json(path, plan)
    finally:
        gpu.close()
    print(f"Dedup {'planned' if dry_run else 'finished'}: {total} deletions", flush=True)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("all", *STAGES, "rollback"), default="audit")
    parser.add_argument("--dry-run", action="store_true", help="only supported for audit/dedup")
    parser.add_argument("--extracted-dir", type=Path, default=Path("/mnt/d/AIC_26/data/extracted"))
    parser.add_argument("--videos-dir", type=Path, default=Path("/mnt/d/AIC_26/data/videos_2"))
    parser.add_argument("--old-index", type=Path, default=Path("/home/dtp/project/aic_local_prep/search_index_ocr"))
    parser.add_argument("--old-videos-dir", type=Path, default=Path("/mnt/d/AIC/data/videos"))
    parser.add_argument("--work-dir", type=Path, default=Path("/mnt/d/AIC_26/data/new_dataset_work"))
    parser.add_argument("--merged-index", type=Path, default=Path("/mnt/d/AIC_26/data/merged_search_index"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--threshold", type=int, default=4)
    parser.add_argument("--clip-batch-size", type=int, default=8)
    parser.add_argument("--ocr-batch-size", type=int, default=12)
    parser.add_argument("--redis-port", type=int, default=16380)
    parser.add_argument("--redis-root", type=Path, default=Path("/home/dtp/project/aic_local_prep"))
    parser.add_argument("--summary-index", type=Path, default=Path("/home/dtp/project/aic_local_prep/summary_index"))
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if os.name != "posix":
        parser.error("Run in WSL/Linux")
    if args.dry_run and args.stage not in {"audit", "dedup"}:
        parser.error("--dry-run only applies to audit/dedup")
    if min(args.workers, args.batch_size, args.clip_batch_size, args.ocr_batch_size) < 1 or not 0 <= args.threshold <= 64:
        parser.error("Invalid threshold, worker or batch size")
    if not 1024 <= args.redis_port <= 65535:
        parser.error("Invalid Redis port")
    for name in ("extracted_dir", "videos_dir", "old_index", "work_dir", "merged_index", "redis_root", "summary_index"):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    roots = [args.extracted_dir, args.videos_dir, args.old_index, args.work_dir, args.merged_index]
    for i, first in enumerate(roots):
        for second in roots[i+1:]:
            if first.is_relative_to(second) or second.is_relative_to(first):
                parser.error(f"Input/index/work paths must be separate: {first}, {second}")
    if str(args.extracted_dir).startswith("/mnt/d/") and not os.path.ismount("/mnt/d"):
        parser.error("Drive D is not mounted at /mnt/d")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (args.work_dir / ".pipeline.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        execute(args)
    return 0


def execute(args):
    work, merged, new = args.work_dir, args.merged_index, args.work_dir / "new_index"
    active = CODE / ".active_dataset.json"
    if args.stage == "rollback":
        saved = work / "previous_ui.json"
        if not saved.exists():
            raise RuntimeError("No saved UI configuration to restore")
        previous = read_json(saved)
        if previous is None:
            active.unlink(missing_ok=True)
        else:
            save_json(active, previous)
        print("Previous UI configuration restored. Restart the UI.")
        return
    if args.stage in {"all", "dedup"}:
        preflight(args.stage)
    for directory in (work / "logs", new / "keyframes", new / "map-keyframes", work / "models"):
        directory.mkdir(parents=True, exist_ok=True)
    old_manifest = read_json(args.old_index / "serve_manifest.json")
    records = inventory(args.extracted_dir, args.videos_dir)
    old_ids = old_video_ids(args.old_index)
    new_ids = {r["folder"].name.casefold() for r in records}
    if old_ids & new_ids:
        raise RuntimeError(f"Old/new video IDs overlap: {sorted(old_ids & new_ids)}")
    state_file = work / "state.json"
    state = read_json(state_file) if state_file.exists() else {"run_id": uuid.uuid4().hex[:12]}
    recipe = {"extracted": str(args.extracted_dir), "videos": str(args.videos_dir),
              "old_index": str(args.old_index), "merged": str(merged), "threshold": args.threshold,
              "policy": "first-only-v1", "clip_model": old_manifest["clip_model"],
              # Preserve historical provenance when resuming an existing work-dir;
              # this field no longer selects or runs an ASR model.
              "whisper_model": state.get('recipe', {}).get('whisper_model', 'external'),
              "video_ids": sorted(new_ids),
              "old_generation": file_digest(args.old_index / "serve_manifest.json")}
    if state.get("recipe", recipe) != recipe:
        raise RuntimeError("Inputs/models/policy changed. Use a fresh work-dir and merged-index")
    state["recipe"] = recipe
    save_json(state_file, state)
    prefix = f"aic:merged:{state['run_id']}:"
    env = os.environ.copy()
    env.update({"DATA_ROOT": str(new), "KEYFRAMES_DIR": str(args.extracted_dir / "keyframes"),
                "MAPFILES_DIR": str(args.extracted_dir / "map-keyframes"), "VIDEOS_DIR": str(args.videos_dir),
                "META_DB": str(new / "meta.db"), "FAISS_DIR": str(new / "faiss_index"),
                "ASR_CACHE_FILE": str(new / "asr_cache.json"), "ASR_FRAMES_DIR": str(new / "asr_frames"),
                "CLIP_MODEL": old_manifest["clip_model"], "CLIP_BATCH_SIZE": str(args.clip_batch_size),
                "FAISS_INDEX_TYPE": "hnsw", "FAISS_HNSW_M": "32", "FAISS_SAVE_EVERY": "5000",
                "FAISS_SQLITE_MAPPING": "0", "VERIFY_FAISS": "1", "REQUIRE_SERVE_BUNDLE": "0",
                "YOLO_MODEL": str(work / "models" / "yolov8m.pt"), "YOLO_IMGSZ": "640",
                "OCR_ENGINE": "easyocr", "OCR_LANGS": "en,vi", "OCR_GPU": "1", "OCR_ALL_FRAMES": "1",
                "OCR_BATCH_SIZE": str(args.ocr_batch_size), "OCR_ENHANCE": os.getenv("OCR_ENHANCE", "0"), "OCR_USE_ANGLE_CLS": "0",
                "OCR_WHITE_ON_BLACK_ONLY": os.getenv("OCR_WHITE_ON_BLACK_ONLY", "1"),
                "OCR_MAX_SIDE": os.getenv("OCR_MAX_SIDE", "960"), "OCR_TARGET_HOURS": "30",
                "ALLOW_HEAVY_RUNTIME_JOBS": "0", "ASR_EXTERNAL": "1",
                "BACKEND_MODE": "sqlite", "STORAGE_BACKEND": "sqlite", "SERVE_READ_ONLY": "0",
                "AUTO_INDEX_ON_STARTUP": "0", "FEDERATED_DATASET_CONFIG": "", "CLIP_TEXT_ONLY": "0",
                "CLIP_TEXT_MODEL_PATH": "", "PYTHONUNBUFFERED": "1",
                "HF_HOME": str(work / "model_cache" / "huggingface"), "TORCH_HOME": str(work / "model_cache" / "torch"),
                "XDG_CACHE_HOME": str(work / "model_cache"), "EASYOCR_MODULE_PATH": str(work / "model_cache" / "easyocr")})
    redis_env = {"REDIS_PORT": str(args.redis_port), "REDIS_URL": f"redis://127.0.0.1:{args.redis_port}/0",
                 "REDIS_ROOT": str(args.redis_root), "REDIS_DATA_DIR": str(work / "redis_data"),
                 "REDIS_PREFIX": prefix}
    env.update(redis_env)
    external = {"format_version": 3, "asset_mode": "external", "sources": [
        {"id": "old", "data_root": str(args.old_index),
         "keyframes_dir": str((args.old_index / "keyframes").resolve()),
         "map_dir": str((args.old_index / "map-keyframes").resolve()), "videos_dir": str(args.old_videos_dir)},
        {"id": "new", "data_root": str(args.extracted_dir),
         "keyframes_dir": str(args.extracted_dir / "keyframes"), "map_dir": str(args.extracted_dir / "map-keyframes"),
         "videos_dir": str(args.videos_dir), "mount_root": "/mnt/d" if str(args.extracted_dir).startswith("/mnt/d/") else None,
         "require_mount": str(args.extracted_dir).startswith("/mnt/d/")},
    ], "video_sources": {**dict.fromkeys(sorted(old_ids), "old"), **dict.fromkeys(sorted(new_ids), "new")}}
    save_json(work / "external_storage.json", external)
    new_external = {**external, "sources": [external["sources"][1]], "video_sources": dict.fromkeys(sorted(new_ids), "new")}
    save_json(new / "external_storage.json", new_external)
    stages = STAGES if args.stage == "all" else (args.stage,)
    for stage in progress_items(stages, "Pipeline", unit="stage", stages=True):
        tqdm.write(f"\n=== {stage.upper()} ===")
        if stage == "audit":
            report = {"videos": len(records), "s01_asr_videos": sum(r["asr"] for r in records),
                      "images": sum(len(r["files"]) for r in records), "old_frames": old_manifest["frame_count"],
                      "free_gib": round(shutil.disk_usage(work).free / 2**30, 2),
                      "videos_detail": [{"video": r["video"].stem, "images": len(r["files"]),
                                         "map_rows": r["map_rows"], "asr": r["asr"]} for r in records]}
            save_json(work / "audit.json", report)
            print(json.dumps({k: v for k, v in report.items() if k != "videos_detail"}, indent=2))
            continue
        if stage == "dedup":
            dedup_records(records, work, args.threshold, args.workers, args.batch_size, args.dry_run)
            if args.dry_run:
                return
            records = inventory(args.extracted_dir, args.videos_dir)
        else:
            for record in records:
                plan = work / "dedup_plans" / (record["folder"].name + ".json")
                if (not plan.exists() or not read_json(plan).get("complete")
                        or read_json(plan)["result_signature"] != record["signature"]
                        or read_json(plan).get("map_signature") != file_digest(record["map"])):
                    raise RuntimeError("Complete dedup first; indexed image set must not change")
            expected_new = sum(len(r["files"]) for r in records)
            if stage in {"verify", "redis", "activate"}:
                expected_provenance = {"recipe": recipe,
                                       "new_bundle": file_digest(new / "serve_manifest.json"),
                                       "new_asr": (file_digest(new / "asr_cache.json") if (new / "asr_cache.json").is_file() else None)}
                if read_json(merged / "new_dataset_recipe.json") != expected_provenance:
                    raise RuntimeError("Merged output does not match this pipeline run")
                merged_manifest = read_json(merged / "serve_manifest.json")
                new_manifest = read_json(new / "serve_manifest.json")
                for field in ("asr_segment_count", "asr_video_count"):
                    if merged_manifest[field] != old_manifest[field] + new_manifest[field]:
                        record_issue('pipeline.asr', f'External ASR counts differ: {field}', path=merged)
            if stage in {"index", "bundle"} and merged.exists():
                print("Merged output already published; reuse frozen new index")
                verify_bundle(new, expected_new)
                continue
            if stage == "index":
                run([sys.executable, CODE / "ocr_index.py"], env, work / "logs" / "index.log")
                with connect_ro(new / "meta.db") as connection:
                    if connection.execute("SELECT COUNT(*) FROM frames").fetchone()[0] != expected_new:
                        record_issue('pipeline.metadata', 'Incomplete image index; completed frames will be bundled', path=new)
            elif stage == "bundle":
                run([sys.executable, CODE / "prepare_serve_bundle.py", "--data-root", new,
                     "--clip-model", old_manifest["clip_model"], "--allow-pending-ocr"], env, work / "logs" / "bundle.log")
                verify_bundle(new, expected_new)
            elif stage == "merge":
                verify_bundle(new, expected_new)
                provenance = {"recipe": recipe, "new_bundle": file_digest(new / "serve_manifest.json"),
                              "new_asr": (file_digest(new / "asr_cache.json") if (new / "asr_cache.json").is_file() else None)}
                save_json(work / "merge_provenance.json", provenance)
                if merged.exists():
                    if not (merged / "new_dataset_recipe.json").exists() or read_json(merged / "new_dataset_recipe.json") != provenance:
                        raise RuntimeError("Merged output already exists without matching provenance; choose another --merged-index")
                    print("Merged output already exists; verifying it")
                else:
                    run([sys.executable, CODE / "merge_processed_data.py", "--sources", args.old_index, new,
                         "--output", merged, "--allow-incomplete-ocr", "--asset-mode", "external", "--external-storage-manifest",
                         work / "external_storage.json", "--pipeline-provenance", work / "merge_provenance.json"],
                        env, work / "logs" / "merge.log")
                verify_bundle(merged, old_manifest["frame_count"] + expected_new)
            elif stage == "verify":
                report = verify_bundle(merged, old_manifest["frame_count"] + expected_new)
                save_json(work / "verification.json", report)
                print(json.dumps(report, indent=2))
            elif stage == "redis":
                verify_bundle(merged, old_manifest["frame_count"] + expected_new)
                run(["bash", CODE / "start_redis_wsl.sh"], env, work / "logs" / "redis.log")
                run([sys.executable, CODE / "migrate_to_redis.py", "--source-index", merged,
                     "--media-root", args.extracted_dir, "--summary-index", args.summary_index,
                     "--redis-url", redis_env["REDIS_URL"], "--prefix", prefix], env, work / "logs" / "redis.log")
                state["redis_ready"] = True
            elif stage == "activate":
                verify_bundle(merged, old_manifest["frame_count"] + expected_new)
                run([sys.executable, CODE / "migrate_to_redis.py", "--verify-only", "--expected-source-index", merged,
                     "--redis-url", redis_env["REDIS_URL"], "--prefix", prefix], env, work / "logs" / "activate.log")
                backup = work / "previous_ui.json"
                if not backup.exists():
                    save_json(backup, read_json(active) if active.exists() else None)
                ui_env = {**redis_env, "DATA_ROOT": str(merged), "KEYFRAMES_DIR": str(merged / "keyframes"),
                          "MAPFILES_DIR": str(merged / "map-keyframes"), "VIDEOS_DIR": str(args.videos_dir),
                          "BACKEND": "hybrid", "REQUIRE_REDIS_SOURCE_MATCH": "1"}
                save_json(active, {"format_version": 1, "environment": ui_env})
                print("UI configured. Restart using your existing TEAM_VENV ... run_local_server_wsl.sh command.")
        state["last_completed_stage"] = stage
        save_json(state_file, state)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted. Run the same command again to resume.", file=sys.stderr)
        raise SystemExit(130)
    except (OSError, RuntimeError, ValueError, KeyError, ImportError, sqlite3.Error) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
