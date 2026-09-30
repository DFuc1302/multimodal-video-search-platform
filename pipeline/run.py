#!/usr/bin/env python3
"""Resumable S01 preparation, GPU dedup, CLIP/YOLO/OCR, ASR and federated search.

Run in WSL. Original videos and extracted images are not modified. JPEG hardlinks
save SSD space; deleting a duplicate in this dataset does not delete the source.
Uses the existing ocr_index.py, asr_index.py and prepare_serve_bundle.py pipeline.
"""
from __future__ import annotations

import argparse
import codecs
import csv
import importlib.util
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
from pathlib import Path

# The processing modules remain at repository root during the migration. Keep
# this explicit so the orchestrator can be moved without changing subprocess
# working directories or the existing stage scripts.
CODE = Path(__file__).resolve().parents[1]
CATEGORY = "05_The_thao_dua_xe_dap_va_thi_dau_duong_pho"
STAGES = ("prepare", "dedup", "index", "asr", "bundle")


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def run(command, env, log_path):
    print("\nRunning:", " ".join(map(str, command)), flush=True)
    with log_path.open("a", encoding="utf-8") as log:
        with subprocess.Popen(list(map(str, command)), cwd=CODE, env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, errors="replace", bufsize=1) as child:
            try:
                decoder = codecs.getincrementaldecoder(child.stdout.encoding)(errors="replace")
                for chunk in iter(lambda: child.stdout.buffer.read1(4096), b""):
                    line = decoder.decode(chunk)
                    # Preserve carriage returns so child tqdm bars refresh in place.
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                tail = decoder.decode(b"", final=True)
                if tail:
                    print(tail, end="", flush=True)
                    log.write(tail)
                result = child.wait()
            except BaseException:
                child.terminate()
                try:
                    child.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
                raise
    if result:
        raise RuntimeError(f"Command failed (exit {result}); see {log_path}")


def read_map(path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["n", "pts_time", "fps", "frame_idx"]:
            raise RuntimeError(f"Unexpected map schema: {path}")
        rows = list(reader)
    ids = [int(row["n"]) for row in rows]
    if not rows or len(set(ids)) != len(ids):
        raise RuntimeError(f"Empty/duplicate map entries: {path}")
    return rows


def sync_map(source, folder, destination):
    rows = read_map(source)
    present = {int(p.stem) for p in folder.glob("*.jpg") if p.stem.isdigit()}
    kept = [row for row in rows if int(row["n"]) in present]
    if not present or len(kept) != len(present):
        raise RuntimeError(f"Missing or inconsistent image/map pairs: {folder}")
    temporary = destination.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["n", "pts_time", "fps", "frame_idx"])
        writer.writeheader()
        writer.writerows(kept)
    temporary.replace(destination)
    return len(kept)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("all", *STAGES, "serve"), default="all")
    parser.add_argument("--extracted-dir", type=Path, default=Path("/mnt/d/AIC_26/data/extracted"))
    parser.add_argument("--videos-dir", type=Path, default=Path("/mnt/d/AIC_26/data/videos_2"))
    parser.add_argument("--output-dir", type=Path, default=Path("/mnt/d/AIC_26/data/s01_processed"))
    parser.add_argument("--old-data-root", type=Path,
                        default=Path("/home/dtp/project/aic_local_prep/search_index_ocr"))
    parser.add_argument("--old-videos-dir", type=Path, default=Path("/mnt/d/AIC/data/videos"))
    parser.add_argument("--workers", type=int, default=4, help="JPEG reader threads for dedup")
    parser.add_argument("--batch-size", type=int, default=256, help="GPU dedup batch size")
    parser.add_argument("--threshold", type=int, default=4)
    parser.add_argument("--clip-batch-size", type=int, default=8)
    parser.add_argument("--ocr-batch-size", type=int, default=8)
    parser.add_argument("--whisper-model", default="small")
    parser.add_argument("--port", type=int, default=8081)
    args = parser.parse_args()
    if os.name != "posix":
        parser.error("Run this script in WSL/Linux")
    if min(args.workers, args.batch_size, args.clip_batch_size, args.ocr_batch_size) < 1:
        parser.error("Worker and batch sizes must be positive")
    if not 0 <= args.threshold <= 64:
        parser.error("threshold must be between 0 and 64")
    root = args.output_dir.resolve()
    # Keep outputs separate from extracted images and the old dataset.
    for protected in (args.extracted_dir.resolve(), args.videos_dir.resolve(), args.old_data_root.resolve()):
        if root == protected or root.is_relative_to(protected) or protected.is_relative_to(root):
            parser.error(f"Output must be separate from input/old dataset: {protected}")
    root.mkdir(parents=True, exist_ok=True)
    import fcntl
    with (root / ".pipeline.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error("Another S01 pipeline is already running for this output")
        execute(args, root)
    return 0


def execute(args, root):
    index = root / "search_index"
    assets = index / "keyframes"
    category_frames = assets / CATEGORY / "keyframes"
    maps = index / "map-keyframes"
    video_manifest = root / "s01_videos.json"
    logs = root / "logs"
    for directory in (category_frames, maps, index, logs):
        directory.mkdir(parents=True, exist_ok=True)
    state_path = root / "pipeline_state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"prepared": [], "deduped": []}
    if args.stage != "serve" and state.get("dedup_policy") != "first-only-v1":
        if (index / "meta.db").exists():
            raise RuntimeError("Dedup policy changed to first-only. This output already has an index; "
                               "use a new --output-dir to avoid stale search entries")
        state["deduped"] = []
        state["dedup_policy"] = "first-only-v1"
        state.pop("search_ready", None)
    manifest_path = args.old_data_root / "serve_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"Old search index not found: {manifest_path}; set --old-data-root")
    old_manifest = json.loads(manifest_path.read_text())
    model = old_manifest["clip_model"]
    env = os.environ.copy()
    env.update({
        "DATA_ROOT": str(index), "KEYFRAMES_DIR": str(index / "keyframes"),
        "MAPFILES_DIR": str(index / "map-keyframes"), "VIDEOS_DIR": str(args.videos_dir.resolve()),
        "META_DB": str(index / "meta.db"), "FAISS_DIR": str(index / "faiss_index"),
        "ASR_CACHE_FILE": str(index / "asr_cache.json"), "ASR_FRAMES_DIR": str(index / "asr_frames"),
        "CLIP_MODEL": model, "CLIP_BATCH_SIZE": str(args.clip_batch_size),
        "FAISS_INDEX_TYPE": "hnsw", "FAISS_HNSW_M": "32", "FAISS_SAVE_EVERY": "5000",
        "FAISS_SQLITE_MAPPING": "0", "VERIFY_FAISS": "1", "REQUIRE_SERVE_BUNDLE": "0",
        "YOLO_MODEL": str(root / "models" / "yolov8m.pt"), "YOLO_IMGSZ": "640",
        "OCR_ENGINE": "easyocr", "OCR_LANGS": "en,vi", "OCR_GPU": "1",
        "OCR_ALL_FRAMES": "1", "OCR_BATCH_SIZE": str(args.ocr_batch_size),
        "OCR_ENHANCE": "1", "OCR_USE_ANGLE_CLS": "0", "WHISPER_MODEL": args.whisper_model,
        "ASR_CHUNK_SECONDS": "2.0", "SERVE_READ_ONLY": "0", "AUTO_INDEX_ON_STARTUP": "0",
        "CLIP_TEXT_ONLY": "0", "CLIP_TEXT_MODEL_PATH": "", "PYTHONUNBUFFERED": "1",
        "BACKEND_MODE": "sqlite", "STORAGE_BACKEND": "sqlite", "FEDERATED_DATASET_CONFIG": "",
        "HF_HOME": str(root / "model_cache" / "huggingface"),
        "TORCH_HOME": str(root / "model_cache" / "torch"),
        "XDG_CACHE_HOME": str(root / "model_cache"),
        "EASYOCR_MODULE_PATH": str(root / "model_cache" / "easyocr"),
    })
    (root / "models").mkdir(exist_ok=True)
    config = root / "search_roots.json"
    if args.stage == "serve":
        if not config.is_file() or not (index / "serve_manifest.json").is_file():
            raise RuntimeError("Run the full pipeline/bundle stage before serving")
        env.update({"FEDERATED_DATASET_CONFIG": str(config), "FEDERATED_ALLOW_OFFLINE": "0",
                    "SERVE_READ_ONLY": "1", "META_CACHE_MODE": "lazy", "FAISS_SQLITE_MAPPING": "1",
                    "FAISS_MMAP": "1", "REQUIRE_SERVE_BUNDLE": "0", "OCR_ON_DEMAND": "0",
                    "ALLOW_HEAVY_RUNTIME_JOBS": "0", "PRELOAD_CLIP_ON_STARTUP": "1",
                    "CLIP_TEXT_ONLY": "1", "CLIP_TEXT_MODEL_PATH": str(index / "clip_text")})
        os.chdir(CODE)
        os.execvpe(sys.executable, [sys.executable, "-m", "uvicorn", "main:app",
                                   "--host", "127.0.0.1", "--port", str(args.port)], env)

    modules = ["torch", "imagehash", "PIL", "numpy", "tqdm", "faiss", "easyocr",
               "ultralytics", "transformers", "whisper", "fastapi", "uvicorn"]
    missing = [module for module in modules if importlib.util.find_spec(module) is None]
    if missing:
        raise RuntimeError("Missing dependencies in this Python environment: " + ", ".join(missing))
    if not shutil.which("ffmpeg"):
        raise RuntimeError("ffmpeg is required for ASR")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("PyTorch cannot access CUDA")
    videos = sorted(p for p in args.videos_dir.rglob("*")
                    if p.is_file() and re.fullmatch(r"S01[-_]V\d+", p.stem, re.I)
                    and p.suffix.lower() in {".mp4", ".mov", ".mkv", ".avi", ".webm"})
    if not videos or len({p.stem for p in videos}) != len(videos):
        raise RuntimeError("No S01 videos found, or duplicate video IDs")
    recipe = {"videos": [str(p.resolve()) for p in videos], "source": str(args.extracted_dir.resolve()),
              "threshold": args.threshold, "hash": "phash", "clip_model": model,
              "whisper_model": args.whisper_model}
    if state.get("recipe") not in (None, recipe):
        raise RuntimeError("Pipeline inputs/settings changed; use another --output-dir")
    state["recipe"] = recipe
    save_json(state_path, state)
    print(f"S01 videos: {len(videos)}; CLIP: {model}; output: {root}", flush=True)
    save_json(video_manifest, {"format_version": 1, "assignments": {
        video.relative_to(args.videos_dir).as_posix(): "s01" for video in videos
    }})
    stages = STAGES if args.stage == "all" else (args.stage,)

    for stage in stages:
        print(f"\n=== {stage.upper()} ===", flush=True)
        if stage == "prepare":
            for video in videos:
                source = args.extracted_dir / "keyframes" / video.stem
                source_map = args.extracted_dir / "map-keyframes" / f"{video.stem}.csv"
                if not source.is_dir() or not source_map.is_file():
                    raise RuntimeError(f"Extract frames and map first: {video.stem}")
                target = category_frames / video.stem
                target.mkdir(exist_ok=True)
                if video.stem not in state["prepared"]:
                    for image in source.glob("*.jpg"):
                        output = target / image.name
                        if not output.exists():
                            try:
                                os.link(image, output)
                            except OSError:
                                shutil.copy2(image, output)
                    sync_map(source_map, target, maps / source_map.name)
                    state["prepared"].append(video.stem)
                    save_json(state_path, state)
                print(f"Prepared {video.stem}", flush=True)
        elif stage == "dedup":
            from dedup_keyframes import GPUHasher, process_video
            gpu = GPUHasher(args.workers, args.batch_size)
            try:
                for video in videos:
                    if video.stem not in state["prepared"]:
                        raise RuntimeError("Run --stage prepare first")
                    if video.stem in state["deduped"]:
                        continue
                    target = category_frames / video.stem
                    result = process_video((str(target), args.threshold, "phash", False), gpu)
                    if result[4]:
                        raise RuntimeError(f"Dedup errors for {video.stem}: {result[4][:10]}")
                    sync_map(args.extracted_dir / "map-keyframes" / f"{video.stem}.csv",
                             target, maps / f"{video.stem}.csv")
                    state["deduped"].append(video.stem)
                    save_json(state_path, state)
                    print(f"{video.stem}: removed {result[2]} / {result[1]}", flush=True)
            finally:
                gpu.close()
        elif stage == "index":
            if set(state["deduped"]) != {v.stem for v in videos}:
                raise RuntimeError("Complete prepare and dedup before indexing")
            run([sys.executable, CODE / "ocr_index.py"], env, logs / "index.log")
            expected = sum(len(read_map(maps / f"{v.stem}.csv")) for v in videos)
            with sqlite3.connect(index / "meta.db") as db:
                actual = db.execute("SELECT COUNT(*) FROM frames").fetchone()[0]
            if actual != expected:
                raise RuntimeError(f"Incomplete image index: {actual}/{expected}; rerun --stage index")
        elif stage == "asr":
            if not all(v.stem in state["prepared"] for v in videos):
                raise RuntimeError("Run --stage prepare first")
            run([sys.executable, CODE / "asr_index.py", "--folder", args.videos_dir,
                 "--video-list", video_manifest, "--shard-id", "s01"], env, logs / "asr.log")
            cache = json.loads((index / "asr_cache.json").read_text())
            transcripts = index / "transcripts_asr"
            transcripts.mkdir(exist_ok=True)
            for video in videos:
                key = f"video-name-v1::{video.name}"
                if key not in cache:
                    raise RuntimeError(f"Missing ASR for {video.name}; rerun --stage asr")
                save_json(transcripts / f"{video.stem}.json", cache[key])
        elif stage == "bundle":
            if not (index / "asr_cache.json").is_file():
                raise RuntimeError("Complete ASR before building the search bundle")
            run([sys.executable, CODE / "prepare_serve_bundle.py", "--data-root", index,
                 "--clip-model", model], env, logs / "bundle.log")
            save_json(config, {"shards": [
                {"id": "old", "data_root": str(args.old_data_root.resolve()),
                 "videos_dir": str(args.old_videos_dir.resolve())},
                {"id": "s01", "data_root": str(index), "videos_dir": str(args.videos_dir.resolve()),
                 "video_manifest": str(video_manifest)},
            ]})
            state["search_ready"] = True
            save_json(state_path, state)
    print(f"\nFinished requested stages. Logs: {logs}", flush=True)
    if config.exists():
        print("Start search with: python run_s01_pipeline.py --stage serve", flush=True)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, OSError, ValueError, sqlite3.Error) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
