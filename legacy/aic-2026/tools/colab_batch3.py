"""Process the 11 Batch 3 archives on Colab; publish portable, video-free bundles.

Completed archives are checkpoints. An interrupted archive is rebuilt locally.
All paths under Drive/videos are read-only. Only runner-owned temporary data is removed.
"""
from __future__ import annotations

import argparse
from contextlib import closing, nullcontext
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import time
import zipfile
from fractions import Fraction
from tqdm import tqdm
from batch3_speed import environment as speed_environment, profile as speed_profile

from re_extract import VIDEO_EXTENSIONS, frame_index_at, make_select_filter, probe_declared_fps

ARCHIVES = [f"Videos_M{i:02}.zip" for i in range(1, 11)] + ["Video_N031-N040.zip"]
BASE_URL = "https://aic-data.ledo.io.vn/"
CODE = Path(__file__).resolve().parent


def progress_bar(*, total=None, desc, unit="item", **kwargs):
    # Workers run outside the notebook kernel; text tqdm renders in Colab output.
    return tqdm(total=total, desc=desc, unit=unit, file=sys.stdout,
                mininterval=1.0, dynamic_ncols=True, position=0, **kwargs)


def progress_items(items, desc, unit="item", *, total=None):
    with progress_bar(total=len(items) if total is None else total, desc=desc, unit=unit) as bar:
        for item in items:
            yield item
            # Do not advance when the work raises, including failed subprocesses.
            bar.update(1)


def copy_with_progress(source, destination, size, desc):
    checksum = hashlib.sha256()
    with progress_bar(total=size, desc=desc, unit="B", unit_scale=True, unit_divisor=1024) as bar:
        while block := source.read(4 * 1024**2):
            destination.write(block)
            checksum.update(block)
            bar.update(len(block))
    return checksum.hexdigest()


def recipe_matches(saved, expected):
    if saved == expected:
        return True
    from colab_batch3_progress_compat import PRE_PROGRESS_CODE, PRE_BUDGET_FIX_CODE, PRE_SPEED_CODE
    if expected.get('speed_profile') == 'a100-fast' and saved.get('version') == 5:
        # Retain completed full-resolution OCR. Newly processed groups record
        # reduced OCR input separately. Models and sampling must still match.
        expected = {k: v for k, v in expected.items() if k != 'speed_profile'}
        expected['version'] = 5
    # Only explicitly reviewed snapshots with unchanged data-processing behavior.
    return (saved.get("code") in (PRE_PROGRESS_CODE, PRE_BUDGET_FIX_CODE, PRE_SPEED_CODE) and
            {k: v for k, v in saved.items() if k != "code"} ==
            {k: v for k, v in expected.items() if k != "code"})


def group_for_video(stem):
    match = re.match(r"^(M\d{2}|N\d{3})(?:[-_]|$)", stem, re.I)
    if not match:
        return None
    group = match[1].upper()
    if group.startswith("M") and 1 <= int(group[1:]) <= 10:
        return f"Videos_{group}.zip"
    if group.startswith("N") and 31 <= int(group[1:]) <= 40:
        return "Video_N031-N040.zip"
    return None


def loose_videos(video_root, name):
    found, seen = [], set()
    for path in sorted(video_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        group = group_for_video(path.stem)
        if group is None:
            raise RuntimeError(f"Cannot assign video to M01-M10/N031-N040: {path}")
        if group != name:
            continue
        if path.stem.casefold() in seen:
            raise RuntimeError(f"Duplicate video ID: {path.stem}")
        seen.add(path.stem.casefold())
        found.append(path)
    if not found:
        raise RuntimeError(f"No videos for {name} under {video_root}")
    return found


def digest(path, desc=None):
    h = hashlib.sha256()
    progress = (progress_bar(total=Path(path).stat().st_size, desc=desc, unit="B",
                            unit_scale=True, unit_divisor=1024) if desc else nullcontext(None))
    with Path(path).open("rb") as f, progress as bar:
        for block in iter(lambda: f.read(4 * 1024**2), b""):
            h.update(block)
            if bar is not None:
                bar.update(len(block))
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".partial")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def video_members(z):
    selected, seen = [], set()
    for item in z.infolist():
        name = PurePosixPath(item.filename.replace("\\", "/"))
        if name.is_absolute() or ".." in name.parts or ":" in item.filename:
            raise RuntimeError(f"Unsafe archive path: {item.filename}")
        if item.is_dir() or name.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        if not re.fullmatch(r"[A-Za-z0-9_-]+", name.stem):
            raise RuntimeError(f"Unsupported video ID: {name.stem}")
        if name.stem.casefold() in seen:
            raise RuntimeError(f"Duplicate video ID: {name.stem}")
        seen.add(name.stem.casefold())
        selected.append(item)
    if not selected:
        raise RuntimeError("ZIP contains no supported videos")
    return selected


def extract_frames(video, root, interval=Fraction(1, 2)):
    """Keep original decoded frame numbers AND actual presentation timestamps."""
    fps = probe_declared_fps("ffprobe", video)
    if fps is None or fps <= 0:
        raise RuntimeError(f"Cannot probe FPS: {video}")
    folder = root / "keyframes" / video.stem
    folder.mkdir(parents=True)
    maps = root / "map-keyframes"
    maps.mkdir(exist_ok=True)
    interval = Fraction(str(interval))
    # The select expression chooses floor(k*interval*fps) in the ORIGINAL stream.
    # Log only selected PTS, without expensive pixel checksums on every decoded frame.
    filt = make_select_filter(interval, fps) + ",showinfo=checksum=0"
    command = ["ffmpeg", "-hide_banner", "-nostats", "-copyts", "-i", str(video),
               "-map", "0:v:0", "-an", "-vf", filt, "-fps_mode", "vfr",
               "-q:v", "2", "-threads", "2", str(folder / "%04d.jpg")]
    rows, tail = [], []
    pattern = re.compile(r"\bn:\s*(\d+).*?\bpts_time:\s*([-+\d.eE]+)")
    with progress_bar(desc=f"Extract {video.stem}", unit="frame") as bar, \
         subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                          text=True, errors="replace") as proc:
        for line in proc.stderr:
            tail = (tail + [line])[-12:]
            match = pattern.search(line) if "showinfo" in line else None
            if match:
                if int(match[1]) != len(rows):
                    raise RuntimeError("Unexpected selected frame sequence")
                rows.append({"n": len(rows) + 1, "pts_time": match[2],
                             "fps": float(fps), "frame_idx": frame_index_at(len(rows), interval, fps)})
                bar.update(1)
        if proc.wait():
            raise RuntimeError("ffmpeg failed: " + "".join(tail))
    if not rows or len(list(folder.glob("*.jpg"))) != len(rows):
        raise RuntimeError(f"Frame/map count mismatch: {video.name}")
    with (maps / f"{video.stem}.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["n", "pts_time", "fps", "frame_idx"])
        writer.writeheader()
        writer.writerows(rows)
    return rows


def dedup_frames(root, stem, rows, threshold):
    import imagehash
    from PIL import Image
    def hash_row(row):
        path = root / "keyframes" / stem / f"{int(row['n']):04d}.jpg"
        with Image.open(path) as image:
            return path, imagehash.phash(image)
    previous, kept = None, []
    with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 2)) as pool:
        for row, (path, current) in progress_items(
                zip(rows, pool.map(hash_row, rows)), f"Dedup {stem}", "frame", total=len(rows)):
            duplicate = previous is not None and previous - current <= threshold
            previous = current  # compare ORIGINAL adjacent frames, preserving first of run
            if duplicate:
                path.unlink()
            else:
                kept.append(row)
    with (root / "map-keyframes" / f"{stem}.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["n", "pts_time", "fps", "frame_idx"])
        writer.writeheader()
        writer.writerows(kept)
    return len(kept)


def run(script, env, *args):
    subprocess.run([sys.executable, str(CODE / script), *map(str, args)], env=env,
                   cwd=CODE, check=True)


def package(root, destination):
    """Allowlist excludes videos, model download caches and temporary source files."""
    allowed = ["keyframes", "map-keyframes", "meta.db", "faiss_index", "asr_cache.json",
               "serve_manifest.json", "serve_stats.json", "clip_text", "batch3_receipt.json"]
    partial = destination.with_suffix(".zip.partial")
    files = []
    for name in allowed:
        path = root / name
        files.extend(p for p in (sorted(path.rglob("*")) if path.is_dir() else [path]) if p.is_file())
    size = sum(p.stat().st_size for p in files)
    with zipfile.ZipFile(partial, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as z, \
         progress_bar(total=size, desc="Export ZIP to Drive", unit="B", unit_scale=True, unit_divisor=1024) as bar:
        for file in files:
            if file.suffix.lower() in VIDEO_EXTENSIONS or file.is_symlink():
                raise RuntimeError(f"Unexpected export file: {file}")
            info = zipfile.ZipInfo.from_file(file, file.relative_to(root).as_posix())
            with file.open("rb") as src, z.open(info, "w", force_zip64=True) as dst:
                while block := src.read(4 * 1024**2):
                    dst.write(block)
                    bar.update(len(block))
    # Reading every member through EOF validates its CRC, like ZipFile.testzip().
    with zipfile.ZipFile(partial) as z, progress_bar(total=size, desc="Verify ZIP CRC", unit="B",
                                                   unit_scale=True, unit_divisor=1024) as bar:
        for info in z.infolist():
            with z.open(info) as src:
                while block := src.read(4 * 1024**2):
                    bar.update(len(block))
    checksum = digest(partial, desc="Verify export SHA-256")
    partial.replace(destination)
    write_json(destination.with_suffix(".sha256.json"), {"sha256": checksum})


def process_archive(args, name, recipe):
    started = time.monotonic()
    out = args.batch_root / "processed_data"
    out.mkdir(parents=True, exist_ok=True)
    final = out / (Path(name).stem + "_processed.zip")
    loose = loose_videos(args.batch_root / "videos", name) if args.input_mode == "videos" else None
    inventory = [{"path": p.relative_to(args.batch_root / "videos").as_posix(),
                  "size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns}
                 for p in loose] if loose else None
    if final.exists():
        checksum = final.with_suffix(".sha256.json")
        if not checksum.exists() or json.loads(checksum.read_text())["sha256"] != digest(final, desc=f"Resume check {name}"):
            raise RuntimeError(f"Missing/mismatched checksum: {final}")
        with zipfile.ZipFile(final) as z:
            saved = json.loads(z.read("batch3_receipt.json"))
        if not recipe_matches(saved["recipe"], recipe):
            raise RuntimeError(f"Recipe changed for {name}; use a separate batch-root")
        if saved.get("source_inventory") != inventory:
            raise RuntimeError(f"Source video inventory changed for {name}; use a separate batch-root")
        print(f"SKIP verified complete: {name}", flush=True)
        return saved["videos"]
    args.work_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="batch3-", dir=args.work_root) as temp, \
         progress_bar(total=4, desc=f"{name}: stages", unit="stage",
                      bar_format="{desc}: {n_fmt}/{total_fmt} [{elapsed}] {postfix}") as stages:
        stages.set_postfix_str("1/4 Extract + dedup")
        work = Path(temp)
        source = args.batch_root / "videos" / name
        if loose is None and not source.is_file():
            if not args.download_missing:
                raise FileNotFoundError(f"Put {name} in {source.parent}, or enable --download-missing")
            source = work / name
            subprocess.run(["curl", "--fail", "--location", "--retry", "5",
                            "--output", str(source), BASE_URL + name], check=True)
        root = work / "processed"
        root.mkdir()
        env = os.environ.copy()
        env.update(DATA_ROOT=str(root), KEYFRAMES_DIR=str(root / "keyframes"),
                   MAPFILES_DIR=str(root / "map-keyframes"), META_DB=str(root / "meta.db"),
                   FAISS_DIR=str(root / "faiss_index"), ASR_CACHE_FILE=str(root / "asr_cache.json"),
                   BACKEND_MODE="sqlite", STORAGE_BACKEND="sqlite", CLIP_MODEL=args.clip_model,
                   CLIP_BATCH_SIZE=str(args.clip_batch_size), OCR_BATCH_SIZE=str(args.ocr_batch_size),
                   OCR_ENGINE="easyocr", OCR_GPU="1", OCR_ALL_FRAMES="1", OCR_LANGS="en,vi",
                   OCR_ENHANCE=str(args.ocr_enhance),
                   OCR_WHITE_ON_BLACK_ONLY="1" if name == "Video_N031-N040.zip" else "0",
                   YOLO_MODEL="yolov8m.pt", YOLO_BATCHED="1",
                   FAISS_INDEX_TYPE="hnsw", FAISS_SQLITE_MAPPING="0",
                   VERIFY_FAISS="1", SERVE_READ_ONLY="0", REQUIRE_SERVE_BUNDLE="0",
                   CLIP_TEXT_ONLY="0", CLIP_TEXT_MODEL_PATH="", PYTHONUNBUFFERED="1")
        env.update(speed_environment(args))
        # Compatibility placeholder only; transcription is delegated to another person.
        write_json(root / "asr_cache.json", {})
        # One group per worker process: import-time data paths cannot leak across groups.
        os.environ.update(env)
        videos, total, source_hashes = [], 0, {}
        with (nullcontext(None) if loose else zipfile.ZipFile(source)) as z:
            members = loose if loose else video_members(z)
            for i, item in enumerate(progress_items(members, f"{name}: extract + dedup", "video"), 1):
                filename = item.name if loose else PurePosixPath(item.filename.replace("\\", "/")).name
                print(f"{name} [{i}/{len(members)}] {filename}", flush=True)
                size = item.stat().st_size if loose else item.file_size
                if shutil.disk_usage(work).free < size + 2 * 1024**3:
                    raise RuntimeError("Not enough local disk for next video; use a larger Colab disk")
                input_dir = work / "input"
                input_dir.mkdir(exist_ok=True)
                video = input_dir / filename
                with (item.open("rb") if loose else z.open(item)) as src, video.open("wb") as dst:
                    source_hashes[filename] = copy_with_progress(src, dst, size, f"Copy + SHA-256 {filename}")
                if video.stat().st_size != size:
                    raise RuntimeError(f"Incomplete local video copy: {filename}")
                rows = extract_frames(video, root, args.interval)
                total += dedup_frames(root, video.stem, rows, args.threshold)
                videos.append(video.stem)
                video.unlink()
        stages.update(1)
        stages.set_postfix_str(f"2/4 CLIP + YOLO + OCR | {total:,} frames")
        visual_start = time.monotonic()
        run("ocr_index.py", env)
        stages.update(1)
        stages.set_postfix_str("3/4 Build + validate search index")
        run("prepare_serve_bundle.py", env, "--data-root", root, "--clip-model", args.clip_model)
        from run_new_dataset_pipeline import verify_bundle
        verify_bundle(root, total)
        with closing(sqlite3.connect(root / "meta.db")) as db:
            db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        receipt = {"recipe": recipe, "archive": name, "source_inventory": inventory,
                   "source_video_sha256": source_hashes, "videos": videos, "frames": total,
                   "asr_status": "delegated_not_processed", "source_url": BASE_URL + name}
        receipt["timing_seconds"] = {"extract_dedup": visual_start - started,
            "visual_index_bundle": time.monotonic() - visual_start}
        if (root / 'visual_timing.json').is_file():
            receipt['timing_seconds'].update(json.loads((root / 'visual_timing.json').read_text()))
        write_json(root / "batch3_receipt.json", receipt)
        stages.update(1)
        stages.set_postfix_str("4/4 Export + verify ZIP")
        package(root, final)
        write_json(final.with_suffix(".timing.json"), {"elapsed_seconds": time.monotonic() - started,
                                                       "frames": total, "videos": len(videos)})
        stages.update(1)
        stages.set_postfix_str("Complete")
        print(f"DONE: {final}", flush=True)
        return videos


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch-root", type=Path, required=True)
    p.add_argument("--work-root", type=Path, default=Path("/content/aic_batch3_work"))
    p.add_argument("--archives", nargs="+", choices=ARCHIVES, default=ARCHIVES)
    p.add_argument("--input-mode", choices=("videos", "zip"), default="videos")
    p.add_argument("--download-missing", action="store_true")
    p.add_argument("--clip-model", default="openai/clip-vit-large-patch14")
    p.add_argument("--threshold", type=int, default=4)
    p.add_argument("--interval", type=float, default=1.0, help="Seconds between frame samples; speed preset: 1")
    p.add_argument("--ocr-enhance", type=int, choices=(0, 1), default=0)
    p.add_argument("--clip-batch-size", type=int, default=64)
    p.add_argument("--ocr-batch-size", type=int, default=32)
    p.add_argument("--speed-profile", choices=("legacy", "a100-fast"), default="a100-fast")
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args()
    if not 0 <= args.threshold <= 64 or min(args.clip_batch_size, args.ocr_batch_size) < 1:
        p.error("Invalid threshold/batch size")
    if not 0.1 <= args.interval <= 10:
        p.error("Invalid interval")
    args.batch_root = args.batch_root.resolve()
    args.work_root = args.work_root.resolve()
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Choose a GPU runtime in Colab")
    # Source code hashes invalidate completed outputs if pipeline code is changed.
    recipe = {"version": 5, "ocr_policy": "N031-N040-white-on-black_M-unrestricted",
              "ocr_enhance": args.ocr_enhance, "asr_status": "delegated_not_processed",
              "input_mode": args.input_mode, "clip_model": args.clip_model,
              "threshold": args.threshold, "interval": args.interval,
              "code": {p.name: digest(p) for p in sorted(CODE.glob("*.py"))}}
    recipe.update(version=6, speed_profile=speed_profile(args))
    seen = set()
    if args.input_mode == "videos":
        # Check all requested groups before starting expensive GPU work.
        for name in args.archives:
            loose_videos(args.batch_root / "videos", name)
    group_iterator = args.archives if args.worker else progress_items(args.archives, "Completed groups", "group")
    for name in group_iterator:
        if args.worker:
            if len(args.archives) != 1:
                raise RuntimeError("Worker handles exactly one group")
            ids = {v.casefold() for v in process_archive(args, name, recipe)}
        else:
            command = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker",
                "--batch-root", str(args.batch_root), "--work-root", str(args.work_root),
                "--input-mode", args.input_mode, "--archives", name,
                "--clip-model", args.clip_model,
                "--threshold", str(args.threshold), "--clip-batch-size", str(args.clip_batch_size),
                "--interval", str(args.interval), "--ocr-enhance", str(args.ocr_enhance),
                "--ocr-batch-size", str(args.ocr_batch_size)]
            command += ["--speed-profile", args.speed_profile]
            if args.download_missing:
                command.append("--download-missing")
            subprocess.run(command, check=True)
            with zipfile.ZipFile(args.batch_root / "processed_data" / (Path(name).stem + "_processed.zip")) as z:
                ids = {v.casefold() for v in json.loads(z.read("batch3_receipt.json"))["videos"]}
        if seen & ids:
            raise RuntimeError(f"Duplicate IDs across ZIPs: {sorted(seen & ids)}")
        seen.update(ids)
    print(f"COMPLETE: {len(args.archives)} archives, {len(seen)} videos")


if __name__ == "__main__":
    main()
