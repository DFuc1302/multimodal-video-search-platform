#!/usr/bin/env python3
"""Extract 2 FPS JPEG keyframes with NVIDIA GPU decoding (CPU mode optional)."""

from __future__ import annotations

import argparse
import csv
import logging
import os
import re
import shutil
import subprocess
import tempfile
from collections import defaultdict, deque
from fractions import Fraction
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".webm", ".mov", ".m4v", ".ts", ".m2ts"}
FRAME_INFO = re.compile(r"\bn:\s*(\d+)\s+pts:\s*(-?\d+)\s+pts_time:\s*(\S+)")


def extract_mapped(video: Path, stage: Path, ffmpeg: str, device: str) -> int:
    """Select first decoded frame in each 0.5s bin; record its true zero-based index.

    Match timestamps across the two showinfo filters, allowing decoder lookahead.
    No frame-rate conversion occurs afterward, so one selected frame is one JPEG.
    timestamp_sec is relative to the first decoded frame, not container start time.
    """
    filters = ("setpts=PTS-STARTPTS,showinfo@source=checksum=0,"
               "select='isnan(prev_selected_t)+"
               "gt(floor(t*2),floor(prev_selected_t*2))',showinfo@selected=checksum=0,"
               "settb=expr=1/2,setpts=N")
    command = [ffmpeg, "-hide_banner", "-loglevel", "info", "-nostats", "-nostdin", "-y"]
    if device == "cuda":
        command += ["-hwaccel", "cuda"]
    # A mid-stream metadata change must not reset STARTPTS/select/showinfo state.
    # Map timestamps are logged BEFORE assigning monotonic JPEG encoder timestamps.
    # FFmpeg 6.x may still rebuild the graph for changing display-matrix metadata
    # even with reinit_filter=0. Preserve encoded pixel orientation explicitly.
    command += ["-noautorotate", "-reinit_filter", "0", "-threads", "2", "-i", str(video), "-map", "0:v:0", "-an", "-sn", "-dn",
                "-vf", filters, "-filter_threads", "1", "-fps_mode", "passthrough",
                "-threads", "2", "-q:v", "2", "-start_number", "1", str(stage / "%03d.jpg")]
    tail: deque[str] = deque(maxlen=20)
    selected = 0
    sources: dict[str, deque[int]] = defaultdict(deque)
    source_count = 0
    fps = None
    with (stage / "keyframe_map.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["n", "pts_time", "fps", "frame_idx"])
        with subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                              text=True, errors="replace") as process:
            try:
                if process.stderr is None:
                    raise RuntimeError("ffmpeg stderr pipe was not opened")
                for line in process.stderr:
                    tail.append(line.strip())
                    if "showinfo@source" in line and "config in" in line:
                        rate = re.search(r"frame_rate:\s*(\d+/\d+)", line)
                        if rate:
                            try:
                                fps = float(Fraction(rate.group(1)))
                            except ZeroDivisionError:
                                fps = 0.0  # Unknown/variable nominal rate; frame_idx remains exact.
                    match = FRAME_INFO.search(line)
                    if not match:
                        continue
                    if "showinfo@source" in line:
                        if int(match.group(1)) != source_count:
                            raise RuntimeError(f"Source filter counter changed: expected {source_count}, got {match.group(1)}; "
                                               "refusing to publish an incorrect map; " + " | ".join(tail))
                        # FFmpeg can log several input frames ahead of a selected output.
                        # Keep actual decoded ordinals, including across filter reinitialization.
                        sources[match.group(2)].append(source_count)
                        source_count += 1
                    elif "showinfo@selected" in line:
                        number, pts, timestamp = match.groups()
                        candidates = sources.get(pts)
                        if not candidates:
                            raise RuntimeError(f"No source frame for selected PTS={pts}, "
                                               f"local_n={number}, output_n={selected}, "
                                               f"decoded={source_count}; " + " | ".join(tail))
                        source_idx = candidates.popleft()
                        selected += 1
                        if fps is None:
                            raise RuntimeError("ffmpeg did not report source frame rate")
                        writer.writerow([selected, timestamp, fps, source_idx])
                code = process.wait()
            except BaseException:
                process.kill()
                process.wait()
                raise
    if code:
        raise RuntimeError(f"ffmpeg exit {code}: " + " | ".join(tail))
    actual = sum(1 for _ in stage.glob("*.jpg"))
    if not selected or actual != selected:
        raise RuntimeError(f"Frame/map mismatch: {actual} JPEGs, {selected} mapping rows")
    return selected


def worker_count(value: str) -> int:
    if value.lower() == "auto":
        return max(1, (os.cpu_count() or 1) - 2)
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("workers must be a positive integer or 'auto'") from exc
    if result < 1:
        raise argparse.ArgumentTypeError("workers must be at least 1")
    return result


def sanitize_sei(video: Path, target: Path, ffmpeg: str) -> str:
    """Remux a temporary H.264/HEVC stream without SEI; never modify source video.

    FFmpeg 6.x/CUDA may expose malformed changing display matrices in SEI.
    Removing SEI keeps coded pictures and their order, but discards supplemental
    metadata (including orientation/HDR metadata). Intended for the failing input.
    """
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise RuntimeError("ffprobe is required to recover this video's timestamp failure")
    probe = subprocess.run([ffprobe, "-v", "error", "-select_streams", "v:0",
                            "-show_entries", "stream=codec_name", "-of", "csv=p=0", str(video)],
                           capture_output=True, text=True, errors="replace", check=True)
    codec = probe.stdout.strip()
    sei_types = {"hevc": "39|40", "h264": "6"}
    if codec not in sei_types:
        raise RuntimeError(f"SEI cleanup unavailable for codec {codec!r}; supported: h264, hevc")
    result = subprocess.run([ffmpeg, "-nostdin", "-v", "error", "-y", "-i", str(video),
                             "-map", "0:v:0", "-c:v", "copy", "-bsf:v",
                             f"filter_units=remove_types={sei_types[codec]}", str(target)],
                            capture_output=True, text=True, errors="replace", check=False)
    if result.returncode:
        raise RuntimeError(f"{codec} remux failed: {result.stderr[-4000:]}")
    return codec


def extract_one(task: tuple[str, str, str, str, str]) -> tuple[str, str, int, str]:
    video_s, destination_s, ffmpeg, device, map_s = task
    video, destination = Path(video_s), Path(destination_s)
    map_path = Path(map_s)
    try:
        if any(destination.glob("*.jpg")):
            if not map_path.is_file():
                return str(video), "error", 0, "Existing frames have no map; extract into a NEW output directory"
            return str(video), "skipped", 0, "frames and map already exist"
        destination.parent.mkdir(parents=True, exist_ok=True)
        map_path.parent.mkdir(parents=True, exist_ok=True)
        # Publish only a complete video directory; interruption cannot mark partial output complete.
        with tempfile.TemporaryDirectory(prefix=".extract-", dir=destination.parent) as temporary:
            stage = Path(temporary)
            recovery = ""
            try:
                frames = extract_mapped(video, stage, ffmpeg, device)
            except RuntimeError as exc:
                if "Source filter counter changed" not in str(exc) and "Invalid pts" not in str(exc):
                    raise
                # Only discard files created by this attempt inside its private staging directory.
                for generated in stage.glob("*.jpg"):
                    generated.unlink()
                clean_video = stage / "sanitized.nut"
                try:
                    codec = sanitize_sei(video, clean_video, ffmpeg)
                    frames = extract_mapped(clean_video, stage, ffmpeg, device)
                except Exception as recovery_error:
                    raise RuntimeError(f"Original extraction failure: {exc}; "
                                       f"SEI recovery failure: {recovery_error}") from recovery_error
                clean_video.unlink()
                recovery = f"Recovered using temporary {codec} SEI cleanup; original video unchanged"
            if destination.exists():
                destination.rmdir()  # Only remove an empty destination; never delete user data.
            (stage / "keyframe_map.csv").replace(map_path)
            stage.rename(destination)
        return str(video), "done", frames, recovery
    except Exception as exc:  # isolate filesystem and subprocess failures per input
        return str(video), "error", 0, f"{type(exc).__name__}: {exc}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("/mnt/d/AIC_26/data/videos_2"))
    parser.add_argument("--output-dir", type=Path, default=Path("/mnt/d/AIC_26/data/keyframes_2"))
    parser.add_argument("--category", default="", help="optional category folder before keyframes; no automatic classification")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda",
                        help="decoder device (default: cuda; requires NVIDIA driver and CUDA-enabled ffmpeg)")
    parser.add_argument("--workers", type=worker_count, default=None,
                        help="parallel videos (default: 2 for CUDA; CPU count minus 2 for CPU)")
    parser.add_argument("--ffmpeg", default="ffmpeg", help="ffmpeg executable")
    args = parser.parse_args()
    if args.category and (args.category in {".", ".."} or "/" in args.category or "\\" in args.category):
        parser.error("category must be a single folder name")

    if not args.input_dir.is_dir():
        parser.error(f"input directory does not exist: {args.input_dir}; check the WSL drive mount")
    if args.workers is None:
        args.workers = 2 if args.device == "cuda" else max(1, (os.cpu_count() or 1) - 2)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(filename=args.output_dir / "extract_progress.log", level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", encoding="utf-8")
    logger = logging.getLogger("extract_keyframes")
    if not args.input_dir.is_dir():
        parser.error(f"input directory does not exist: {args.input_dir}")
    ffmpeg = shutil.which(args.ffmpeg)
    if not ffmpeg:
        parser.error(f"ffmpeg executable not found: {args.ffmpeg}")
    if args.device == "cuda":
        capabilities = subprocess.run([ffmpeg, "-hide_banner", "-hwaccels"],
                                      capture_output=True, text=True, errors="replace", check=False)
        if capabilities.returncode or "cuda" not in capabilities.stdout.split():
            parser.error("ffmpeg does not list CUDA support; install a CUDA-capable ffmpeg or use --device cpu")
    print(f"Decoder: {args.device}; workers: {args.workers}; output: {args.output_dir}")
    logger.info("START | device=%s workers=%d", args.device, args.workers)

    videos = sorted(p for p in args.input_dir.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS)
    # Keep the requested simple <video_name> layout; disambiguate duplicate stems deterministically.
    destinations: dict[Path, Path] = {}
    seen: set[str] = set()
    for video in videos:
        stem = video.stem
        folder = stem
        if folder.casefold() in seen:
            import hashlib
            relative = video.relative_to(args.input_dir).as_posix().encode("utf-8", "surrogateescape")
            folder = f"{stem}_{hashlib.sha1(relative).hexdigest()[:8]}"
        seen.add(folder.casefold())
        destinations[video] = args.output_dir / args.category / "keyframes" / folder

    tasks = [(str(v), str(destinations[v]), ffmpeg, args.device,
              str(args.output_dir / "map-keyframes" / (destinations[v].name + ".csv"))) for v in videos]
    counts = {"done": 0, "skipped": 0, "error": 0}
    total_frames = 0
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(extract_one, task) for task in tasks]
        for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting", unit="video"):
            try:
                video, status, frames, message = future.result()
            except Exception as exc:
                video, status, frames, message = "<worker>", "error", 0, f"{type(exc).__name__}: {exc}"
            counts[status] += 1
            if status == "error":
                tqdm.write(f"ERROR: {video}: {message}")
            total_frames += frames
            (logger.error if status == "error" else logger.info)("%s | %s | %d frames | %s", status, video, frames, message)
    logger.info("SUMMARY | videos=%d done=%d skipped=%d errors=%d frames=%d", len(videos), counts["done"],
                counts["skipped"], counts["error"], total_frames)
    print(f"Finished: {counts['done']} extracted, {counts['skipped']} skipped, {counts['error']} errors; {total_frames} frames.")
    return 1 if counts["error"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
