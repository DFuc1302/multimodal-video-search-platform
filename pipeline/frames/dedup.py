#!/usr/bin/env python3
"""Keep only the first frame of each near-identical consecutive run (length >= 2)."""

from __future__ import annotations

import argparse
import logging
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import imagehash
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm


class GPUHasher:
    """One CUDA context with parallel CPU JPEG readers and bounded GPU batches."""

    def __init__(self, workers: int, batch_size: int):
        import numpy as np
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable in PyTorch; install a CUDA build or use --device cpu")
        self.np, self.torch = np, torch
        self.batch_size = batch_size
        # Match scipy.fftpack.dct(type=2, norm=None), retaining the first 8 frequencies.
        basis = 2 * np.cos(np.pi * np.arange(8)[:, None] * (2 * np.arange(32) + 1) / 64)
        self.basis = torch.as_tensor(basis, dtype=torch.float64, device="cuda")
        self.pool = ThreadPoolExecutor(max_workers=workers)

    def close(self):
        self.pool.shutdown(wait=True)

    def read(self, item):
        path, hash_name = item
        try:
            with Image.open(path) as image:
                size = (32, 32) if hash_name == "phash" else (9, 8)
                pixels = self.np.asarray(image.convert("L").resize(size, Image.Resampling.LANCZOS))
            return pixels, None
        except Exception as exc:
            return None, f"{path.name}: {type(exc).__name__}: {exc}"

    def hash_pixels(self, pixels, hash_name):
        torch, np = self.torch, self.np
        with torch.inference_mode():
            data = torch.as_tensor(np.stack(pixels), device="cuda", dtype=torch.float64)
            if hash_name == "dhash":
                bits = (data[:, :, 1:] > data[:, :, :-1]).cpu().numpy()
            else:
                dct = self.basis @ data @ self.basis.T
                flat = dct.flatten(1)
                ordered = flat.sort(dim=1).values
                median = (ordered[:, 31] + ordered[:, 32]) / 2
                bits = (dct > median[:, None, None]).cpu().numpy()
                # Constant/symmetric images can have DCT coefficients at the median.
                # Recompute those on CPU to preserve imagehash's numerical tie behavior.
                tolerance = flat.abs().amax(dim=1).clamp(min=1) * 1e-10
                ambiguous = ((flat - median[:, None]).abs().amin(dim=1) <= tolerance).cpu().numpy()
                for idx in np.flatnonzero(ambiguous):
                    bits[idx] = imagehash.phash(Image.fromarray(pixels[idx])).hash
        return [imagehash.ImageHash(array) for array in bits]

    def hashes(self, frames, hash_name, progress=None):
        hashes, errors = [], []
        for offset in range(0, len(frames), self.batch_size):
            chunk = frames[offset:offset + self.batch_size]
            loaded = list(self.pool.map(self.read, ((p, hash_name) for p in chunk)))
            pixels = [array for array, error in loaded if error is None]
            computed = iter(self.hash_pixels(pixels, hash_name)) if pixels else iter(())
            for array, error in loaded:
                if error is not None:
                    errors.append(error)
                    hashes.append(None)
                else:
                    hashes.append(next(computed))
            if progress is not None:
                progress(len(chunk))
        return hashes, errors


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


def frame_number(path: Path) -> tuple[int, str]:
    # Numeric ordering is strict for frame_000001.jpg and remains deterministic for other jpg names.
    try:
        return int(path.stem.rsplit("_", 1)[-1]), path.name
    except (IndexError, ValueError):
        return (2**63 - 1, path.name)


def process_video(task: tuple[str, int, str, bool], gpu=None) -> tuple[str, int, int, int, list[str]]:
    folder_s, threshold, hash_name, dry_run = task
    folder = Path(folder_s)
    frames = sorted((p for p in folder.glob("*.jpg") if p.is_file()), key=frame_number)
    hashes: list[imagehash.ImageHash | None] = []
    errors: list[str] = []
    if gpu is not None:
        hashes, errors = gpu.hashes(frames, hash_name)
    for image_path in ([] if gpu is not None else frames):
        try:
            with Image.open(image_path) as image:
                image.load()
                hashes.append(imagehash.phash(image) if hash_name == "phash" else imagehash.dhash(image))
        except (OSError, ValueError, UnidentifiedImageError) as exc:
            hashes.append(None)
            errors.append(f"{image_path.name}: {type(exc).__name__}: {exc}")
        except Exception as exc:
            hashes.append(None)
            errors.append(f"{image_path.name}: unexpected {type(exc).__name__}: {exc}")

    remove_indices: set[int] = set()
    # A run consists of adjacent frames whose hashes are within threshold; invalid images split runs.
    start = 0
    while start < len(frames):
        if hashes[start] is None:
            start += 1
            continue
        end = start
        while end + 1 < len(frames) and hashes[end + 1] is not None and (hashes[end] - hashes[end + 1]) <= threshold:
            end += 1
        if end > start:
            remove_indices.update(range(start + 1, end + 1))
        start = end + 1

    removed_bytes = 0
    removed_count = 0
    for idx in sorted(remove_indices):
        path = frames[idx]
        try:
            size = path.stat().st_size
            if not dry_run:
                path.unlink()
            removed_count += 1
            removed_bytes += size
        except OSError as exc:
            errors.append(f"{path.name}: could not {'count' if dry_run else 'delete'}: {exc}")
    return str(folder), len(frames), removed_count, removed_bytes, errors


def results(tasks, workers, gpu):
    if gpu is not None:
        for task in tasks:
            try:
                yield process_video(task, gpu)
            except Exception as exc:
                yield task[0], 0, 0, 0, [f"{type(exc).__name__}: {exc}"]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {executor.submit(process_video, task): task[0] for task in tasks}
            for future in as_completed(futures):
                try:
                    yield future.result()
                except Exception as exc:
                    yield futures[future], 0, 0, 0, [f"{type(exc).__name__}: {exc}"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("/mnt/d/AIC_26/data/keyframes_2"),
                        help="root containing one folder per video's frames")
    parser.add_argument("--workers", type=worker_count, default=None,
                        help="CUDA: JPEG reader threads (default 4); CPU: video processes")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=256, help="images per GPU hash batch")
    parser.add_argument("--threshold", type=int, default=4, help="maximum pHash/dHash Hamming distance")
    parser.add_argument("--hash", choices=("phash", "dhash"), default="phash")
    parser.add_argument("--dry-run", action="store_true", help="report removals without deleting files")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("batch-size must be positive")
    if args.workers is None:
        args.workers = 4 if args.device == "cuda" else max(1, (os.cpu_count() or 1) - 2)
    if args.threshold < 0:
        parser.error("threshold must be non-negative")
    if not args.input_dir.is_dir():
        parser.error(f"input directory does not exist: {args.input_dir}")
    gpu = None
    if args.device == "cuda":
        try:
            gpu = GPUHasher(args.workers, args.batch_size)
        except (ImportError, RuntimeError) as exc:
            parser.error(f"{exc}. Install torch in this Python environment; CPU fallback: --device cpu")
    print(f"Device: {args.device}; hash: {args.hash}; workers: {args.workers}; batch: {args.batch_size}; "
          "policy: keep FIRST only, minimum run=2")

    log_path = args.input_dir / "dedup_summary.log"
    logging.basicConfig(filename=log_path, level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        encoding="utf-8")
    logger = logging.getLogger("dedup_keyframes")
    folders = sorted({p.parent for p in args.input_dir.rglob("*.jpg")
                      if not any(part.startswith(".extract-") for part in p.parts)})
    tasks = [(str(folder), args.threshold, args.hash, args.dry_run) for folder in folders]
    errors_total = 0
    removed_total = bytes_total = frames_total = 0
    mode = "DRY-RUN" if args.dry_run else "DELETE"
    try:
        for folder, frames, removed, size, errors in tqdm(results(tasks, args.workers, gpu),
                                                        total=len(tasks), desc="Deduplicating", unit="video"):
            frames_total += frames
            removed_total += removed
            bytes_total += size
            errors_total += len(errors)
            logger.info("%s | %s | inspected=%d removed=%d bytes=%d", mode, folder, frames, removed, size)
            for error in errors:
                logger.error("%s | %s | %s", folder, mode, error)
                tqdm.write(f"ERROR: {folder}: {error}")
    finally:
        if gpu is not None:
            gpu.close()
    logger.info("SUMMARY | mode=%s folders=%d frames=%d removed=%d bytes=%d errors=%d",
                mode, len(folders), frames_total, removed_total, bytes_total, errors_total)
    print(f"{mode}: inspected {frames_total} frames; {'would remove' if args.dry_run else 'removed'} "
          f"{removed_total} ({bytes_total / (1024 * 1024):.2f} MiB); {errors_total} errors.")
    return 1 if errors_total else 0


if __name__ == "__main__":
    raise SystemExit(main())
