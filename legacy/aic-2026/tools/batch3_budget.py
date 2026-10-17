"""Calibrate remaining visual work from completed groups; separate sample startup cost."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import zipfile

from colab_batch3 import ARCHIVES, loose_videos, write_json, progress_items, digest
from batch3_speed import profile as speed_profile


def duration(path):
    raw = subprocess.check_output(["ffprobe", "-v", "error", "-show_entries",
        "format=duration", "-of", "json", str(path)], text=True)
    seconds = float(json.loads(raw)["format"]["duration"])
    if not math.isfinite(seconds) or seconds <= 0:
        raise RuntimeError(f"Invalid duration: {path}")
    return path, seconds


def family(name):
    return "M" if name.startswith("Videos_M") else "N"


def completed_measurement(args, name, items):
    archive = args.batch_root / "processed_data" / (Path(name).stem + "_processed.zip")
    if not archive.exists():
        return None
    checksum = archive.with_suffix(".sha256.json")
    if not checksum.is_file() or digest(archive, desc=f"Check completed {name}") != json.loads(checksum.read_text())["sha256"]:
        raise RuntimeError(f"Completed ZIP/checksum mismatch: {archive}")
    with zipfile.ZipFile(archive) as z:
        receipt = json.loads(z.read("batch3_receipt.json"))
    recipe = receipt["recipe"]
    for field, expected in {"clip_model": args.clip_model, "interval": args.interval,
                            "ocr_enhance": args.ocr_enhance,
                            "asr_status": "delegated_not_processed", "threshold": 4}.items():
        if recipe.get(field) != expected:
            raise RuntimeError(f"Completed {name} has different {field}; do not mix recipes")
    inventory = [{"path": path.relative_to(args.batch_root / "videos").as_posix(),
                  "size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns} for path, _ in items]
    if receipt.get("source_inventory") != inventory:
        raise RuntimeError(f"Source changed since {name} was processed")
    timing_file = archive.with_suffix(".timing.json")
    elapsed = None
    if timing_file.is_file():
        timing = json.loads(timing_file.read_text())
        elapsed = float(timing["elapsed_seconds"])
        if (not math.isfinite(elapsed) or elapsed <= 0 or timing["videos"] != len(items)
                or timing["frames"] != receipt["frames"]):
            raise RuntimeError(f"Invalid full-group timing: {timing_file}")
    return {"group": name, "videos": len(items), "source_hours": sum(d for _, d in items)/3600,
            "status": "completed", "method": "verified_completed_group",
            "speed_profile": recipe.get('speed_profile', 'legacy'),
            "actual_elapsed_hours": elapsed / 3600 if elapsed is not None else None,
            "estimated_hours": 0.0}


def fit_sample_cost(samples):
    """Split fixed startup from sustained processing using two different durations."""
    (d1, t1), (d2, t2) = sorted(samples)
    if min(d1, d2, t1, t2) <= 0 or not all(math.isfinite(x) for pair in samples for x in pair):
        raise ValueError("Invalid sample measurements")
    if d2 - d1 < 30 or t2 <= t1:
        return {"fixed_seconds": 0.0, "seconds_per_source_second": max(t1/d1, t2/d2),
                "method": "inconclusive_samples_conservative_bound", "low_confidence": True}
    slope = (t2-t1)/(d2-d1)
    intercept = t1-slope*d1
    if intercept < 0:
        slope, intercept = max(t1/d1, t2/d2), 0.0
    return {"fixed_seconds": intercept, "seconds_per_source_second": slope,
            "method": "two_length_samples", "low_confidence": True}


def remaining_from_actual(name, source_seconds, completed):
    candidates = [r for r in completed if family(r["group"]) == family(name)
                  and r.get("actual_elapsed_hours") is not None and r["source_hours"] > 0]
    if not candidates:
        return None
    # Use the slowest SAME-family observed rate; M is not a calibration for N.
    anchor = max(candidates, key=lambda r: r["actual_elapsed_hours"] / r["source_hours"])
    return {"estimated_hours": source_seconds/3600 * anchor["actual_elapsed_hours"]/anchor["source_hours"],
            "method": "completed_same_family_rate", "calibration_group": anchor["group"],
            "low_confidence": True}


def sample_group(args, name, items):
    path, seconds = sorted(items, key=lambda item: item[1])[len(items)//2]
    long_seconds = min(600.0, seconds)
    short_seconds = min(60.0, long_seconds/2)
    measured = []
    phase_measurements = []
    with tempfile.TemporaryDirectory(prefix="aic-budget-") as tmp:
        for index, length in enumerate((short_seconds, long_seconds)):
            root = Path(tmp) / str(index)
            (root / "videos").mkdir(parents=True)
            sample = root / "videos" / (path.stem + ".mp4")
            subprocess.run(["ffmpeg", "-v", "error", "-ss", str(max(0, seconds/2-length/2)),
                "-i", str(path), "-t", str(length), "-map", "0:v:0", "-an", "-c", "copy",
                "-avoid_negative_ts", "make_zero", str(sample)], check=True)
            actual_length = duration(sample)[1]
            tick = time.monotonic()
            subprocess.run([sys.executable, "-u", str(Path(__file__).with_name("colab_batch3.py")),
                "--batch-root", str(root), "--archives", name, "--clip-model", args.clip_model,
                "--clip-batch-size", str(args.clip_batch_size), "--ocr-batch-size", str(args.ocr_batch_size),
                "--interval", str(args.interval), "--ocr-enhance", str(args.ocr_enhance),
                "--speed-profile", speed_profile(args)], check=True)
            measured.append((actual_length, time.monotonic()-tick))
            with zipfile.ZipFile(root / 'processed_data' / (Path(name).stem + '_processed.zip')) as z:
                phase_measurements.append(json.loads(z.read('batch3_receipt.json')).get('timing_seconds', {}))
    fit = fit_sample_cost(measured)
    return {**fit, "sample_video": str(path), "sample_measurements": measured,
            "sample_phase_seconds": phase_measurements,
            "estimated_hours": (fit["fixed_seconds"] + fit["seconds_per_source_second"] *
                                sum(d for _, d in items))/3600}


def summarize(groups, benchmark_hours, reserve_hours, target_hours):
    remaining = sum(g["estimated_hours"] for g in groups if g["status"] != "completed")
    done = [g for g in groups if g["status"] == "completed"]
    actual = sum(g["actual_elapsed_hours"] or 0 for g in done)
    missing = [g["group"] for g in done if g["actual_elapsed_hours"] is None]
    remaining_upper = remaining*1.5 + reserve_hours
    total = actual + benchmark_hours + remaining_upper
    return {"estimated_processing_hours": remaining, "estimated_remaining_upper_hours": remaining_upper,
            "known_completed_hours": actual, "completed_groups": [g["group"] for g in done],
            "missing_completed_timing": missing, "benchmark_hours": benchmark_hours,
            "reserve_hours": reserve_hours, "uncertainty_multiplier": 1.5,
            "estimated_upper_hours": total, "target_hours": target_hours,
            "within_budget_estimate": total <= target_hours if not missing else None}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch-root", type=Path, required=True)
    p.add_argument("--clip-model", default="openai/clip-vit-large-patch14")
    p.add_argument("--clip-batch-size", type=int, default=64)
    p.add_argument("--ocr-batch-size", type=int, default=32)
    p.add_argument("--interval", type=float, default=1.0)
    p.add_argument("--ocr-enhance", type=int, choices=(0, 1), default=0)
    p.add_argument("--target-hours", type=float, default=30)
    p.add_argument("--archives", nargs="+", choices=ARCHIVES, default=ARCHIVES)
    p.add_argument("--speed-profile", choices=("legacy", "a100-fast"), default="a100-fast")
    p.add_argument("--reserve-hours", type=float, default=4)
    args = p.parse_args()
    if args.target_hours <= 0 or args.reserve_hours < 0:
        p.error("Invalid time budget")
    args.batch_root = args.batch_root.resolve()
    started = time.monotonic()
    inventory, completed = {}, []
    for name in args.archives:
        paths = loose_videos(args.batch_root / "videos", name)
        with ThreadPoolExecutor(max_workers=4) as pool:
            inventory[name] = list(progress_items(pool.map(duration, paths),
                f"Probe durations: {name}", "video", total=len(paths)))
        result = completed_measurement(args, name, inventory[name])
        if result:
            completed.append(result)
            print(f"COMPLETE {name}; remaining work = 0; actual hours = {result['actual_elapsed_hours']}", flush=True)
    completed_names = {r["group"] for r in completed}
    groups = list(completed)
    pending = [(name, items) for name, items in inventory.items() if name not in completed_names]
    for name, items in progress_items(pending, "Estimate remaining groups", "group"):
        fit = remaining_from_actual(name, sum(d for _, d in items),
            [r for r in completed if r.get("speed_profile", "legacy") == speed_profile(args)])
        if fit is None:
            fit = sample_group(args, name, items)
        groups.append({"group": name, "videos": len(items), "source_hours": sum(d for _, d in items)/3600,
                       "status": "pending", **fit})
        print(f"{name}: {fit['estimated_hours']:.2f} h, method={fit['method']}", flush=True)
    report = {"report_version": 3, "scope": "visual_only_asr_delegated", "groups": groups,
              "configuration": {k: v for k, v in vars(args).items() if k != "batch_root"},
              **summarize(groups, (time.monotonic()-started)/3600, args.reserve_hours, args.target_hours),
              "caveats": ["Assumes comparable GPU, settings, content and Drive throughput.",
                          "M and N calibrated separately; samples cannot guarantee a deadline.",
                          "Previous benchmarks, retries, queueing and downtime are not included.",
                          "Transfer/local merge reserve is an unmeasured assumption."]}
    out = args.batch_root / "processed_data"
    out.mkdir(exist_ok=True)
    previous = out / "budget_report.json"
    if previous.is_file():
        write_json(out / f"budget_report_previous_{time.time_ns()}.json", json.loads(previous.read_text()))
    write_json(previous, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
