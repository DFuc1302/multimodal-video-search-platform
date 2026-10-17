#!/usr/bin/env python3
"""Cắt một frame mỗi 0,5 giây và tạo map-keyframes chính xác.

Ảnh được chọn trực tiếp theo frame nguồn (0-based). Vì vậy frame_idx trong CSV
là frame FFmpeg thực sự đã giải mã, không phải giá trị ước lượng sau khi cắt.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
import tempfile
from fractions import Fraction
from pathlib import Path

from video_manifest import VideoManifestError, filter_assigned_videos


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".flv", ".webm", ".m4v"}

FPS_2644_VIDEO_IDS = frozenset({"L24_V044"})

FPS_2997_VIDEO_IDS = frozenset(
    {
        "L25_V004",
        "L25_V005",
        "L25_V008",
        "L25_V013",
        "L25_V014",
        "L25_V017",
        "L25_V021",
        "L25_V022",
        "L25_V025",
        "L25_V030",
        "L25_V031",
        "L25_V034",
        "L25_V039",
        "L25_V040",
        "L25_V043",
        "L25_V048",
        "L25_V049",
        "L25_V052",
        "L25_V057",
        "L25_V058",
        "L25_V061",
        "L25_V066",
        "L25_V067",
        "L25_V070",
        "L25_V074",
        "L25_V077",
        "L25_V078",
        "L25_V084",
        "L25_V085",
        "L25_V088",
    }
)

FPS_30_VIDEO_IDS = frozenset(
    {
        "L21_V001",
        "L21_V002",
        "L21_V005",
        "L21_V006",
        "L21_V007",
        "L21_V012",
        "L21_V013",
        "L21_V014",
        "L21_V015",
        "L21_V016",
        "L21_V019",
        "L21_V021",
        "L21_V022",
        "L21_V023",
        "L21_V026",
        "L21_V027",
        "L21_V028",
        "L21_V029",
        "L21_V030",
        "L22_V001",
        "L22_V004",
        "L22_V005",
        "L22_V006",
        "L22_V011",
        "L22_V012",
        "L22_V013",
        "L22_V014",
        "L22_V015",
        "L22_V018",
        "L22_V019",
        "L22_V020",
        "L22_V021",
        "L22_V022",
        "L22_V025",
        "L22_V026",
        "L22_V027",
        "L22_V028",
        "L22_V029",
        "L24_V017",
        "L24_V019",
        "L24_V020",
        "L24_V021",
        "L24_V022",
        "L24_V023",
        "L24_V024",
        "L24_V025",
        "L24_V027",
        "L24_V028",
        "L24_V029",
        "L24_V030",
        "L24_V031",
        "L24_V033",
        "L24_V035",
        "L24_V036",
        "L24_V037",
        "L24_V038",
        "L24_V039",
        "L24_V040",
        "L24_V041",
        "L24_V042",
        "L24_V045",
    }
)


class OutputExistsError(RuntimeError):
    """Output đã tồn tại và người dùng chưa yêu cầu ghi đè."""


def fps_for_video(video_stem: str) -> Fraction:
    """Tra FPS theo mã video; 25 FPS là giá trị mặc định cho 781 video còn lại."""
    video_id = video_stem.upper()
    if video_id in FPS_2644_VIDEO_IDS:
        return Fraction("26.44")
    if video_id in FPS_2997_VIDEO_IDS:
        return Fraction("29.97")
    if video_id in FPS_30_VIDEO_IDS:
        return Fraction(30)
    return Fraction(25)
CSV_FIELDS = ["n", "pts_time", "fps", "frame_idx"]


def positive_fraction(value: str) -> Fraction:
    """Đọc số thập phân (29.97) hoặc phân số (30000/1001)."""
    try:
        result = Fraction(value)
    except (ValueError, ZeroDivisionError) as exc:
        raise argparse.ArgumentTypeError(
            f"Giá trị không hợp lệ: {value!r}; hãy dùng 25, 29.97 hoặc 30000/1001"
        ) from exc
    if result <= 0:
        raise argparse.ArgumentTypeError("Giá trị phải lớn hơn 0")
    return result


def frame_index_at(sample_number: int, interval: Fraction, fps: Fraction) -> int:
    """Trả về floor(sample_number * interval * fps), hoàn toàn bằng số nguyên."""
    if sample_number < 0:
        raise ValueError("sample_number phải >= 0")
    return sample_number * interval.numerator * fps.numerator // (
        interval.denominator * fps.denominator
    )


def build_frame_indices(
    output_count: int,
    interval: Fraction,
    fps: Fraction,
) -> list[int]:
    return [frame_index_at(i, interval, fps) for i in range(output_count)]


def build_frame_plan(
    duration: Fraction,
    interval: Fraction,
    fps: Fraction,
    *,
    total_frames: int | None = None,
) -> list[dict]:
    """Lập kế hoạch frame bằng Fraction, không tích lũy sai số thời gian."""
    duration = Fraction(duration)
    interval = Fraction(interval)
    fps = Fraction(fps)
    if duration < 0:
        raise ValueError("duration phải >= 0")
    if interval <= 0:
        raise ValueError("interval phải > 0")
    if fps <= 0:
        raise ValueError("fps phải > 0")
    if total_frames is not None and total_frames < 0:
        raise ValueError("total_frames phải >= 0")

    rows: list[dict] = []
    sample_number = 0
    while Fraction(sample_number) * interval < duration:
        frame_idx = frame_index_at(sample_number, interval, fps)
        if total_frames is not None and frame_idx >= total_frames:
            break
        rows.append(
            {
                "n": sample_number + 1,
                "pts_time": Fraction(frame_idx, 1) / fps,
                "fps": fps,
                "frame_idx": frame_idx,
            }
        )
        sample_number += 1
    return rows


def make_select_filter(interval: Fraction, fps: Fraction) -> str:
    """Chọn đúng dãy frame floor(k * interval * fps) ngay trong FFmpeg."""
    step = interval * fps
    if step < 1:
        raise ValueError(
            "interval * fps phải >= 1 để mỗi mốc ứng với một frame riêng"
        )
    numerator, denominator = step.numerator, step.denominator
    return (
        "select="
        f"eq(n\\,floor(ceil(n*{denominator}/{numerator}-1e-9)"
        f"*{numerator}/{denominator}+1e-9))"
    )


def decimal_text(value: Fraction, places: int = 7) -> str:
    text = f"{float(value):.{places}f}".rstrip("0").rstrip(".")
    return text if "." in text else f"{text}.0"


def read_fps_from_map(csv_path: Path) -> Fraction:
    """Đọc FPS chính xác từ CSV AIC, hoặc suy ra từ pts_time/frame_idx."""
    with csv_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        fields = set(reader.fieldnames or [])
    if not rows:
        raise ValueError(f"Map CSV rỗng: {csv_path}")

    if "fps" in fields:
        values: list[Fraction] = []
        for row in rows:
            raw = str(row.get("fps") or "").strip()
            try:
                value = Fraction(raw)
            except (ValueError, ZeroDivisionError) as exc:
                raise ValueError(f"FPS không hợp lệ trong {csv_path}: {raw!r}") from exc
            if value <= 0:
                raise ValueError(f"FPS phải lớn hơn 0 trong {csv_path}")
            values.append(value)
        if len(set(values)) != 1:
            raise ValueError(f"FPS không nhất quán trong {csv_path}")
        return values[0]

    if not {"pts_time", "frame_idx"} <= fields:
        raise ValueError(f"Không thể suy ra FPS từ {csv_path}")
    points: list[tuple[Fraction, int]] = []
    for row in rows:
        try:
            points.append((Fraction(str(row["pts_time"])), int(row["frame_idx"])))
        except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
            raise ValueError(f"Map CSV không hợp lệ: {csv_path}") from exc
    if len(points) < 2:
        raise ValueError(f"Cần ít nhất hai mốc để suy ra FPS: {csv_path}")
    base_time, base_idx = points[0]
    estimates = {
        Fraction(frame_idx - base_idx, 1) / (pts_time - base_time)
        for pts_time, frame_idx in points[1:]
        if pts_time > base_time
    }
    if len(estimates) != 1 or next(iter(estimates)) <= 0:
        raise ValueError(f"Không thể suy ra một FPS nhất quán từ {csv_path}")
    return estimates.pop()


def find_program(name: str) -> str:
    result = shutil.which(name)
    if result:
        return result
    raise RuntimeError(f"Không tìm thấy {name}. Hãy chạy: sudo apt install ffmpeg")


def probe_declared_fps(ffprobe: str, video: Path) -> Fraction | None:
    command = [
        ffprobe,
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=avg_frame_rate",
        "-of", "json",
        str(video),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        detail = result.stderr.strip() or "không có thông báo lỗi"
        raise RuntimeError(f"ffprobe không đọc được video: {detail}")

    try:
        streams = json.loads(result.stdout).get("streams") or []
        value = streams[0].get("avg_frame_rate") if streams else None
        return Fraction(value) if value and value != "0/0" else None
    except (json.JSONDecodeError, ValueError, ZeroDivisionError):
        return None


def numbered_images(directory: Path, stem: str) -> list[tuple[int, Path]]:
    pattern = re.compile(rf"^{re.escape(stem)}_(\d+)\.jpg$", re.IGNORECASE)
    images: list[tuple[int, Path]] = []
    for path in directory.glob(f"{stem}_*.jpg"):
        match = pattern.fullmatch(path.name)
        if match and path.is_file() and path.stat().st_size > 0:
            images.append((int(match.group(1)), path))
    return sorted(images)


def validate_numbering(images: list[tuple[int, Path]]) -> None:
    actual = [number for number, _ in images]
    if not images:
        raise RuntimeError("FFmpeg không xuất được frame nào")
    if actual != list(range(1, len(images) + 1)):
        raise RuntimeError("Tên frame FFmpeg xuất ra không liên tục từ 1")


def write_mapping(
    csv_path: Path,
    image_count: int,
    interval: Fraction,
    fps: Fraction,
) -> None:
    frame_plan = build_frame_indices(image_count, interval, fps)
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for n, frame_idx in enumerate(frame_plan, start=1):
            # pts_time là thời gian thật của frame nguồn theo FPS đã nhập.
            # Ví dụ mốc 0,5 s ở 25 FPS chọn frame 12, tức pts_time = 0,48 s.
            writer.writerow(
                {
                    "n": n,
                    "pts_time": decimal_text(Fraction(frame_idx, 1) / fps),
                    "fps": decimal_text(fps),
                    "frame_idx": frame_idx,
                }
            )


def csv_matches_plan(csv_path: Path, plan: list[dict]) -> bool:
    try:
        with csv_path.open(newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file))
        if len(rows) != len(plan):
            return False
        for position, (row, expected) in enumerate(zip(rows, plan), start=1):
            raw_n = str(row["n"]).strip()
            raw_idx = str(row["frame_idx"]).strip()
            if int(raw_n) != position or int(raw_idx) != int(expected["frame_idx"]):
                return False
        return True
    except (
        OSError,
        csv.Error,
        KeyError,
        TypeError,
        ValueError,
    ):
        return False


def output_is_complete(
    video: Path,
    frames_dir: Path,
    csv_path: Path,
    plan: list[dict],
    *,
    digits: int,
) -> bool:
    expected_names = {
        f"{video.stem}_{position:0{digits}d}.jpg"
        for position in range(1, len(plan) + 1)
    }
    actual_names = {
        path.name
        for path in frames_dir.glob(f"{video.stem}_*.jpg")
        if path.is_file() and path.stat().st_size > 0
    }
    return actual_names == expected_names and csv_matches_plan(csv_path, plan)


def ffmpeg_command(
    ffmpeg: str,
    video: Path,
    output_pattern: Path,
    select_filter: str,
    *,
    expected_count: int,
    threads: int,
    gpu_decode: bool,
    quality: int = 2,
) -> list[str]:
    if expected_count < 0:
        raise ValueError("expected_count phải >= 0")
    command = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y"]
    if gpu_decode:
        command.extend(["-hwaccel", "cuda"])
    command.extend(
        [
            "-threads", str(threads),
            "-i", str(video),
            "-map", "0:v:0",
            "-an",
            "-vf", select_filter,
            "-fps_mode", "vfr",
            "-frames:v", str(expected_count),
            "-start_number", "1",
            "-q:v", str(quality),
            str(output_pattern),
        ]
    )
    return command


def extract_video(
    video: Path,
    fps: Fraction,
    interval: Fraction,
    keyframes_root: Path,
    maps_root: Path,
    digits: int,
    quality: int,
    threads: int,
    force: bool,
) -> tuple[Path, Path, int]:
    if not video.is_file():
        raise RuntimeError(f"Video không tồn tại: {video}")
    if video.suffix.lower() not in VIDEO_EXTENSIONS:
        raise RuntimeError(f"Định dạng video chưa được hỗ trợ: {video.suffix}")
    if digits < 1:
        raise RuntimeError("--digits phải >= 1")
    if not 2 <= quality <= 31:
        raise RuntimeError("--quality phải nằm trong khoảng 2..31")
    if threads < 1:
        raise RuntimeError("--threads phải >= 1")

    keyframes_root.mkdir(parents=True, exist_ok=True)
    maps_root.mkdir(parents=True, exist_ok=True)
    frame_dir = keyframes_root / video.stem
    csv_path = maps_root / f"{video.stem}.csv"

    old_images = numbered_images(frame_dir, video.stem) if frame_dir.is_dir() else []
    if (old_images or csv_path.exists()) and not force:
        raise OutputExistsError(
            f"Output của {video.stem} đã tồn tại và --skip-existing đang bật."
        )

    ffmpeg = find_program("ffmpeg")
    ffprobe = find_program("ffprobe")
    select_filter = make_select_filter(interval, fps)
    declared_fps = probe_declared_fps(ffprobe, video)
    if declared_fps is not None and abs(float(declared_fps - fps)) > 0.02:
        print(
            "[CẢNH BÁO] FPS truyền vào "
            f"({decimal_text(fps)}) khác FPS video khai báo "
            f"({decimal_text(declared_fps)}). Script vẫn dùng FPS truyền vào.",
            file=sys.stderr,
        )

    with tempfile.TemporaryDirectory(
        prefix=f".{video.stem}_extract_", dir=keyframes_root
    ) as temp_name:
        temp_dir = Path(temp_name)
        output_pattern = temp_dir / f"{video.stem}_%0{digits}d.jpg"
        command = [
            ffmpeg,
            "-hide_banner", "-loglevel", "error", "-y",
            "-threads", str(threads),
            "-i", str(video),
            "-map", "0:v:0",
            "-an",
            "-vf", select_filter,
            "-fps_mode", "vfr",
            "-start_number", "1",
            "-q:v", str(quality),
            str(output_pattern),
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            detail = result.stderr.strip() or "không có thông báo lỗi"
            raise RuntimeError(f"FFmpeg cắt frame thất bại:\n{detail}")

        images = numbered_images(temp_dir, video.stem)
        validate_numbering(images)
        temp_csv = temp_dir / f"{video.stem}.csv"
        write_mapping(temp_csv, len(images), interval, fps)

        # Chỉ thay output sau khi FFmpeg và CSV đã hoàn tất.
        frame_dir.mkdir(parents=True, exist_ok=True)
        if force:
            for _, old_image in numbered_images(frame_dir, video.stem):
                old_image.unlink()
            csv_path.unlink(missing_ok=True)

        for _, image in images:
            image.replace(frame_dir / image.name)
        temp_csv.replace(csv_path)

    return frame_dir, csv_path, len(images)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Cắt toàn bộ video trong một folder mỗi 0,5 giây, tự tra FPS theo "
            "mã video và tạo CSV n,pts_time,fps,frame_idx."
        )
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        type=Path,
        help="Folder chứa toàn bộ video đầu vào",
    )
    parser.add_argument(
        "--interval",
        type=positive_fraction,
        default=Fraction(1, 2),
        help="Khoảng cắt theo giây (mặc định: 0.5)",
    )
    parser.add_argument(
        "--keyframes-dir",
        type=Path,
        default=Path("keyframes"),
        help="Folder gốc chứa keyframes/<tên_video>",
    )
    parser.add_argument(
        "--map-dir",
        type=Path,
        default=Path("map-keyframes"),
        help="Folder chứa <tên_video>.csv",
    )
    parser.add_argument("--digits", type=int, default=4, help="Số chữ số tên ảnh")
    parser.add_argument(
        "--quality",
        type=int,
        default=2,
        help="Chất lượng JPEG: 2 tốt nhất, 31 thấp nhất",
    )
    parser.add_argument("--threads", type=int, default=8, help="Số CPU thread mỗi FFmpeg")
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Bỏ qua video đã có output thay vì cắt lại",
    )
    parser.add_argument(
        "--video-list",
        type=Path,
        help="Manifest JSON gán đường dẫn video tương đối cho từng shard",
    )
    parser.add_argument(
        "--shard-id",
        help="Shard cần chọn trong --video-list",
    )
    return parser


def find_video_files(
    input_dir: Path,
    video_list: Path | None = None,
    shard_id: str | None = None,
) -> list[Path]:
    candidates = sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )
    return filter_assigned_videos(candidates, input_dir, video_list, shard_id)


def main() -> int:
    args = build_parser().parse_args()
    if not args.input_dir.is_dir():
        print(f"[LỖI] Folder video không tồn tại: {args.input_dir}", file=sys.stderr)
        return 1

    try:
        video_files = find_video_files(
            args.input_dir,
            video_list=args.video_list,
            shard_id=args.shard_id,
        )
    except VideoManifestError as exc:
        print(f"[LỖI] {exc}", file=sys.stderr)
        return 1
    if not video_files:
        print(f"[LỖI] Không tìm thấy video trong: {args.input_dir}", file=sys.stderr)
        return 1

    fps_counts: dict[Fraction, int] = {}
    for video in video_files:
        fps = fps_for_video(video.stem)
        fps_counts[fps] = fps_counts.get(fps, 0) + 1

    print(f"Tìm thấy {len(video_files)} video trong {args.input_dir}")
    print(
        "Phân bố FPS: "
        + " | ".join(
            f"{decimal_text(fps)} FPS: {fps_counts.get(fps, 0)}"
            for fps in (
                Fraction(25),
                Fraction("26.44"),
                Fraction("29.97"),
                Fraction(30),
            )
        )
    )

    stats = {"processed": 0, "skipped": 0, "failed": 0, "frames": 0}
    for index, video in enumerate(video_files, start=1):
        fps = fps_for_video(video.stem)
        print(
            f"\n[{index}/{len(video_files)}] {video.name} | "
            f"FPS {decimal_text(fps)}"
        )
        try:
            frame_dir, csv_path, count = extract_video(
                video=video,
                fps=fps,
                interval=args.interval,
                keyframes_root=args.keyframes_dir,
                maps_root=args.map_dir,
                digits=args.digits,
                quality=args.quality,
                threads=args.threads,
                force=not args.skip_existing,
            )
        except OutputExistsError as exc:
            stats["skipped"] += 1
            print(f"[BỎ QUA] {exc}")
            continue
        except (OSError, RuntimeError, ValueError) as exc:
            stats["failed"] += 1
            print(f"[LỖI] {video.name}: {exc}", file=sys.stderr)
            continue

        stats["processed"] += 1
        stats["frames"] += count
        print(f"[XONG] {count} frame")
        print(f"       Keyframes: {frame_dir}")
        print(f"       Mapping:   {csv_path}")

    print(
        "\nHOÀN TẤT | "
        f"đã cắt: {stats['processed']} | "
        f"bỏ qua: {stats['skipped']} | "
        f"lỗi: {stats['failed']} | "
        f"tổng frame mới: {stats['frames']}"
    )
    print("frame_idx trong CSV dùng hệ 0-based")
    return 1 if stats["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
