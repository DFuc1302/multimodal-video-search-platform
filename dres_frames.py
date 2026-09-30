"""Strict, read-only frame resolution for competition submissions.

Unlike display timelines, this module never infers a timestamp or substitutes
the ordinal of a sampled image for an original-video frame index.
"""
from __future__ import annotations

import csv
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
import re


class FrameError(ValueError):
    pass


def video_id(value):
    value = str(value).strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,119}", value):
        raise FrameError("Video ID chỉ gồm chữ, số, _ hoặc -; không nhập đường dẫn hay .mp4.")
    return value


def integer(value, name="Frame ID"):
    try:
        if isinstance(value, bool):
            raise ValueError()
        number = Decimal(str(value))
        if not number.is_finite() or number < 0 or number != number.to_integral_value():
            raise ValueError()
        return int(number)
    except (ValueError, InvalidOperation):
        raise FrameError(f"{name} phải là số nguyên không âm.") from None


def read_map(path):
    """Return validated rows plus errors; preserve evidence, never repair data."""
    rows, errors = [], []
    seen_n, seen_frames = set(), set()
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not {"n", "frame_idx", "pts_time", "fps"}.issubset(reader.fieldnames or []):
            return [], ["Thiếu cột n, frame_idx, pts_time hoặc fps."]
        for line, row in enumerate(reader, 2):
            try:
                n, frame = integer(row["n"], "n"), integer(row["frame_idx"])
                pts, fps = Decimal(row["pts_time"]), Decimal(row["fps"])
                if not pts.is_finite() or pts < 0 or not fps.is_finite() or fps <= 0:
                    raise ValueError("pts_time/fps không hợp lệ")
                if n in seen_n or frame in seen_frames:
                    errors.append(f"Dòng {line}: n hoặc frame_idx bị trùng")
                seen_n.add(n)
                seen_frames.add(frame)
                rows.append({"n": n, "frame_idx": frame,
                             "pts_time": float(pts), "fps": float(fps),
                             "time_ms": int((pts * 1000).quantize(Decimal(1), rounding=ROUND_HALF_UP))})
            except (ValueError, InvalidOperation, TypeError) as exc:
                errors.append(f"Dòng {line}: {exc}")
    ordered = sorted(rows, key=lambda row: row["frame_idx"])
    if any(b["frame_idx"] > a["frame_idx"] and b["pts_time"] <= a["pts_time"] for a, b in zip(ordered, ordered[1:])):
        errors.append("Thời gian không tăng theo frame_idx.")
    if not rows:
        errors.append("Không có ánh xạ hợp lệ.")
    return rows, errors


class FrameResolver:
    def __init__(self, find_map):
        self.find_map = find_map

    def resolve(self, video, frames):
        video = video_id(video)
        if not isinstance(frames, list) or not 1 <= len(frames) <= 100:
            raise FrameError("Nhập từ 1 đến 100 Frame ID.")
        frames = [integer(frame) for frame in frames]
        if len(frames) != len(set(frames)):
            raise FrameError("Các Frame ID không được trùng nhau.")
        path = self.find_map(video)
        if path is None or not Path(path).is_file():
            raise FrameError(f"Không tìm thấy map-keyframes của {video}. Không thể xác minh đáp án.")
        rows, errors = read_map(path)
        fatal = [error for error in errors if 'n hoặc frame_idx bị trùng' not in error]
        if fatal:
            raise FrameError(f"Ánh xạ của {video} có lỗi: " + "; ".join(fatal[:3]))
        by_frame = {row["frame_idx"]: row for row in rows}
        for frame in frames:
            times = {row['pts_time'] for row in rows if row['frame_idx'] == frame}
            if len(times) > 1:
                raise FrameError(f"Frame {frame} có nhiều pts_time khác nhau trong CSV. Chọn frame khác đã xác minh.")
        missing = [frame for frame in frames if frame not in by_frame]
        if missing:
            nearest = sorted(by_frame, key=lambda frame: abs(frame - missing[0]))[:4]
            raise FrameError(f"Frame {missing[0]} chưa có trong bảng ánh xạ. Frame gốc gần nhất: "
                             + ", ".join(map(str, sorted(nearest)))
                             + ". Không dùng số thứ tự ảnh n và không tự nội suy thời gian.")
        return {"video_id": Path(path).stem, "source": str(path), 'map_warnings': errors[:3],
                "frames": [dict(by_frame[frame]) for frame in frames]}


def build_answer(resolver, task_type, video, frames, answer=""):
    if task_type not in {"textual_kis", "video_kis", "qa", "trake"}:
        raise FrameError("Chọn Textual KIS, Video KIS, Q&A hoặc TRAKE.")
    mapping = resolver.resolve(video, frames)
    resolved = mapping["frames"]
    video = mapping["video_id"]
    if task_type != "trake" and len(resolved) != 1:
        raise FrameError("KIS và Q&A chỉ nộp một frame mỗi lần.")
    if task_type == "trake":
        if any(b["frame_idx"] <= a["frame_idx"] for a, b in zip(resolved, resolved[1:])):
            raise FrameError("TRAKE cần frame tăng dần, theo thứ tự các sự kiện trong đề.")
        item = {"text": "TR-" + video + "-" + ",".join(str(row["frame_idx"]) for row in resolved)}
    elif task_type == "qa":
        answer = str(answer).strip()
        if not answer or len(answer) > 2000 or any(ord(c) < 32 for c in answer):
            raise FrameError("Nhập câu trả lời Q&A trên một dòng (tối đa 2.000 ký tự).")
        item = {"text": f"QA-{answer}-{video}-{resolved[0]['time_ms']}"}
    else:
        item = {"mediaItemName": video, "start": resolved[0]["time_ms"], "end": resolved[0]["time_ms"]}
    return {"task_type": task_type, "mapping": mapping,
            "payload": {"answerSets": [{"answers": [item]}]},
            "warnings": (["Câu trả lời chứa dấu '-'; kiểm tra quy tắc phân cách QA với BTC."]
                         if task_type == "qa" and "-" in answer else [])}
