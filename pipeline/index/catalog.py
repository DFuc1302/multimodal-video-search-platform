#!/usr/bin/env python3
"""Build a portable video -> storage catalog from independent dataset roots."""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from dataset_layout import load_dataset_config


def build_catalog(config_path: Path, output: Path) -> dict:
    config = load_dataset_config(config_path)
    videos: dict[str, dict] = {}
    offline: dict[str, str] = {}
    for shard in config.select():
        try:
            shard.preflight(write=False)
            if not shard.meta_db.is_file():
                raise FileNotFoundError(f"Chưa có meta.db: {shard.meta_db}")
            with sqlite3.connect(
                f"file:{shard.meta_db.resolve()}?mode=ro", uri=True
            ) as db:
                rows = db.execute(
                    "WITH normalized AS ("
                    " SELECT replace(file, char(92), '/') AS name FROM frames"
                    "), videos AS ("
                    " SELECT CASE WHEN instr(name, '/') > 0"
                    " THEN substr(name, 1, instr(name, '/') - 1) ELSE name END AS video"
                    " FROM normalized"
                    ") SELECT video, COUNT(*) FROM videos WHERE video != '' GROUP BY video"
                ).fetchall()
            counts: dict[str, int] = {}
            for raw_video, frame_count in rows:
                parts = PurePosixPath(str(raw_video).replace("\\", "/")).parts
                if not parts or not parts[0]:
                    continue
                video = parts[0]
                counts[video] = counts.get(video, 0) + int(frame_count)
            for video, frame_count in sorted(counts.items()):
                if video in videos:
                    videos[video].setdefault("duplicates", []).append(shard.shard_id)
                    continue
                videos[video] = {
                    "storage_id": shard.shard_id,
                    "data_root": str(shard.data_root),
                    "frame_count": frame_count,
                }
        except Exception as exc:
            offline[shard.shard_id] = str(exc)
    payload = {
        "format_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(config.path),
        "video_count": len(videos),
        "videos": videos,
        "offline": offline,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        payload = build_catalog(args.config.resolve(), args.output.resolve())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print(
        f"Storage catalog: {payload['video_count']} video(s), "
        f"{len(payload['offline'])} offline root(s) -> {args.output.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
