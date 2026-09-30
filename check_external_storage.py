#!/usr/bin/env python3
"""Validate removable/local shard sources used by a symlinked serving bundle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dataset_layout import DatasetShard, StorageUnavailableError


MANIFEST_NAME = "external_storage.json"


def check_external_storage(data_root: Path) -> int:
    manifest = data_root / MANIFEST_NAME
    if not manifest.is_file():
        return 0
    try:
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StorageUnavailableError(f"Không đọc được {manifest}: {exc}") from exc
    sources = payload.get("sources") if isinstance(payload, dict) else None
    if not isinstance(sources, list) or not sources:
        raise StorageUnavailableError(f"Manifest external storage không hợp lệ: {manifest}")
    for raw in sources:
        if not isinstance(raw, dict):
            raise StorageUnavailableError(f"Source không hợp lệ trong {manifest}")
        shard = DatasetShard(
            shard_id=str(raw.get("id", "external")),
            data_root=Path(raw["data_root"]),
            videos_dir=Path(raw.get("videos_dir", raw["data_root"])),
            mount_root=Path(raw["mount_root"]) if raw.get("mount_root") else None,
            volume_marker=raw.get("volume_marker"),
            require_mount=bool(raw.get("require_mount", False)),
        )
        shard.preflight(write=False)
        for relative in ("keyframes", "map-keyframes"):
            if not (shard.data_root / relative).is_dir():
                raise StorageUnavailableError(
                    f"Shard {shard.shard_id} thiếu {relative}: {shard.data_root / relative}"
                )
    return len(sources)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    args = parser.parse_args()
    try:
        count = check_external_storage(args.data_root.resolve())
    except (KeyError, StorageUnavailableError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    if count:
        print(f"External storage OK: {count} shard(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
