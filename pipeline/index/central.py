#!/usr/bin/env python3
"""Build one local search index while leaving large assets on source storage."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from merge_processed_data import build_parser, merge


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCES = [
    Path("/mnt/d/AIC/data/processed/PHUC"),
    Path("/mnt/d/AIC/data/processed/KHOI"),
    Path("/mnt/d/AIC/data/processed/KHOA_NHO"),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "search_index",
    )
    parser.add_argument("--source", action="append", type=Path)
    args = parser.parse_args()
    sources = args.source or DEFAULT_SOURCES
    merge_args = build_parser().parse_args([
        "--sources", *[str(path) for path in sources],
        "--output", str(args.output),
        "--asset-mode", "external",
        "--index-type", "flat",
        "--allow-incomplete-ocr",
        "--skip-clip-text-export",
    ])
    merge(merge_args)
    source_clip = PROJECT_ROOT / "data" / "clip_text"
    target_clip = args.output / "clip_text"
    if not target_clip.exists():
        if not source_clip.is_dir():
            raise RuntimeError(f"Missing local CLIP text artifact: {source_clip}")
        shutil.copytree(source_clip, target_clip)
    from prepare_serve_bundle import export_clip_text_artifact
    manifest_path = args.output / "serve_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    export_clip_text_artifact(args.output, str(manifest["clip_model"]))
    manifest["clip_text_artifact"] = "clip_text"
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
