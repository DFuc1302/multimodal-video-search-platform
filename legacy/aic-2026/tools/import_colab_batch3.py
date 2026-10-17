"""Verify downloaded Batch 3 ZIPs, unpack, and merge indices without GPU inference."""
import argparse
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import zipfile

from colab_batch3 import ARCHIVES, digest, write_json
from external_assets import video_id_from_path


def unpack(archive, destination):
    checksum_file = archive.with_suffix(".sha256.json")
    expected = json.loads(checksum_file.read_text())["sha256"]
    if digest(archive) != expected:
        raise RuntimeError(f"Download checksum mismatch: {archive}")
    marker = destination / ".import_sha256.json"
    if destination.exists():
        if not marker.is_file() or json.loads(marker.read_text())["sha256"] != expected:
            raise RuntimeError(f"Existing destination does not match ZIP: {destination}")
        return
    with tempfile.TemporaryDirectory(prefix=".batch3-import-", dir=destination.parent) as temp:
        root = Path(temp) / "data"
        root.mkdir()
        with zipfile.ZipFile(archive) as z:
            for item in z.infolist():
                name = PurePosixPath(item.filename.replace("\\", "/"))
                if (name.is_absolute() or ".." in name.parts or ":" in item.filename
                        or (item.external_attr >> 16) & 0o170000 == 0o120000):
                    raise RuntimeError(f"Unsafe ZIP member: {item.filename}")
                if item.is_dir():
                    continue
                target = root.joinpath(*name.parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                with z.open(item) as src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst, 4 * 1024**2)
        write_json(root / ".import_sha256.json", {"sha256": expected})
        root.replace(destination)


def build_manifest(sources):
    records, owners = [], {}
    for i, root in enumerate(sources):
        external = root / "external_storage.json"
        payload = json.loads(external.read_text()) if external.exists() else None
        if payload:
            mapping = {r["id"]: f"s{i}_{r['id']}" for r in payload["sources"]}
            records.extend({**r, "id": mapping[r["id"]]} for r in payload["sources"])
        else:
            records.append({"id": f"s{i}", "data_root": str(root),
                            "keyframes_dir": str((root / "keyframes").resolve()),
                            "map_dir": str((root / "map-keyframes").resolve()),
                            "videos_dir": str(root / "videos"), "require_mount": False})
        with sqlite3.connect((root / "meta.db").as_uri() + "?mode=ro", uri=True) as db:
            ids = {video_id_from_path(row[0]).casefold() for row in db.execute("SELECT file FROM frames")}
        if ids & owners.keys():
            raise RuntimeError(f"Overlapping video IDs; no merge performed: {sorted(ids & owners.keys())[:20]}")
        old_owners = {k.casefold(): v for k, v in payload["video_sources"].items()} if payload else {}
        for video in ids:
            owners[video] = mapping[old_owners[video]] if payload else f"s{i}"
    return {"format_version": 3, "asset_mode": "external", "sources": records, "video_sources": owners}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--downloads", type=Path, required=True)
    p.add_argument("--shards", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--old-index", type=Path)
    args = p.parse_args()
    args.shards = args.shards.resolve()
    args.output = args.output.resolve()
    if args.output.exists():
        raise RuntimeError("Output already exists; choose a new output directory")
    args.shards.mkdir(parents=True, exist_ok=True)
    sources = [args.old_index.resolve()] if args.old_index else []
    # Require ALL 11 downloads before any extraction or merge.
    archives = [args.downloads / (Path(name).stem + "_processed.zip") for name in ARCHIVES]
    for archive in archives:
        if not archive.is_file() or not archive.with_suffix(".sha256.json").is_file():
            raise FileNotFoundError(f"Missing ZIP/checksum: {archive}")
    from run_new_dataset_pipeline import verify_bundle
    for archive in archives:
        destination = args.shards / archive.stem
        unpack(archive, destination)
        verify_bundle(destination)
        sources.append(destination)
    manifest = args.shards / "batch3_external_storage.json"
    write_json(manifest, build_manifest(sources))
    env = {**os.environ, "BACKEND_MODE": "sqlite", "STORAGE_BACKEND": "sqlite"}
    subprocess.run([sys.executable, str(Path(__file__).with_name("merge_processed_data.py")),
                    "--sources", *map(str, sources), "--output", str(args.output),
                    "--asset-mode", "external", "--external-storage-manifest", str(manifest)],
                   env=env, check=True)
    verify_bundle(args.output)
    print(f"READY: {args.output}\nKeep the shards directory: image/map files are served from it.")


if __name__ == "__main__":
    main()
