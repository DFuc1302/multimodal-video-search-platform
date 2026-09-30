#!/usr/bin/env python3
"""Plan source-frame deduplication, filter the active bundle, then switch the UI.

Run offline under WSL. Without --apply only the resumable plan is written.
The original aggregate index is retained; source JPEG deletion is permanent.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import sqlite3
import subprocess
import time


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(path)


def log(message):
    print(message, flush=True)


def readonly(path):
    return sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)


def fingerprint(path):
    st = path.stat()
    return [st.st_size, st.st_mtime_ns]


def source_signature(source):
    names = ("meta.db", "meta.db-wal", "faiss_index/frames.faiss",
             "faiss_index/frames_meta.json", "external_storage.json", "active_summary.json")
    return {name: fingerprint(source / name) for name in names if (source / name).exists()}


def removal_indices(hashes, threshold):
    # Adjacent-pair chains, identical to dedup_keyframes.py: retain first only.
    return [i for i in range(1, len(hashes))
            if hashes[i - 1] is not None and hashes[i] is not None
            and hashes[i - 1] - hashes[i] <= threshold]


def make_plan(root, work, device, workers, batch_size, threshold):
    from dedup_keyframes import GPUHasher, frame_number
    import imagehash
    from PIL import Image
    from tqdm import tqdm
    plans = work / "plans"
    plans.mkdir(exist_ok=True)
    folders = sorted({p.parent for p in root.rglob("*.jpg")
                      if not any(x.startswith(".extract-") for x in p.parts)})
    if not folders:
        raise RuntimeError(f"No JPEG frames: {root}")
    gpu = GPUHasher(workers, batch_size) if device == "cuda" else None
    result = []
    try:
        for folder in tqdm(folders, desc="Plan dedupe", unit="video"):
            relative = folder.relative_to(root).as_posix()
            path = plans / (hashlib.sha256(relative.encode()).hexdigest() + ".json")
            if path.exists():
                result.append(read_json(path))
                continue
            files = sorted(folder.glob("*.jpg"), key=frame_number)
            if any(not p.resolve().is_relative_to(root) for p in files):
                raise RuntimeError(f"Symlink escapes input root: {folder}")
            before = {p.name: fingerprint(p) for p in files}
            if gpu:
                hashes, errors = gpu.hashes(files, "phash")
                if errors:
                    raise RuntimeError(f"Unreadable images in {folder}: {errors[:5]}")
            else:
                hashes = []
                for p in files:
                    with Image.open(p) as image:
                        hashes.append(imagehash.phash(image))
            if before != {p.name: fingerprint(p) for p in files}:
                raise RuntimeError(f"Images changed during hashing: {folder}")
            plan = {"folder": relative, "files": before,
                    "remove": [files[i].name for i in removal_indices(hashes, threshold)]}
            write_json(path, plan)
            result.append(plan)
    finally:
        if gpu:
            gpu.close()
    return result


def validate_images(root, plans, allow_deleted=False):
    if plans and plans[0].get("mode") == "existing":
        if existing_plan(root) != plans:
            raise RuntimeError("Image inventory changed. Stop extraction/deletion jobs and use a new work directory")
        return
    expected_folders = {p["folder"] for p in plans}
    observed_folders = {p.parent.relative_to(root).as_posix() for p in root.rglob("*.jpg")
                        if not any(x.startswith(".extract-") for x in p.parts)}
    if expected_folders != observed_folders:
        raise RuntimeError("Video folders changed since planning; stop other extraction jobs")
    for plan in plans:
        folder = root / plan["folder"]
        observed = {p.name: fingerprint(p) for p in folder.glob("*.jpg")}
        expected = plan["files"]
        allowed = set(plan["remove"]) if allow_deleted else set()
        if set(observed) - set(expected) or set(expected) - set(observed) - allowed:
            raise RuntimeError(f"Image inventory changed: {folder}")
        if any(value != expected[name] for name, value in observed.items()):
            raise RuntimeError(f"Image content/mtime changed: {folder}")


def existing_plan(root):
    """Directory enumeration only: no image opens, hashes or per-image stat/resolve."""
    result = []
    def fail(error):
        raise error
    for folder, directories, files in os.walk(root, onerror=fail, followlinks=False):
        directories[:] = sorted(d for d in directories if not d.startswith(".extract-"))
        path = Path(folder)
        # Reject symlinked directories rather than mistaking unvisited media for deletions.
        if path.is_symlink() or any((path / d).is_symlink() for d in directories):
            raise RuntimeError(f"Symlinked image directory is not supported: {path}")
        if path == root:
            continue
        result.append({"folder": path.relative_to(root).as_posix(), "mode": "existing",
                       "files": sorted(f for f in files if f.lower().endswith(".jpg")), "remove": []})
    if not result or not any(p["files"] for p in result):
        raise RuntimeError("No source images found; refusing to treat unavailable storage as deleted frames")
    return sorted(result, key=lambda p: p["folder"])


def removal_predicate(source, root, plans):
    from external_assets import ExternalAssetResolver
    resolver = ExternalAssetResolver.load(source)
    if not resolver:
        raise RuntimeError("This workflow requires external_storage.json in the aggregate index")
    # Resolve each storage ROOT once, not ~1 million individual image paths over DrvFS.
    for record in resolver.records.values():
        record["keyframes_dir"] = str(Path(record.get("keyframes_dir", Path(record["data_root"]) / "keyframes")).resolve())
    existing = bool(plans and plans[0].get("mode") == "existing")
    inventories = {p["folder"]: set(p["files"]) for p in plans} if existing else {}
    removed = {(root / p["folder"] / name)
               for p in plans for name in p["remove"]}
    def predicate(filename):
        resolved = resolver.resolve_image(filename)
        if resolved is None or not resolved.is_relative_to(root):
            return False
        if not existing:
            return resolved in removed
        relative = resolved.relative_to(root)
        folder = relative.parent.as_posix()
        if folder not in inventories:
            raise RuntimeError(f"Source directory absent, cannot infer deleted frames: {root / folder}")
        return relative.name not in inventories[folder]
    return predicate


def selected_files(source, root, plans):
    predicate = removal_predicate(source, root, plans)
    selected = []
    with readonly(source / "meta.db") as db:
        for fid, filename, vector_id in db.execute("SELECT id,file,faiss_id FROM frames"):
            if predicate(filename):
                selected.append([fid, filename, vector_id])
    return selected


def filter_map(path, numbers, *, keep=False):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        rows = list(reader)
    if not fields or "n" not in fields:
        raise RuntimeError(f"Invalid frame map: {path}")
    kept = [r for r in rows if (int(r["n"]) in numbers) == keep]
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(kept)
    temporary.replace(path)


def video_removals(plans):
    result = {}
    for p in plans:
        video = Path(p["folder"]).name.casefold()
        if video in result:
            raise RuntimeError(f"Ambiguous duplicate video directory: {video}")
        filenames = p["files"] if p.get("mode") == "existing" else p["remove"]
        result[video] = {int(Path(name).stem.rsplit("_", 1)[-1]) for name in filenames}
    return result


def build_filtered(source, target, root, plans, selected, faiss_mode="preserve"):
    """Only mutate a pipeline-owned output, never the active source index."""
    import faiss
    import numpy as np
    from prepare_serve_bundle import prepare_dataset
    from tqdm import tqdm

    target.mkdir(parents=True, exist_ok=True)
    # SQLite backup includes committed WAL data and is safe for a clean restart.
    with readonly(source / "meta.db") as original, sqlite3.connect(target / "meta.db") as db:
        original.backup(db)
        db.executemany("DELETE FROM frames WHERE id=?", [(r[0],) for r in selected])
        db.commit()
    for name in ("asr_cache.json",):
        if (source / name).exists():
            shutil.copy2(source / name, target / name)
    for name in ("clip_text", "keyframes"):
        dest = target / name
        if not dest.exists():
            dest.symlink_to(source / name, target_is_directory=True)

    # Copy every map source, so summary/ASR timelines use the filtered maps too.
    manifest = read_json(source / "external_storage.json")
    remap = {}
    numbers = video_removals(plans)
    for i, record in enumerate(manifest["sources"]):
        old_map = Path(record.get("map_dir", Path(record["data_root"]) / "map-keyframes"))
        new_map = target / "map-keyframes" / f"source_{i}"
        if not old_map.is_dir():
            raise RuntimeError(f"Missing map source: {old_map}")
        shutil.copytree(old_map, new_map, dirs_exist_ok=True)
        remap[str(old_map)] = str(new_map)
        record["map_dir"] = str(new_map)
        image_root = Path(record.get("keyframes_dir", Path(record["data_root"]) / "keyframes"))
        if image_root.resolve() == root:
            for path in new_map.rglob("*.csv"):
                if path.stem.casefold() in numbers:
                    filter_map(path, numbers[path.stem.casefold()], keep=plans[0].get("mode") == "existing")
    write_json(target / "external_storage.json", manifest)

    summary = source / "summary_index"
    if (source / "active_summary.json").is_file():
        summary = (source / read_json(source / "active_summary.json")["release"]).resolve()
        if not summary.is_relative_to((source / "summary_releases").resolve()):
            raise RuntimeError("Invalid summary release")
    if summary.is_dir():
        shutil.copytree(summary, target / "summary_index", dirs_exist_ok=True)
        summary_db = summary / "summaries.db"
        if summary_db.exists():
            with readonly(summary_db) as original, sqlite3.connect(target / "summary_index" / "summaries.db") as db:
                original.backup(db)
                # Summary search maintains its own frame locator, separate from meta.db.
                predicate = removal_predicate(source, root, plans)
                tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
                if "frames" in tables:
                    delete_files = []
                    for (filename,) in db.execute("SELECT file FROM frames"):
                        if predicate(filename):
                            delete_files.append((filename,))
                    db.executemany("DELETE FROM frames WHERE file=?", delete_files)
                    db.commit()
        summary_manifest = target / "summary_index" / "manifest.json"
        if summary_manifest.is_file():
            rows = read_json(summary_manifest)
            if isinstance(rows, list):
                for row in rows:
                    old = row.get("map_csv", "")
                    for before, after in remap.items():
                        if old.startswith(before + "/"):
                            row["map_csv"] = after + old[len(before):]
                            break
                write_json(summary_manifest, rows)

    log("Rebuilding FAISS from existing vectors (no CLIP inference)...")
    index = faiss.read_index(str(source / "faiss_index" / "frames.faiss"))
    if not isinstance(index, faiss.IndexIDMap2):
        raise RuntimeError("Expected FAISS IndexIDMap2")
    base = faiss.downcast_index(index.index)
    if not isinstance(base, (faiss.IndexFlat, faiss.IndexHNSWFlat)):
        raise RuntimeError(f"Unsupported FAISS base: {type(base).__name__}")
    ids = faiss.vector_to_array(index.id_map)
    with readonly(target / "meta.db") as db:
        kept_ids = np.array([r[0] for r in db.execute("SELECT faiss_id FROM frames")], dtype="int64")
    with readonly(source / "meta.db") as db:
        source_ids = np.array([r[0] for r in db.execute("SELECT faiss_id FROM frames")], dtype="int64")
    if not np.array_equal(np.sort(ids), np.sort(source_ids)):
        raise RuntimeError("Source FAISS/SQLite IDs disagree; repair source before dedup")
    # Clone/reset preserves Flat/HNSW type and search/construction parameters.
    if faiss_mode == "flat":
        rebuilt = faiss.IndexIDMap2(faiss.IndexFlat(index.d, index.metric_type))
    else:
        rebuilt = faiss.clone_index(index)
        rebuilt.reset()
    keep = np.isin(ids, kept_ids)
    for start in tqdm(range(0, len(ids), 4096), desc="Filter vectors", unit="batch"):
        mask = keep[start:start + 4096]
        if mask.any():
            vectors = base.reconstruct_n(start, len(mask))
            rebuilt.add_with_ids(np.ascontiguousarray(vectors[mask]), ids[start:start + 4096][mask])
    (target / "faiss_index").mkdir(exist_ok=True)
    faiss.write_index(rebuilt, str(target / "faiss_index" / "frames.faiss"))
    del rebuilt, index, base
    sidecar = read_json(source / "faiss_index" / "frames_meta.json")
    removed_ids = {r[0] for r in selected}
    sidecar["fid_to_meta"] = {k: v for k, v in sidecar["fid_to_meta"].items() if k not in removed_ids}
    if {int(v["faiss_id"]) for v in sidecar["fid_to_meta"].values()} != set(kept_ids.tolist()):
        raise RuntimeError("Filtered JSON sidecar and SQLite IDs disagree")
    write_json(target / "faiss_index" / "frames_meta.json", sidecar)
    old_manifest = read_json(source / "serve_manifest.json")
    prepared = prepare_dataset(target, strict_ocr=False, export_clip_text=False,
                               clip_model=old_manifest.get("clip_model", "openai/clip-vit-large-patch14"))
    if not (target / "clip_text").is_dir():
        raise RuntimeError("Source has no local CLIP text artifact")
    # prepare_dataset marks this null when export is skipped; we reuse the
    # existing artifact instead, and the UI requires the explicit locator.
    prepared["clip_text_artifact"] = "clip_text"
    write_json(target / "serve_manifest.json", prepared)
    with sqlite3.connect(target / "meta.db") as db:
        if db.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
            raise RuntimeError("Filtered SQLite integrity check failed")
        db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    log(f"Validated filtered bundle: {len(kept_ids)} frames; removed {len(selected)} indexed frames")


def connect_redis(config, source, target):
    import redis
    from migrate_to_redis import wait_for_redis
    env = config["environment"]
    if env.get("BACKEND") != "hybrid":
        raise RuntimeError("This workflow currently supports the active hybrid backend only")
    client = redis.Redis.from_url(env["REDIS_URL"], socket_timeout=120, protocol=2)
    try:
        client.ping()
    except redis.exceptions.ConnectionError:
        subprocess.run(["bash", str(Path(__file__).with_name("start_redis_wsl.sh"))],
                       env={**os.environ, **env}, check=True)
    wait_for_redis(client)
    prefix = env["REDIS_PREFIX"]
    manifest = json.loads(client.get(prefix + "manifest") or "{}")
    if Path(manifest.get("source_index", "")).resolve() not in {source, target}:
        raise RuntimeError("Redis manifest does not belong to the source or filtered index")
    if manifest.get("frame_vectors") is not False:
        raise RuntimeError("Expected text-only Redis for hybrid serving")
    return client, prefix, manifest


def refresh_redis(client, prefix, manifest, target, selected):
    from migrate_to_redis import key, import_frames, load_rows, wait_index, verify
    manifest = {**manifest, "ready": False}
    client.set(prefix + "manifest", json.dumps(manifest))
    pipe = client.pipeline(transaction=False)
    for i, (_, filename, _) in enumerate(selected, 1):
        pipe.delete(key(prefix, "frame", filename))
        if i % 1000 == 0:
            pipe.execute()
    pipe.execute()
    for video in {Path(row[1]).parent.name.upper() for row in selected}:
        client.delete(prefix + "timeline:" + video)
    files, timelines, statuses = import_frames(
        client, prefix, target, load_rows(target / "meta.db"), target, text_only=True)
    wait_index(client, prefix, "frame", len(files))
    verify(client, prefix)
    manifest.update(ready=True, source_index=str(target), media_root=str(target),
                    frame_count=len(files), video_count=len(timelines),
                    ocr_statuses=statuses, migrated_at=time.time())
    client.set(prefix + "stats", json.dumps(read_json(target / "serve_stats.json")))
    client.set(prefix + "manifest", json.dumps(manifest))
    # Persist to survive run_hybrid_server_wsl.sh's shutdown nosave behavior.
    client.save()


def delete_sources(root, map_root, plans, work):
    from tqdm import tqdm
    numbers = video_removals(plans)
    map_files = list(map_root.rglob("*.csv"))
    present = {p.stem.casefold() for p in map_files}
    if set(numbers) - present:
        raise RuntimeError(f"Missing source maps: {sorted(set(numbers) - present)[:10]}")
    for plan in tqdm(plans, desc="Delete duplicate JPEGs", unit="video"):
        folder = root / plan["folder"]
        for name in plan["remove"]:
            path = (folder / name).resolve()
            if not path.is_relative_to(root):
                raise RuntimeError(f"Deletion outside input root: {path}")
            path.unlink(missing_ok=True)
    for path in map_files:
        removed = numbers.get(path.stem.casefold())
        if removed:
            backup = work / "maps_before" / path.relative_to(map_root)
            if not backup.exists():
                backup.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, backup)
            filter_map(path, removed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path("/mnt/d/AIC_26/data/extracted/keyframes"))
    parser.add_argument("--active-config", type=Path, default=Path(__file__).with_name(".active_dataset.json"))
    parser.add_argument("--output-index", type=Path)
    parser.add_argument("--work-dir", type=Path, default=Path("/mnt/d/AIC_26/data/dedup_active_work"))
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--threshold", type=int, default=4)
    parser.add_argument("--faiss-mode", choices=["preserve", "flat"], default="preserve",
                        help="flat avoids HNSW graph construction; image searches can be slower")
    parser.add_argument("--existing-only", action="store_true",
                        help="Remove references to absent JPEGs; do not hash or delete any source images/maps")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if args.threshold < 0 or args.workers < 1 or args.batch_size < 1:
        parser.error("threshold >= 0, workers/batch-size >= 1 required")
    if os.name != "posix":
        parser.error("Run this workflow in WSL Ubuntu")
    import fcntl
    root, work = args.input_dir.resolve(), args.work_dir.resolve()
    if not root.is_dir() or not (root.parent / "map-keyframes").is_dir():
        parser.error("Input keyframes and sibling map-keyframes must exist")
    work.mkdir(parents=True, exist_ok=True)
    with (work / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state_path = work / "state.json"
        if state_path.exists():
            state = read_json(state_path)
        else:
            config = read_json(args.active_config)
            source = Path(config["environment"]["DATA_ROOT"]).resolve()
            target = (args.output_index or source.with_name(source.name + "_dedup")).resolve()
            if target.exists() or target == source or target.is_relative_to(source) or source.is_relative_to(target):
                raise RuntimeError("Output must be a new directory outside the source index")
            state = {"source": str(source), "target": str(target), "config": config,
                     "config_path": str(args.active_config.resolve()),
                     "root": str(root), "threshold": args.threshold, "phase": "plan",
                     "existing_only": args.existing_only}
            write_json(state_path, state)
        source, target = Path(state["source"]), Path(state["target"])
        if state.get("existing_only", False) != args.existing_only:
            raise RuntimeError("Use a NEW work directory when switching to --existing-only")
        if state["root"] != str(root) or state["threshold"] != args.threshold:
            raise RuntimeError("Resume must use the same input and threshold")
        if (args.output_index and args.output_index.resolve() != target) or str(args.active_config.resolve()) != state["config_path"]:
            raise RuntimeError("Resume must use the same output/config")
        if state["phase"] == "done":
            log(f"Already complete. UI configured for {target}")
            return
        if args.apply:
            # Prevent deletion while the UI still holds the old metadata in RAM.
            with socket.socket() as sock:
                port = int(state["config"]["environment"].get("SERVER_PORT", "8080"))
                if sock.connect_ex(("127.0.0.1", port)) == 0:
                    raise RuntimeError(f"Stop the UI on port {port} and indexing/extraction jobs before --apply")
            import faiss  # dependency check before any image deletion
            import redis
        plan_path = work / "plan.json"
        if plan_path.exists():
            plans = read_json(plan_path)
            validate_images(root, plans, allow_deleted=state["phase"] in {"delete", "redis", "activate"})
        elif args.existing_only:
            log("Enumerating existing JPEG filenames (no pHash, no image decoding)...")
            plans = existing_plan(root)
        else:
            plans = make_plan(root, work, args.device, args.workers, args.batch_size, args.threshold)
            validate_images(root, plans)
        write_json(plan_path, plans)
        selected_path = work / "selected.json"
        if "source_signature" in state and state["phase"] == "plan" and source_signature(source) != state["source_signature"]:
            raise RuntimeError("Source index changed since planning. Use a new work/output directory")
        if "source_signature" not in state:
            state["source_signature"] = source_signature(source)
            write_json(state_path, state)
        selected = read_json(selected_path) if selected_path.exists() else selected_files(source, root, plans)
        write_json(selected_path, selected)
        log(f"Plan: {len(selected)} indexed rows to remove; " +
            ("existing-only, no source JPEG deletion" if args.existing_only else f"{sum(len(p['remove']) for p in plans)} duplicate JPEGs"))
        if not args.apply:
            log("Plan only; add --apply to build, delete, synchronize Redis, and activate")
            return
        config = state["config"]
        current = read_json(args.active_config)
        if current != config and current["environment"].get("DATA_ROOT") != str(target):
            raise RuntimeError("Active configuration changed since planning")
        client, prefix, redis_manifest = connect_redis(config, source, target)
        if state["phase"] == "plan":
            owner = target / ".dedup_owner.json"
            if target.exists() and (not owner.exists() or read_json(owner) != {"work": str(work)}):
                raise RuntimeError("Output directory is not owned by this run")
            target.mkdir(parents=True, exist_ok=True)
            write_json(owner, {"work": str(work)})
            build_filtered(source, target, root, plans, selected, args.faiss_mode)
            if source_signature(source) != state["source_signature"]:
                raise RuntimeError("Source index changed while building. No source images deleted")
            state["phase"] = "delete"
            write_json(state_path, state)
        if state["phase"] == "delete":
            validate_images(root, plans, allow_deleted=True)
            if not args.existing_only:
                delete_sources(root, root.parent / "map-keyframes", plans, work)
            state["phase"] = "redis"
            write_json(state_path, state)
        if state["phase"] == "redis":
            refresh_redis(client, prefix, redis_manifest, target, selected)
            state["phase"] = "activate"
            write_json(state_path, state)
        if state["phase"] == "activate":
            from migrate_to_redis import verify
            verify(client, prefix)
            if json.loads(client.get(prefix + "manifest") or "{}").get("source_index") != str(target):
                raise RuntimeError("Redis has not switched to filtered index")
            write_json(work / "active_config_before.json", config)
            new = json.loads(json.dumps(config))
            new["environment"].update(DATA_ROOT=str(target), KEYFRAMES_DIR=str(target / "keyframes"),
                                      MAPFILES_DIR=str(target / "map-keyframes"),
                                      SUMMARY_INDEX_DIR=str(target / "summary_index"))
            write_json(args.active_config, new)
            state["phase"] = "done"
            write_json(state_path, state)
            log(f"DONE. UI now configured for {target}. Restart run_local_server_wsl.sh.")


if __name__ == "__main__":
    main()
