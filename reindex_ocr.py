"""Stage and resume an OCR-only upgrade without changing the serving dataset."""
import argparse
import json
import os
import shutil
import sqlite3
from pathlib import Path


def queue_outdated(conn, version):
    """Preserve old text until a successful replacement; retry the new recipe fresh."""
    cursor = conn.execute("""
        UPDATE frames SET ocr_status='pending', ocr_attempts=0, ocr_error=''
        WHERE COALESCE(ocr_version,'') != ?
    """, (version,))
    return cursor.rowcount


def stage(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Output must be a separate sibling directory")
    if output.exists():
        raise ValueError("Output already exists; use --resume for a staged upgrade")
    output.mkdir(parents=True)
    try:
        with sqlite3.connect(f"file:{source / 'meta.db'}?mode=ro", uri=True) as src:
            with sqlite3.connect(output / "meta.db") as dst:
                src.backup(dst)
        for name in ("keyframes", "map-keyframes", "faiss_index", "clip_text", "asr_cache.json"):
            asset = source / name
            if asset.exists():
                (output / name).symlink_to(asset, target_is_directory=asset.is_dir())
        if (source / "external_storage.json").exists():
            # Catalog roots are absolute in serving bundles.
            shutil.copy2(source / "external_storage.json", output / "external_storage.json")
        (output / "ocr_upgrade.json").write_text(json.dumps({"source": str(source)}, indent=2))
    except Exception:
        # Retain an interrupted copy for inspection, but it cannot be resumed
        # unless the final staging marker exists.
        raise
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if not args.resume and not args.source:
        parser.error("--source is required unless --resume is used")
    if not args.apply:
        print("Dry run: creates an independent SQLite copy, reruns outdated OCR, then validates it.")
        print("Source:", args.source, "Output:", args.output)
        return 0
    if args.resume:
        output = args.output.resolve()
        if not (output / "ocr_upgrade.json").is_file() or (output / "meta.db").is_symlink():
            raise ValueError("Not an independent staged OCR upgrade")
    else:
        output = stage(args.source, args.output)
    # The serving manifest must be regenerated after OCR and FTS validation.
    (output / "serve_manifest.json").unlink(missing_ok=True)
    os.environ.update(DATA_ROOT=str(output), SERVE_READ_ONLY="0", OCR_ALL_FRAMES="1",
                      META_DB=str(output / "meta.db"), KEYFRAMES_DIR=str(output / "keyframes"))
    from shared import init_db, get_db, OCR_PIPELINE_VERSION, OCR_ENGINE
    from ocr_index import backfill_ocr
    if OCR_ENGINE in {"none", "off", "0", "false", "no"}:
        raise ValueError("OCR engine must be enabled for an upgrade")
    init_db()
    with get_db() as conn:
        queued = queue_outdated(conn, OCR_PIPELINE_VERSION)
    print("Queued:", queued, "Target version:", OCR_PIPELINE_VERSION)
    backfill_ocr()
    with get_db() as conn:
        versions = dict(conn.execute("SELECT ocr_version,COUNT(*) FROM frames GROUP BY ocr_version"))
        incomplete = conn.execute("""SELECT COUNT(*) FROM frames WHERE
            ocr_version != ? OR ocr_version IS NULL OR
            ocr_status NOT IN ('done','no_text','reused')""", (OCR_PIPELINE_VERSION,)).fetchone()[0]
        conn.execute("INSERT INTO frames_fts(frames_fts) VALUES ('rebuild')")
        conn.execute("INSERT INTO frames_fts(frames_fts,rank) VALUES ('integrity-check',1)")
    report = {"target_version": OCR_PIPELINE_VERSION, "versions": versions,
              "incomplete": incomplete, "ready_for_bundle": incomplete == 0}
    (output / "ocr_upgrade_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 1 if incomplete else 0


if __name__ == "__main__":
    raise SystemExit(main())
