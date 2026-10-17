#!/usr/bin/env python3
"""Finalize merged preprocessing data and leave the project ready to serve.

Run once from anywhere:

    python3 /home/dtp/project/finalize_merged_data.py

The default run is strict. It validates every transferred artifact, refuses
incomplete OCR before touching the active dataset, finds/copies required
videos, prepares a working Python runtime, promotes merged_data, runs ASR,
builds the read-only serving bundle, and performs final integrity checks. OCR
is only run centrally when --run-pending-ocr is explicitly supplied.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_SUFFIXES = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".m4v"}
ACTIVE_NAMES = (
    "keyframes",
    "map-keyframes",
    "faiss_index",
    "meta.db",
    "asr_cache.json",
    "serve_manifest.json",
    "serve_stats.json",
    "clip_text",
)
REQUIRED_MERGED = (
    "keyframes",
    "map-keyframes",
    "faiss_index/frames.faiss",
    "faiss_index/frames_meta.json",
    "meta.db",
)
RUNTIME_REQUIREMENTS = {
    "faiss": "faiss-cpu==1.9.0.post1",
    "easyocr": "easyocr==1.7.2",
    "whisper": "openai-whisper==20240930",
    "fastapi": "fastapi",
    "uvicorn": "uvicorn[standard]",
    "transformers": "transformers",
    "ultralytics": "ultralytics",
    "multipart": "python-multipart",
}


class FinalizeError(RuntimeError):
    pass


@dataclass
class DatasetReport:
    root: Path
    sqlite_count: int
    fts_count: int
    faiss_count: int
    sidecar_count: int
    image_count: int
    csv_count: int
    video_stems: list[str]
    missing_ocr_candidates: int


def log(message: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def run(command: list[str], *, cwd: Path, env: dict | None = None) -> None:
    printable = " ".join(command)
    log(f"RUN: {printable}")
    completed = subprocess.run(command, cwd=cwd, env=env)
    if completed.returncode != 0:
        raise FinalizeError(f"Command failed ({completed.returncode}): {printable}")


def normalize_rel(value: str) -> str:
    raw = str(value or "").replace("\\", "/")
    path = PurePosixPath(raw)
    if not raw or path.is_absolute() or ".." in path.parts:
        raise FinalizeError(f"Unsafe relative path in metadata: {value!r}")
    return str(path)


def stable_faiss_id(fid: str) -> int:
    digest = hashlib.blake2b(fid.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") & 0x7FFFFFFFFFFFFFFF


def inspect_dataset(root: Path, *, require_fts: bool = True) -> DatasetReport:
    root = root.resolve()
    for relative in REQUIRED_MERGED:
        if not (root / relative).exists():
            raise FinalizeError(f"Missing required merged artifact: {root / relative}")

    db_path = root / "meta.db"
    try:
        with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as db:
            integrity = db.execute("PRAGMA integrity_check").fetchone()[0]
            if integrity != "ok":
                raise FinalizeError(f"SQLite integrity failed for {db_path}: {integrity}")
            rows = db.execute("SELECT id, file FROM frames").fetchall()
            table_names = {
                row[0]
                for row in db.execute(
                    "SELECT name FROM sqlite_master WHERE type IN ('table','view')"
                )
            }
            fts_count = (
                db.execute("SELECT COUNT(*) FROM frames_fts").fetchone()[0]
                if "frames_fts" in table_names
                else -1
            )
            frame_columns = {
                row[1] for row in db.execute("PRAGMA table_info(frames)")
            }
            if "ocr_status" in frame_columns:
                missing_ocr = db.execute(
                    "SELECT COUNT(*) FROM frames WHERE "
                    "CASE WHEN ocr_status IN "
                    "('pending','processing','done','no_text','skipped','error','reused') "
                    "THEN ocr_status "
                    "WHEN COALESCE(ocr_joined,'')!='' THEN 'done' "
                    "WHEN COALESCE(has_text,0)=0 THEN 'skipped' ELSE 'pending' END "
                    "IN ('pending','processing','error')"
                ).fetchone()[0]
            else:
                missing_ocr = db.execute(
                    "SELECT COUNT(*) FROM frames WHERE COALESCE(has_text,0)=1 "
                    "AND COALESCE(ocr_joined,'')=''"
                ).fetchone()[0]
    except sqlite3.Error as exc:
        raise FinalizeError(f"Cannot validate SQLite {db_path}: {exc}") from exc

    db_ids: set[str] = set()
    db_files: set[str] = set()
    for fid, filename in rows:
        rel = normalize_rel(filename)
        if fid in db_ids or rel in db_files:
            raise FinalizeError(f"Duplicate frame in SQLite: {fid!r} / {rel!r}")
        db_ids.add(str(fid))
        db_files.add(rel)

    sidecar_path = root / "faiss_index" / "frames_meta.json"
    try:
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        vector_rows = sidecar["fid_to_meta"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise FinalizeError(f"Cannot validate FAISS sidecar {sidecar_path}: {exc}") from exc
    if not isinstance(vector_rows, dict):
        raise FinalizeError(f"Invalid fid_to_meta in {sidecar_path}")

    vector_ids: set[str] = set()
    vector_files: set[str] = set()
    for fid, metadata in vector_rows.items():
        rel = normalize_rel(metadata.get("file", ""))
        faiss_id = int(metadata.get("faiss_id", -1))
        if faiss_id != stable_faiss_id(fid):
            raise FinalizeError(f"Invalid stable FAISS id for {fid}")
        vector_ids.add(fid)
        vector_files.add(rel)

    image_root = root / "keyframes"
    image_files = {
        str(path.relative_to(image_root)).replace("\\", "/")
        for path in image_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }
    if db_ids != vector_ids:
        raise FinalizeError(
            f"SQLite/FAISS ids differ: SQLite={len(db_ids)}, sidecar={len(vector_ids)}"
        )
    if db_files != vector_files or db_files != image_files:
        raise FinalizeError(
            "SQLite/FAISS/keyframe file sets differ: "
            f"SQLite={len(db_files)}, FAISS={len(vector_files)}, images={len(image_files)}"
        )

    video_stems = sorted({PurePosixPath(filename).parts[0] for filename in db_files})
    csv_files = {path.stem for path in (root / "map-keyframes").glob("*.csv") if path.is_file()}
    missing_csv = sorted(set(video_stems) - csv_files)
    if missing_csv:
        raise FinalizeError(f"Missing map CSV files for: {', '.join(missing_csv)}")

    if require_fts and fts_count != len(rows):
        raise FinalizeError(f"SQLite FTS count differs: frames={len(rows)}, FTS={fts_count}")

    try:
        import faiss

        index = faiss.read_index(str(root / "faiss_index" / "frames.faiss"))
        if index.ntotal != len(vector_rows):
            raise FinalizeError(
                f"FAISS binary count differs: index={index.ntotal}, sidecar={len(vector_rows)}"
            )
        if index.d != int(sidecar.get("dim", index.d)):
            raise FinalizeError(
                f"FAISS dimensions differ: index={index.d}, sidecar={sidecar.get('dim')}"
            )
        first = next(iter(vector_rows.values()))
        vector = index.reconstruct(int(first["faiss_id"]))
        if len(vector) != index.d:
            raise FinalizeError("FAISS vector reconstruction returned the wrong dimension")
    except ImportError as exc:
        raise FinalizeError("Python runtime does not have faiss installed") from exc
    except Exception as exc:
        if isinstance(exc, FinalizeError):
            raise
        raise FinalizeError(f"Cannot validate FAISS binary index: {exc}") from exc

    return DatasetReport(
        root=root,
        sqlite_count=len(rows),
        fts_count=fts_count,
        faiss_count=int(index.ntotal),
        sidecar_count=len(vector_rows),
        image_count=len(image_files),
        csv_count=len(csv_files),
        video_stems=video_stems,
        missing_ocr_candidates=missing_ocr,
    )


def print_report(label: str, report: DatasetReport) -> None:
    log(
        f"{label}: SQLite={report.sqlite_count}, FTS={report.fts_count}, "
        f"FAISS={report.faiss_count}, sidecar={report.sidecar_count}, "
        f"images={report.image_count}, CSV={report.csv_count}, "
        f"pending OCR candidates={report.missing_ocr_candidates}"
    )


def python_imports_ok(python: Path, modules: list[str]) -> bool:
    code = "; ".join(f"import {name}" for name in modules)
    return subprocess.run(
        [str(python), "-c", code],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode == 0


def ensure_runtime(project: Path, script: Path) -> Path:
    candidates = [
        project / ".venv_runtime" / "bin" / "python",
        project / ".venv_rebuild" / "bin" / "python",
        Path(sys.executable),
    ]
    runtime = next(
        (path for path in candidates if path.is_file() and os.access(path, os.X_OK)),
        None,
    )
    if runtime is None or not python_imports_ok(runtime, ["faiss", "numpy"]):
        runtime_dir = project / ".venv_runtime"
        if not runtime_dir.exists():
            log(f"Creating Python runtime: {runtime_dir}")
            run(
                [sys.executable, "-m", "venv", "--system-site-packages", str(runtime_dir)],
                cwd=project,
            )
        runtime = runtime_dir / "bin" / "python"
        run(
            [str(runtime), "-m", "pip", "install", "faiss-cpu==1.9.0.post1"],
            cwd=project,
        )

    if Path(sys.executable).absolute() != runtime.absolute() and os.getenv("FINALIZE_REEXEC") != "1":
        log(f"Restarting finalize script with runtime: {runtime}")
        environment = os.environ.copy()
        environment["FINALIZE_REEXEC"] = "1"
        os.execve(str(runtime), [str(runtime), str(script), *sys.argv[1:]], environment)
    return runtime


def install_runtime_dependencies(runtime: Path, project: Path) -> None:
    missing = []
    for module, package in RUNTIME_REQUIREMENTS.items():
        if not python_imports_ok(runtime, [module]):
            missing.append(package)
    if missing:
        log(f"Installing missing runtime packages: {', '.join(missing)}")
        run([str(runtime), "-m", "pip", "install", *missing], cwd=project)
    remaining = [module for module in RUNTIME_REQUIREMENTS if not python_imports_ok(runtime, [module])]
    if remaining:
        raise FinalizeError(f"Runtime is still missing modules: {', '.join(remaining)}")


def stop_project_processes(project: Path) -> None:
    current_pid = os.getpid()
    stopped = []
    for proc_dir in Path("/proc").iterdir():
        if not proc_dir.name.isdigit() or int(proc_dir.name) == current_pid:
            continue
        try:
            cwd = proc_dir.joinpath("cwd").resolve()
            command = proc_dir.joinpath("cmdline").read_bytes().replace(b"\0", b" ").decode(
                "utf-8", errors="replace"
            )
        except (OSError, PermissionError):
            continue
        if cwd != project and project not in cwd.parents:
            continue
        if not any(token in command for token in ("uvicorn", "main.py", "ocr_index.py", "asr_index.py")):
            continue
        pid = int(proc_dir.name)
        try:
            os.kill(pid, signal.SIGTERM)
            stopped.append(pid)
        except ProcessLookupError:
            pass
    if stopped:
        log(f"Stopped project process(es): {stopped}")
        deadline = time.time() + 30
        remaining = set(stopped)
        while remaining and time.time() < deadline:
            remaining = {pid for pid in remaining if Path(f"/proc/{pid}").exists()}
            if remaining:
                time.sleep(0.25)
        if remaining:
            raise FinalizeError(
                f"Project process(es) did not stop after SIGTERM: {sorted(remaining)}"
            )


def find_and_copy_videos(project: Path, stems: list[str]) -> list[str]:
    destination = project / "data" / "videos"
    destination.mkdir(exist_ok=True)
    search_roots = [
        project / "data" / "merged_data" / "videos",
        project / "data" / "shard",
    ]
    found_by_stem: dict[str, Path] = {}

    for path in destination.iterdir():
        if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES:
            found_by_stem[path.stem] = path
    for search_root in search_roots:
        if not search_root.is_dir():
            continue
        for path in search_root.rglob("*"):
            if path.is_file() and path.suffix.lower() in VIDEO_SUFFIXES:
                found_by_stem.setdefault(path.stem, path)

    missing = []
    for stem in stems:
        source = found_by_stem.get(stem)
        if source is None:
            missing.append(stem)
            continue
        target = destination / source.name
        if source.resolve() != target.resolve() and not target.exists():
            log(f"Copying video {source} -> {target}")
            shutil.copy2(source, target)
    return missing


def backup_and_promote(project: Path, merged: Path) -> Path:
    backup = project / f"backup_before_finalize_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    backup.mkdir()
    moved_active: list[tuple[Path, Path]] = []
    promoted: list[tuple[Path, Path]] = []
    try:
        for name in ACTIVE_NAMES:
            source = project / name
            if source.exists():
                destination = backup / name
                shutil.move(str(source), str(destination))
                moved_active.append((destination, source))
        for name in ACTIVE_NAMES:
            source = merged / name
            if source.exists():
                destination = project / name
                shutil.move(str(source), str(destination))
                promoted.append((destination, source))
        for temp_name in ("meta.db-wal", "meta.db-shm"):
            source = project / temp_name
            if source.exists():
                shutil.move(str(source), str(backup / temp_name))
    except Exception:
        for destination, original in reversed(promoted):
            if destination.exists():
                shutil.move(str(destination), str(original))
        for saved, original in reversed(moved_active):
            if saved.exists():
                shutil.move(str(saved), str(original))
        raise
    log(f"Active dataset backup: {backup}")
    return backup


def restore_after_failed_promotion(project: Path, merged: Path, backup: Path) -> None:
    """Restore the previous active dataset and return the candidate to merged_data."""
    merged.mkdir(parents=True, exist_ok=True)
    for name in reversed(ACTIVE_NAMES):
        active = project / name
        candidate = merged / name
        if active.exists():
            if candidate.exists():
                raise FinalizeError(f"Cannot rollback; candidate path exists: {candidate}")
            shutil.move(str(active), str(candidate))
    for name in ACTIVE_NAMES:
        saved = backup / name
        active = project / name
        if saved.exists():
            if active.exists():
                raise FinalizeError(f"Cannot rollback; active path exists: {active}")
            shutil.move(str(saved), str(active))
    log(f"Promotion rolled back from backup: {backup}")


def run_pending_ocr(runtime: Path, project: Path, data_root: Path) -> None:
    with sqlite3.connect(data_root / "meta.db") as db:
        pending = db.execute(
            "SELECT COUNT(*) FROM frames "
            "WHERE COALESCE(has_text,0)=1 AND COALESCE(ocr_joined,'')=''"
        ).fetchone()[0]
    if pending == 0:
        log("OCR: no pending text-candidate frames")
        return
    log(f"OCR: processing {pending} pending text-candidate frame(s)")
    environment = os.environ.copy()
    environment.update(
        {
            "DATA_ROOT": str(data_root),
            "OCR_ENGINE": "easyocr",
            "OCR_LANGS": "en,vi",
            "OCR_GPU": "1",
            "OCR_BATCH_SIZE": os.getenv("OCR_BATCH_SIZE", "6"),
            "OCR_ALL_FRAMES": "0",
            "OCR_MAX_SIDE": os.getenv("OCR_MAX_SIDE", "960"),
            "OCR_PREFETCH_WORKERS": os.getenv("OCR_PREFETCH_WORKERS", "4"),
            "OCR_DEDUP_CACHE_SIZE": os.getenv("OCR_DEDUP_CACHE_SIZE", "4096"),
            "OCR_PIPELINE_VERSION": os.getenv(
                "OCR_PIPELINE_VERSION", "easyocr-batched-v2"
            ),
        }
    )
    code = (
        "from shared import init_db; "
        "from ocr_index import backfill_ocr; "
        "init_db(); backfill_ocr()"
    )
    run([str(runtime), "-c", code], cwd=project, env=environment)


def run_asr(runtime: Path, project: Path, data_root: Path) -> None:
    environment = os.environ.copy()
    environment["WHISPER_MODEL"] = os.getenv("WHISPER_MODEL", "small")
    environment["DATA_ROOT"] = str(data_root)
    run(
        [
            str(runtime),
            str(project / "asr_index.py"),
            "--folder",
            str(project / "data" / "videos"),
        ],
        cwd=project,
        env=environment,
    )


def prepare_serving_bundle(data_root: Path) -> None:
    from prepare_serve_bundle import PrepareError, prepare_dataset

    sidecar = json.loads(
        (data_root / "faiss_index" / "frames_meta.json").read_text(encoding="utf-8")
    )
    clip_model = str(sidecar.get("clip_model") or "openai/clip-vit-large-patch14")
    log("Building SQLite facets/stats and local CLIP text-only serving artifact")
    try:
        prepare_dataset(
            data_root, strict_ocr=True, export_clip_text=True, clip_model=clip_model
        )
    except PrepareError as exc:
        raise FinalizeError(f"Cannot prepare serving bundle: {exc}") from exc


def asr_video_names(cache_path: Path) -> set[str]:
    if not cache_path.exists():
        return set()
    data = json.loads(cache_path.read_text(encoding="utf-8"))
    names = set()
    for key in data:
        value = str(key)
        if value.startswith("video-name-v1::"):
            names.add(Path(value.split("::", 1)[1]).stem)
        else:
            legacy_path = value.rsplit("::", 1)[0].replace("\\", "/")
            names.add(Path(legacy_path).stem)
    return names


def write_final_report(project: Path, report: DatasetReport, backup: Path | None) -> None:
    payload = {
        "completed_at": datetime.now().isoformat(timespec="seconds"),
        "sqlite_count": report.sqlite_count,
        "fts_count": report.fts_count,
        "faiss_count": report.faiss_count,
        "sidecar_count": report.sidecar_count,
        "image_count": report.image_count,
        "csv_count": report.csv_count,
        "videos": report.video_stems,
        "asr_videos": sorted(asr_video_names(project / "data" / "asr_cache.json")),
        "backup": str(backup) if backup else None,
        "server_command": "./run_server_wsl.sh",
    }
    (project / "data" / "finalize_report.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project",
        default=str(Path(__file__).resolve().parents[2]),
        help="Project root (defaults to this script's directory)",
    )
    parser.add_argument(
        "--run-pending-ocr",
        action="store_true",
        help="Explicitly allow this machine to finish incomplete OCR after promotion",
    )
    parser.add_argument(
        "--skip-asr",
        action="store_true",
        help="Skip central ASR completion and its strict completeness check",
    )
    args = parser.parse_args()
    project = Path(args.project).resolve()
    storage_root = project / "data"
    merged = storage_root / "merged_data"
    script = Path(__file__).resolve()
    lock = project / ".finalize_merged_data.lock"

    runtime = ensure_runtime(project, script)
    if lock.exists():
        raise FinalizeError(f"Another finalize run may be active: {lock}")
    lock.write_text(str(os.getpid()), encoding="utf-8")
    backup = None
    promoted = False
    try:
        has_candidate = all(
            (merged / relative).exists() for relative in REQUIRED_MERGED
        )
        data_root = merged if has_candidate else storage_root
        if has_candidate:
            log("STEP 1/8: validating merged_data candidate")
        else:
            log("STEP 1/8: validating the already-active dataset")
        source_report = inspect_dataset(data_root)
        print_report("candidate validated", source_report)
        expected_stems = source_report.video_stems

        if source_report.missing_ocr_candidates and not args.run_pending_ocr:
            raise FinalizeError(
                f"OCR is incomplete for {source_report.missing_ocr_candidates} frame(s). "
                "Let each source machine finish its existing OCR run, then merge again. "
                "Use --run-pending-ocr only if this machine should explicitly finish them."
            )

        log("STEP 2/8: locating and copying required original videos")
        missing_videos = find_and_copy_videos(project, expected_stems)
        if missing_videos:
            raise FinalizeError(
                "Missing original videos; no data was promoted. Put these files in "
                "data/videos, data/merged_data/videos, or data/shard/<machine>/videos and rerun: "
                + ", ".join(f"{stem}.mp4" for stem in missing_videos)
            )

        log("STEP 3/8: preparing OCR/ASR/server Python dependencies")
        install_runtime_dependencies(runtime, project)

        log("STEP 4/8: completing optional OCR/ASR on the candidate")
        if args.run_pending_ocr:
            run_pending_ocr(runtime, project, data_root)
        else:
            log("OCR: central rerun disabled; using completed shard results")
        if args.skip_asr:
            log("ASR: skipped by --skip-asr")
        else:
            run_asr(runtime, project, data_root)

        log("STEP 5/8: preparing the read-only serving bundle before promotion")
        prepare_serving_bundle(data_root)

        log("STEP 6/8: validating the fully prepared candidate")
        candidate_report = inspect_dataset(data_root)
        if candidate_report.missing_ocr_candidates:
            raise FinalizeError(
                f"OCR is still incomplete for "
                f"{candidate_report.missing_ocr_candidates} frame(s)"
            )
        cached_asr = asr_video_names(data_root / "asr_cache.json")
        missing_asr = sorted(set(candidate_report.video_stems) - cached_asr)
        if missing_asr and not args.skip_asr:
            raise FinalizeError(f"ASR cache is still missing videos: {', '.join(missing_asr)}")
        print_report("prepared candidate", candidate_report)

        if has_candidate:
            log("STEP 7/8: stopping writers and atomically promoting prepared data")
            stop_project_processes(project)
            backup = backup_and_promote(storage_root, merged)
            promoted = True
        else:
            log("STEP 7/8: dataset is already active; promotion skipped")

        log("STEP 8/8: verifying the active dataset")
        final_report = inspect_dataset(storage_root)
        print_report("FINAL", final_report)
        write_final_report(project, final_report, backup)

        print("\nFINALIZE COMPLETE", flush=True)
        print(f"Report: {project / 'data' / 'finalize_report.json'}", flush=True)
        print(f"Start the server with: cd {project} && ./run_server_wsl.sh", flush=True)
        return 0
    except BaseException as exc:
        if promoted and backup is not None:
            try:
                restore_after_failed_promotion(storage_root, merged, backup)
            except Exception as rollback_exc:
                print(
                    f"\nFINALIZE ERROR: {exc}; ROLLBACK ERROR: {rollback_exc}",
                    file=sys.stderr,
                    flush=True,
                )
                return 1
        if isinstance(exc, KeyboardInterrupt):
            print("\nFINALIZE INTERRUPTED", file=sys.stderr, flush=True)
            return 130
        if not isinstance(exc, FinalizeError):
            raise
        print(f"\nFINALIZE ERROR: {exc}", file=sys.stderr, flush=True)
        return 1
    finally:
        try:
            lock.unlink(missing_ok=True)
        except OSError:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
