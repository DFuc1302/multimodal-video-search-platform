#!/usr/bin/env bash
set -e

cd "$(dirname "${BASH_SOURCE[0]}")"

if [ "$#" -gt 1 ] || { [ "$#" -eq 1 ] && [ "$1" != "--check" ]; }; then
  echo "Usage: DATA_ROOT=/path/to/index bash run_ocr_only_wsl.sh [--check]" >&2
  exit 2
fi

# Respect the environment the caller activated (for example .venv_sys).
if [ -z "${VIRTUAL_ENV:-}" ] && [ -d ".venv_rebuild" ]; then
  source .venv_rebuild/bin/activate
fi

# OCR-only requires an existing index. Never create an empty database when
# DATA_ROOT is missing or points to an unmounted drive.
python - <<'PY'
import os
from pathlib import Path
import sqlite3
import sys

root = Path(os.environ.get("DATA_ROOT", "data")).expanduser().resolve()
database = Path(os.environ.get("META_DB", str(root / "meta.db"))).expanduser().resolve()
print(f"OCR DATA_ROOT: {root}", flush=True)
print(f"OCR META_DB:   {database}", flush=True)
if not database.is_file():
    sys.exit(
        f"ERROR: Khong tim thay database OCR: {database}\n"
        "Hay dat DATA_ROOT den thu muc index da co meta.db. Vi du dataset moi:\n"
        "DATA_ROOT=/mnt/d/AIC_26/data/new_dataset_work/new_index bash run_ocr_only_wsl.sh\n"
        "Neu META_DB da duoc export, kiem tra lai bien nay; kiem tra o D da mount."
    )
try:
    with sqlite3.connect(database.as_uri() + "?mode=rw", uri=True) as conn:
        conn.execute("SELECT file FROM frames LIMIT 1").fetchone()
    if not os.access(database, os.W_OK) or not os.access(database.parent, os.W_OK):
        raise PermissionError("Can quyen ghi database va thu muc chua database")
except (sqlite3.Error, OSError) as exc:
    sys.exit(f"ERROR: Khong mo duoc index OCR {database}: {exc}")
print("OCR database check: OK", flush=True)
PY

if [ "${1:-}" = "--check" ]; then
  exit 0
fi

# OCR-only phase. This does not rebuild CLIP/FAISS/YOLO. It only fills missing
# OCR text in meta.db, then updates SQLite FTS5 for fast OCR search.
export OCR_ENGINE=easyocr
export OCR_LANGS=en,vi
export OCR_GPU=1
# Keep white camera-overlay text (road names and timestamps) on black panels.
export OCR_WHITE_ON_BLACK_ONLY=${OCR_WHITE_ON_BLACK_ONLY:-1}
export OCR_BATCH_SIZE=${OCR_BATCH_SIZE:-12}
export OCR_ENHANCE=${OCR_ENHANCE:-0}
export OCR_TARGET_HOURS=${OCR_TARGET_HOURS:-30}
export OCR_MAX_SIDE=${OCR_MAX_SIDE:-960}
export OCR_PREFETCH_WORKERS=${OCR_PREFETCH_WORKERS:-8}
export OCR_DEDUP_CACHE_SIZE=${OCR_DEDUP_CACHE_SIZE:-4096}
export OCR_PIPELINE_VERSION=${OCR_PIPELINE_VERSION:-ocr-regions-v3}
export OCR_USE_ANGLE_CLS=0

# Process all remaining frames, including those skipped by the initial heuristic.
export OCR_ALL_FRAMES=${OCR_ALL_FRAMES:-1}

python - <<'PY'
import torch
from shared import (init_db, get_db, log, OCR_WHITE_ON_BLACK_ONLY,
                    OCR_BATCH_SIZE, OCR_ENHANCE)
from ocr_index import backfill_ocr

if not torch.cuda.is_available():
    raise SystemExit("ERROR: CUDA khong san sang; can GPU de nham muc tieu OCR 30 gio.")
log.info("OCR GPU: %s; batch=%d; enhance=%s", torch.cuda.get_device_name(),
         OCR_BATCH_SIZE, OCR_ENHANCE)
log.info("=== OCR-only: initializing database ===")
init_db()
log.info("White-on-black OCR filter: %s", OCR_WHITE_ON_BLACK_ONLY)
backfill_ocr()

with get_db() as conn:
    total = conn.execute("SELECT COUNT(*) FROM frames").fetchone()[0]
    counts = dict(conn.execute(
        "SELECT ocr_status, COUNT(*) FROM frames GROUP BY ocr_status"
    ).fetchall())

log.info(
    "=== OCR-only complete: total=%d, done=%d, reused=%d, no_text=%d, "
    "skipped=%d, pending=%d, error=%d ===",
    total,
    counts.get("done", 0),
    counts.get("reused", 0),
    counts.get("no_text", 0),
    counts.get("skipped", 0),
    counts.get("pending", 0) + counts.get("processing", 0),
    counts.get("error", 0),
)
PY
