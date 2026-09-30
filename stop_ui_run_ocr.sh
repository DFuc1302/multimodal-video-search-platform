#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/dtp/project"
CODE_DIR="$PROJECT_ROOT/ex_code"
SOURCE_INDEX="${SOURCE_INDEX:-$PROJECT_ROOT/aic_local_prep/search_index}"
OCR_INDEX="${OCR_INDEX:-$PROJECT_ROOT/aic_local_prep/search_index_ocr}"
PORTS="${PORTS:-8080 18088 8000 7860 8501}"

cd "$CODE_DIR"

stop_port() {
  local port="$1"
  local pids=""
  pids="$(fuser -n tcp "$port" 2>/dev/null || true)"
  if [ -n "$pids" ]; then
    echo "Stopping process(es) on port $port: $pids"
    kill $pids 2>/dev/null || true
    sleep 1
    pids="$(fuser -n tcp "$port" 2>/dev/null || true)"
    if [ -n "$pids" ]; then
      echo "Force stopping process(es) on port $port: $pids"
      kill -9 $pids 2>/dev/null || true
    fi
  fi
}

echo "=== Stop all local retrieval UI/backend ports ==="
for port in $PORTS; do
  stop_port "$port"
done

# Catch uvicorn instances that may not be on the known ports anymore.
pkill -f "uvicorn main:app" 2>/dev/null || true
pkill -f "python.*main.py" 2>/dev/null || true

echo "=== Remaining server processes ==="
ps -eo pid,cmd | grep -E "uvicorn|main:app|python.*main.py" | grep -v grep || true

if [ -f "$CODE_DIR/.venv/bin/activate" ]; then
  source "$CODE_DIR/.venv/bin/activate"
elif [ -f "$PROJECT_ROOT/.venv_rebuild/bin/activate" ]; then
  source "$PROJECT_ROOT/.venv_rebuild/bin/activate"
elif [ -f "$CODE_DIR/.venv_rebuild/bin/activate" ]; then
  source "$CODE_DIR/.venv_rebuild/bin/activate"
else
  echo "No virtualenv found. Create one first or set PATH to Python env." >&2
  exit 1
fi

export OCR_ENGINE="${OCR_ENGINE:-easyocr}"
export OCR_LANGS="${OCR_LANGS:-en,vi}"
export OCR_GPU="${OCR_GPU:-1}"
export OCR_BATCH_SIZE="${OCR_BATCH_SIZE:-4}"
export OCR_MAX_SIDE="${OCR_MAX_SIDE:-960}"
export OCR_PREFETCH_WORKERS="${OCR_PREFETCH_WORKERS:-4}"
export OCR_DEDUP_CACHE_SIZE="${OCR_DEDUP_CACHE_SIZE:-4096}"
export OCR_PIPELINE_VERSION="${OCR_PIPELINE_VERSION:-ocr-regions-v3}"
export OCR_REGION_LIMIT="${OCR_REGION_LIMIT:-8}"
export OCR_ALL_FRAMES="${OCR_ALL_FRAMES:-1}"
export SERVE_READ_ONLY=0

mkdir -p "$(dirname "$OCR_INDEX")"

if [ -f "$OCR_INDEX/ocr_upgrade.json" ] && [ -f "$OCR_INDEX/meta.db" ]; then
  echo "=== Resume OCR upgrade: $OCR_INDEX ==="
  python "$CODE_DIR/reindex_ocr.py" --output "$OCR_INDEX" --resume --apply
else
  echo "=== Create OCR upgrade copy ==="
  echo "Source: $SOURCE_INDEX"
  echo "Output: $OCR_INDEX"
  python "$CODE_DIR/reindex_ocr.py" --source "$SOURCE_INDEX" --output "$OCR_INDEX" --apply
fi

echo "=== OCR finished. Report ==="
if [ -f "$OCR_INDEX/ocr_upgrade_report.json" ]; then
  python -m json.tool "$OCR_INDEX/ocr_upgrade_report.json"
else
  echo "No report produced. Check terminal errors above."
fi
