#!/usr/bin/env bash
set -e

APP_PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$APP_PROJECT_DIR"

if [ -n "${TEAM_VENV:-}" ]; then
  source "$TEAM_VENV/bin/activate"
elif [ -d ".venv_rebuild" ]; then
  source .venv_rebuild/bin/activate
fi

# Prefer the central local index. All runtime data lives below project/data.
if [ -z "${DATA_ROOT:-}" ]; then
  if [ -f "data/search_index/serve_manifest.json" ]; then
    DATA_ROOT="data/search_index"
  elif [ -f "data/merged_data/serve_manifest.json" ]; then
    DATA_ROOT="data/merged_data"
  else
    DATA_ROOT="data"
  fi
fi
export DATA_ROOT
export ALLOW_PARTIAL_DATA="${ALLOW_PARTIAL_DATA:-1}"
export SERVER_LOG_PATH="${SERVER_LOG_PATH:-$APP_PROJECT_DIR/sever_log.json}"
export MAPFILES_DIR="${MAPFILES_DIR:-$DATA_ROOT/map-keyframes}"
# Original videos stay on the configured removable storage by default and are
# deliberately separate from merged index artifacts under DATA_ROOT.
export VIDEOS_DIR="${VIDEOS_DIR:-/mnt/d/AIC/data/videos}"

# Bundles built with --asset-mode symlink keep images/CSVs on their original
# local/USB volumes. Validate mount points and volume markers before startup.
if [ "$ALLOW_PARTIAL_DATA" != "1" ]; then
  python "$APP_PROJECT_DIR/check_external_storage.py" --data-root "$DATA_ROOT"
fi

if [ -n "${FEDERATED_DATASET_CONFIG:-}" ]; then
  # DATA_ROOT is only the CLIP text artifact host. SQLite/FAISS/keyframes/maps
  # are validated independently by FederatedRuntime for every online root.
  required_paths=(
    "$DATA_ROOT/serve_manifest.json"
    "$DATA_ROOT/clip_text"
  )
else
  required_paths=(
    "$DATA_ROOT/meta.db"
    "$DATA_ROOT/faiss_index/frames.faiss"
    "$DATA_ROOT/serve_manifest.json"
    "$DATA_ROOT/serve_stats.json"
    "$DATA_ROOT/clip_text"
    "${KEYFRAMES_DIR:-$DATA_ROOT/keyframes}"
    "$MAPFILES_DIR"
  )
fi

for required_path in "${required_paths[@]}"; do
  if [ ! -e "$required_path" ]; then
    echo "ERROR: Dữ liệu merge chưa đầy đủ, thiếu: $required_path" >&2
    if [ "$ALLOW_PARTIAL_DATA" != "1" ]; then exit 1; fi
  fi
done

export AUTO_INDEX_ON_STARTUP=0
export PRELOAD_CLIP_ON_STARTUP=1
export SERVE_READ_ONLY=1
export META_CACHE_MODE=lazy
export FAISS_SQLITE_MAPPING=1
export FAISS_MMAP=${FAISS_MMAP:-1}
export REQUIRE_SERVE_BUNDLE="${REQUIRE_SERVE_BUNDLE:-1}"
export OCR_ON_DEMAND=0
export CLIP_TEXT_ONLY=1
export CLIP_TEXT_MODEL_PATH="${CLIP_TEXT_MODEL_PATH:-$DATA_ROOT/clip_text}"
if [ -z "${CLIP_MODEL:-}" ]; then
  CLIP_MODEL="$(python -c 'import json,sys; from pathlib import Path; p=Path(sys.argv[1]); print(json.loads(p.read_text(encoding="utf-8")).get("clip_model", "openai/clip-vit-large-patch14") if p.is_file() else "openai/clip-vit-large-patch14")' "$DATA_ROOT/serve_manifest.json" 2>/dev/null)" || CLIP_MODEL=openai/clip-vit-large-patch14
fi
export CLIP_MODEL
export SEARCH_ASR_BOOST=${SEARCH_ASR_BOOST:-0}
export ALLOW_HEAVY_RUNTIME_JOBS=${ALLOW_HEAVY_RUNTIME_JOBS:-0}
export CLIP_BATCH_SIZE=8
export FAISS_HNSW_M=32
export FAISS_EF_SEARCH=96
export YOLO_MODEL="$APP_PROJECT_DIR/models/yolov8m.pt"
export YOLO_IMGSZ=640
export WHISPER_MODEL=small
export OCR_GPU=0
export OCR_USE_ANGLE_CLS=0
export FLAGS_use_mkldnn=0
export FLAGS_use_onednn=0
export FLAGS_enable_pir_api=0
export FLAGS_enable_pir_in_executor=0

SERVER_HOST="${SERVER_HOST:-localhost}"
export SERVER_PORT="${SERVER_PORT:-8080}"

echo "Server đang nạp CLIP/FAISS/FTS và kiểm tra search readiness..."
echo "Nguồn dữ liệu: $DATA_ROOT"
echo "Nguồn video: $VIDEOS_DIR"
echo "Đường dẫn server sẽ xuất hiện sau khi CLIP và kiểm tra tìm kiếm sẵn sàng."

exec python -m uvicorn main:app --host "$SERVER_HOST" --port "$SERVER_PORT" --no-access-log
