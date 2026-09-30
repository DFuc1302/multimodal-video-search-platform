#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
export TEAM_DATA="${TEAM_DATA:-$ROOT/../DATA_FOR_TEAM}"
TEAM_ENV_DIR="${TEAM_VENV:-$HOME/.venvs/aic-team}"
if [ -z "${TEAM_VENV:-}" ] && [ ! -f "$TEAM_ENV_DIR/bin/activate" ]; then
  for candidate in "$ROOT/.venv" "$ROOT/../CODE/.venv"; do
    if [ -f "$candidate/bin/activate" ]; then
      TEAM_ENV_DIR="$candidate"
      break
    fi
  done
fi
if [ ! -f "$TEAM_ENV_DIR/bin/activate" ]; then
  echo "ERROR: Khong tim thay Python environment: $TEAM_ENV_DIR" >&2
  echo "Chay: bash install_requirements_wsl.sh" >&2
  echo "Hoac: TEAM_VENV=/duong/dan/toi/venv bash run_local_server_wsl.sh" >&2
  exit 1
fi
echo "Python environment: $TEAM_ENV_DIR"
source "$TEAM_ENV_DIR/bin/activate"
export BACKEND_MODE=hybrid STORAGE_BACKEND=sqlite
export DATA_ROOT="${DATA_ROOT:-$TEAM_DATA/search_index_ocr}"
export KEYFRAMES_DIR="${KEYFRAMES_DIR:-$ROOT/../filtered_data_all_873_moved}"
export MAPFILES_DIR="$KEYFRAMES_DIR/map-keyframes"
export VIDEOS_DIR="${VIDEOS_DIR:-$ROOT/../videos}"
export CLIP_TEXT_MODEL_PATH="$DATA_ROOT/clip_text"
export SUMMARY_MODEL_PATH="$TEAM_DATA/models/multilingual-e5-small"
export CLIP_MODEL=openai/clip-vit-large-patch14
export CLIP_TEXT_ONLY=1 SERVE_READ_ONLY=1 META_CACHE_MODE=lazy FAISS_SQLITE_MAPPING=1 FAISS_MMAP=1
export REQUIRE_SERVE_BUNDLE=1 AUTO_INDEX_ON_STARTUP=0 ALLOW_HEAVY_RUNTIME_JOBS=0 OCR_ON_DEMAND=0
export PRELOAD_CLIP_ON_STARTUP=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 FEDERATED_DATASET_CONFIG=''
export REDIS_PORT="${REDIS_PORT:-16379}" REDIS_PREFIX=aic:v2:
export REDIS_URL="redis://127.0.0.1:$REDIS_PORT/0"
export SERVER_PORT="${SERVER_PORT:-8080}"
python team_runtime.py
