#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

export TEAM_VENV="${TEAM_VENV:-/home/dtp/project/aic_local_prep/.venv_sys}"
export AIC_USE_ACTIVE_DATASET=0
export BACKEND=hybrid
export DATA_ROOT=/mnt/d/AIC_26/data/merged_search_index
export META_DB="$DATA_ROOT/meta.db"
export FAISS_DIR="$DATA_ROOT/faiss_index"
export KEYFRAMES_DIR="$DATA_ROOT/keyframes"
export MAPFILES_DIR="$DATA_ROOT/map-keyframes"
export VIDEOS_DIR=/mnt/d/AIC_26/data/videos_2
export CLIP_TEXT_MODEL_PATH="$DATA_ROOT/clip_text"
export REDIS_PORT=16380
export REDIS_URL=redis://127.0.0.1:16380/0
export REDIS_PREFIX=aic:merged:20260924:
export REDIS_DATA_DIR=/mnt/d/AIC_26/data/new_dataset_work/redis_data
export REQUIRE_REDIS_SOURCE_MATCH=1
export STOP_REDIS_ON_EXIT=0
export ALLOW_PARTIAL_DATA=1
exec bash run_local_server_wsl.sh
