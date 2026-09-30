#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export TEAM_VENV="${TEAM_VENV:-/home/dtp/project/aic_local_prep/.venv_sys}"
source "$TEAM_VENV/bin/activate"
export AIC_USE_ACTIVE_DATASET=0 BACKEND="${BACKEND:-hybrid}"
export DATA_ROOT=/mnt/c/PERSONAL_PROJECT/AIC_26/final_search_index
if [ "${1:-run}" = "run" ] && [ "$BACKEND" != "sqlite" ] && [ -f "$DATA_ROOT/redis_refresh_required.json" ]; then
  echo 'Redis chua dong bo voi index moi. Tam chay: BACKEND=sqlite bash run_final_hybrid_wsl.sh run' >&2
  exit 1
fi
export SUMMARY_INDEX_DIR="$(python -c 'from summary_catalog import active_index; print(active_index())')"
export SUMMARY_BACKEND=sqlite
export SUMMARY_MODEL_PATH="${SUMMARY_MODEL_PATH:-$DATA_ROOT/../DATA_FOR_TEAM/models/multilingual-e5-small}"
export DRES_OFFICIAL_MAPS="${DRES_OFFICIAL_MAPS:-$DATA_ROOT/aic26-b2-map-keyframes.zip}"
export KEYFRAMES_DIR="$DATA_ROOT/keyframes"
export MAPFILES_DIR="$DATA_ROOT/map-keyframes"
export ASR_CACHE_FILE="$DATA_ROOT/asr_cache.json"
export CLIP_TEXT_MODEL_PATH="$DATA_ROOT/clip_text"
export VIDEOS_DIR=/mnt/d/AIC_26/data/videos_2
export REDIS_PORT=16381
export REDIS_URL=redis://127.0.0.1:16381/0
export REDIS_PREFIX=aic:final:ssd:text:v1:
export REDIS_DATA_DIR=/mnt/c/PERSONAL_PROJECT/AIC_26/redis_final_text
export REQUIRE_REDIS_SOURCE_MATCH=1 ALLOW_PARTIAL_DATA=0
export ISOLATE_VISUAL_WORKER="${ISOLATE_VISUAL_WORKER:-1}"
export VISUAL_SERVER_PORT="${VISUAL_SERVER_PORT:-8082}"
export REDIS_MAXMEMORY="${REDIS_MAXMEMORY:-6gb}"
case "${1:-run}" in
  summary)
    shift
    exec python -u update_summary_index.py --data-root "$DATA_ROOT" "$@"
    ;;
  prepare)
    bash start_redis_wsl.sh
    python -u migrate_to_redis.py --source-index "$DATA_ROOT" \
      --media-root "$DATA_ROOT" --summary-index "$SUMMARY_INDEX_DIR" \
      --text-only --redis-url "$REDIS_URL" --prefix "$REDIS_PREFIX"
    ;;
  run)
    if [ "${AIC_GATEWAY_CHILD:-0}" != "1" ]; then
      export STOP_REDIS_ON_EXIT="${STOP_REDIS_ON_EXIT:-0}"
      exec python -m uvicorn competition_server:app --host "${SERVER_HOST:-localhost}" --port "${SERVER_PORT:-8080}" --no-access-log
    fi
    exec bash run_local_server_wsl.sh
    ;;
  *) echo 'Usage: bash run_final_hybrid_wsl.sh summary|prepare|run' >&2; exit 2 ;;
esac
