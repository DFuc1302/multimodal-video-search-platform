#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
if [ -n "${TEAM_VENV:-}" ]; then
  source "$TEAM_VENV/bin/activate"
elif [ -f .venv/bin/activate ]; then
  source .venv/bin/activate
elif [ -f ../aic_local_prep/.venv/bin/activate ]; then
  source ../aic_local_prep/.venv/bin/activate
elif [ -f ../aic_local_prep/.venv_sys/bin/activate ]; then
  source ../aic_local_prep/.venv_sys/bin/activate
elif [ -f /home/dtp/project/aic_local_prep/.venv_sys/bin/activate ]; then
  source /home/dtp/project/aic_local_prep/.venv_sys/bin/activate
elif [ -f /home/dtp/project/aic_local_prep/.venv/bin/activate ]; then
  source /home/dtp/project/aic_local_prep/.venv/bin/activate
else
  echo "ERROR: Chua co Python environment. Hay chay install_requirements_wsl.sh." >&2
  exit 1
fi

export BACKEND_MODE=redis
export STORAGE_BACKEND=redis
export REDIS_URL="${REDIS_URL:-redis://127.0.0.1:${REDIS_PORT:-16379}/0}"
export REDIS_PREFIX="${REDIS_PREFIX:-aic:v2:}"
export KEYFRAMES_DIR="${KEYFRAMES_DIR:-$(cd ../filtered_data_all_873_moved && pwd)}"
export MAPFILES_DIR="${MAPFILES_DIR:-$KEYFRAMES_DIR/map-keyframes}"
export VIDEOS_DIR="${VIDEOS_DIR:-$ROOT/../videos}"
export SERVER_HOST="${SERVER_HOST:-localhost}"
export SERVER_PORT="${SERVER_PORT:-8080}"
if [ -d /home/dtp/project/aic_local_prep/search_index_ocr/clip_text ]; then
  DEFAULT_CLIP_TEXT=/home/dtp/project/aic_local_prep/search_index_ocr/clip_text
elif [ -d "$ROOT/../aic_local_prep/search_index_ocr/clip_text" ]; then
  DEFAULT_CLIP_TEXT="$ROOT/../aic_local_prep/search_index_ocr/clip_text"
else
  DEFAULT_CLIP_TEXT="$ROOT/../clip_text"
fi
export CLIP_TEXT_MODEL_PATH="${CLIP_TEXT_MODEL_PATH:-$DEFAULT_CLIP_TEXT}"
export CLIP_TEXT_ONLY=1
export PRELOAD_CLIP_ON_STARTUP=1
export AUTO_INDEX_ON_STARTUP=0
export SERVE_READ_ONLY=1
export REDIS_FAST_STARTUP=0

REDIS_PORT="${REDIS_PORT:-16379}"
if [ -x /home/dtp/project/aic_local_prep/redis_runtime/usr/bin/redis-cli ]; then
  REDIS_CLI=/home/dtp/project/aic_local_prep/redis_runtime/usr/bin/redis-cli
elif [ -x "$ROOT/../redis_runtime/usr/bin/redis-cli" ]; then
  REDIS_CLI="$ROOT/../redis_runtime/usr/bin/redis-cli"
else
  REDIS_CLI="$(command -v redis-cli || true)"
fi

export ALLOW_PARTIAL_DATA="${ALLOW_PARTIAL_DATA:-1}"
redis_ready=1
bash "$ROOT/start_redis_wsl.sh" || redis_ready=0
if [ "$redis_ready" = "1" ]; then
  python migrate_to_redis.py --verify-only --redis-url "$REDIS_URL" --prefix "$REDIS_PREFIX" || redis_ready=0
fi
if [ "$redis_ready" != "1" ]; then
  if [ "$ALLOW_PARTIAL_DATA" != "1" ]; then exit 1; fi
  export BACKEND_MODE=sqlite STORAGE_BACKEND=sqlite SERVER_REDIS_FALLBACK=1
  echo "Redis chua san sang; server dung SQLite/FAISS hien co."
  exec bash "$ROOT/run_server_wsl.sh"
fi

SERVER_PID=""
cleanup() {
  local exit_code=$?
  trap - INT TERM EXIT
  if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill -TERM "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
  fi
  if [ "${STOP_REDIS_ON_EXIT:-1}" = "1" ] && [ -n "$REDIS_CLI" ]; then
    "$REDIS_CLI" -p "$REDIS_PORT" shutdown nosave >/dev/null 2>&1 || true
  fi
  echo "Da dung server va Redis port $REDIS_PORT."
  exit "$exit_code"
}
trap cleanup INT TERM EXIT

python -m uvicorn main:app --host "$SERVER_HOST" --port "$SERVER_PORT" --no-access-log &
SERVER_PID=$!
wait "$SERVER_PID"
