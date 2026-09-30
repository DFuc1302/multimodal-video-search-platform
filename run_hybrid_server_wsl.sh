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

# Hybrid mode keeps frame metadata/FAISS local and reads OCR, ASR, and summary
# indexes from the Redis instance belonging to this machine only.
export BACKEND_MODE=hybrid
export STORAGE_BACKEND=sqlite
export REDIS_URL="${REDIS_URL:-redis://127.0.0.1:${REDIS_PORT:-16379}/0}"
export REDIS_PREFIX="${REDIS_PREFIX:-aic:v2:}"
export REDIS_FAST_STARTUP="${REDIS_FAST_STARTUP:-1}"
export PRELOAD_CLIP_ON_STARTUP=1
export SERVER_HOST="${SERVER_HOST:-localhost}"
export SERVER_PORT="${SERVER_PORT:-8080}"

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
if [ "${AIC_SKIP_REDIS_START:-0}" != "1" ]; then
  bash "$ROOT/start_redis_wsl.sh" || redis_ready=0
fi
verify_args=()
if [ "${REQUIRE_REDIS_SOURCE_MATCH:-0}" = "1" ]; then
  verify_args+=(--expected-source-index "$DATA_ROOT")
fi
if [ "$redis_ready" = "1" ] && [ "${AIC_GATEWAY_CHILD:-0}" != "1" ]; then
  python migrate_to_redis.py --verify-only --redis-url "$REDIS_URL" --prefix "$REDIS_PREFIX" "${verify_args[@]}" || redis_ready=0
fi
if [ "$redis_ready" != "1" ]; then
  if [ "$ALLOW_PARTIAL_DATA" != "1" ]; then exit 1; fi
  echo "Redis chua san sang; server dung SQLite/FAISS hien co."
  export BACKEND_MODE=sqlite
  export STORAGE_BACKEND=sqlite
  export SERVER_REDIS_FALLBACK=1
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
  echo "Da dung server hybrid va Redis port $REDIS_PORT."
  exit "$exit_code"
}
trap cleanup INT TERM EXIT

bash "$ROOT/run_server_wsl.sh" &
SERVER_PID=$!
wait "$SERVER_PID"
