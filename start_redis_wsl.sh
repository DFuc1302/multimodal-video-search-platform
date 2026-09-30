#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [ -d /home/dtp/project/aic_local_prep ]; then
  DEFAULT_REDIS_ROOT=/home/dtp/project/aic_local_prep
elif [ -d "$ROOT/../aic_local_prep" ]; then
  DEFAULT_REDIS_ROOT="$ROOT/../aic_local_prep"
else
  DEFAULT_REDIS_ROOT="$ROOT/.."
fi
REDIS_ROOT="${REDIS_ROOT:-$DEFAULT_REDIS_ROOT}"
REDIS_DATA_DIR="${REDIS_DATA_DIR:-$REDIS_ROOT/redis_data_local}"
REDIS_PORT="${REDIS_PORT:-16379}"
REDIS_BIN="${REDIS_BIN:-$REDIS_ROOT/redis_runtime/usr/bin/redis-server}"
REDIS_CLI="${REDIS_CLI:-$REDIS_ROOT/redis_runtime/usr/bin/redis-cli}"
MODULE_DIR="${REDIS_ROOT}/redis_runtime/usr/lib/redis/modules"

if [ ! -x "$REDIS_BIN" ]; then
  REDIS_BIN="$(command -v redis-server || true)"
  REDIS_CLI="$(command -v redis-cli || true)"
fi
if [ -z "$REDIS_BIN" ] || [ -z "$REDIS_CLI" ]; then
  echo "ERROR: Can redis-server/redis-cli hoac gui aic_local_prep/redis_runtime." >&2
  exit 1
fi

mkdir -p "$REDIS_DATA_DIR"
if "$REDIS_CLI" -p "$REDIS_PORT" ping >/dev/null 2>&1; then
  echo "Redis dang chay tai 127.0.0.1:$REDIS_PORT"
  exit 0
fi

CONFIG="$REDIS_DATA_DIR/redis-local.conf"
{
  echo "bind 127.0.0.1"
  echo "port $REDIS_PORT"
  echo "protected-mode yes"
  echo "dir $REDIS_DATA_DIR"
  echo "appendonly yes"
  echo "appendfsync everysec"
  echo "save 900 1"
  echo "maxmemory ${REDIS_MAXMEMORY:-6gb}"
  echo "maxmemory-policy noeviction"
  if [ -f "$MODULE_DIR/rejson.so" ]; then echo "loadmodule $MODULE_DIR/rejson.so"; fi
  if [ -f "$MODULE_DIR/redisearch.so" ]; then echo "loadmodule $MODULE_DIR/redisearch.so"; fi
  echo "search-max-search-results 1000000"
  echo "search-timeout 30000"
  echo "search-on-timeout fail"
} > "$CONFIG"

exec "$REDIS_BIN" "$CONFIG" --daemonize yes --logfile "$REDIS_DATA_DIR/redis.log" \
  --pidfile "$REDIS_DATA_DIR/redis.pid"
