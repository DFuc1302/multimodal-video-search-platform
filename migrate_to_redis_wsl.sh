#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
if [ -f .venv/bin/activate ]; then
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

bash "$ROOT/start_redis_wsl.sh"
export REDIS_URL="${REDIS_URL:-redis://127.0.0.1:${REDIS_PORT:-16379}/0}"
export REDIS_PREFIX="${REDIS_PREFIX:-aic:v2:}"

if [ -d /home/dtp/project/aic_local_prep ]; then
  DEFAULT_SOURCE_INDEX=/home/dtp/project/aic_local_prep/search_index_ocr
  DEFAULT_SUMMARY_INDEX=/home/dtp/project/aic_local_prep/summary_index
else
  DEFAULT_SOURCE_INDEX=../aic_local_prep/search_index_ocr
  DEFAULT_SUMMARY_INDEX=../aic_local_prep/summary_index
fi

python migrate_to_redis.py \
  --source-index "${SOURCE_INDEX:-$DEFAULT_SOURCE_INDEX}" \
  --summary-index "${SUMMARY_INDEX:-$DEFAULT_SUMMARY_INDEX}" \
  --media-root "${MEDIA_ROOT:-../filtered_data_all_873_moved}" \
  --redis-url "$REDIS_URL" --prefix "$REDIS_PREFIX"
