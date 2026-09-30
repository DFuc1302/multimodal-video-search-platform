#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CENTRAL_ROOT="${CENTRAL_SEARCH_ROOT:-$PROJECT_DIR/data/search_index}"

"$PROJECT_DIR/mount_aic_usb_wsl.sh" --quiet

if [ ! -f "$CENTRAL_ROOT/serve_manifest.json" ]; then
  echo "LỖI: Chưa có central search index: $CENTRAL_ROOT" >&2
  echo "Hãy chạy: python3 build_central_search_index.py" >&2
  exit 1
fi

unset FEDERATED_DATASET_CONFIG
export DATA_ROOT="$CENTRAL_ROOT"
export VIDEOS_DIR="${VIDEOS_DIR:-/mnt/d/AIC/data/videos}"
export SERVE_READ_ONLY=1
export OCR_ON_DEMAND=0

exec "$PROJECT_DIR/run_server_wsl.sh"
