#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [ "$#" -eq 0 ]; then
  echo "Cách dùng: $0 COMMAND [ARGS...]" >&2
  echo "Ví dụ:    $0 status" >&2
  echo "           $0 run --all" >&2
  exit 2
fi

"$PROJECT_DIR/mount_aic_usb_wsl.sh" --quiet
exec python3 "$PROJECT_DIR/preprocess_dataset.py" \
  --config "$PROJECT_DIR/dataset_roots.json" "$@"
