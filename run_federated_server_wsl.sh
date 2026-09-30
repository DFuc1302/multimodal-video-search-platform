#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="$PROJECT_DIR/federated_roots.json"

"$PROJECT_DIR/mount_aic_usb_wsl.sh" --quiet

clip_root=""
for candidate in \
  "$PROJECT_DIR/data" "/mnt/d/AIC/data" \
  "/mnt/d/AIC/data/processed/PHUC" \
  "/mnt/d/AIC/data/processed/KHOA_NHO" \
  "$PROJECT_DIR"; do
  if [ -f "$candidate/serve_manifest.json" ] && \
     [ -f "$candidate/meta.db" ] && \
     [ -f "$candidate/faiss_index/frames.faiss" ] && \
     [ -d "$candidate/clip_text" ]; then
    clip_root="$candidate"
    break
  fi
done

if [ -z "$clip_root" ]; then
  echo "LỖI: Chưa có dataset nào được bundle để chạy server." >&2
  echo "Hãy chạy: ./run_multi_storage_wsl.sh run --all --bundle" >&2
  exit 1
fi

export FEDERATED_DATASET_CONFIG="$CONFIG"
export FEDERATED_ALLOW_OFFLINE="${FEDERATED_ALLOW_OFFLINE:-1}"
export DATA_ROOT="$clip_root"
export CLIP_TEXT_MODEL_PATH="$clip_root/clip_text"
export REQUIRE_SERVE_BUNDLE=0
export SERVE_READ_ONLY=1
export AUTO_INDEX_ON_STARTUP=0
export OCR_ON_DEMAND=0

python3 "$PROJECT_DIR/build_storage_catalog.py" \
  --config "$CONFIG" \
  --output "$PROJECT_DIR/data/video_storage_catalog.json"

exec "$PROJECT_DIR/run_server_wsl.sh"
