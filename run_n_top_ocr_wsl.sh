#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source /home/dtp/project/aic_local_prep/.venv_sys/bin/activate
export AIC_USE_ACTIVE_DATASET=0
export DATA_ROOT=/mnt/d/AIC_26/data/new_dataset_work/new_index
exec python -u rerun_n_top_ocr.py \
  --root /mnt/d/AIC_26/data/extracted/keyframes \
  --root /mnt/d/AIC_26/data/batch3_downloads/Video_N031-N040_processed-001/keyframes \
  --output /mnt/d/AIC_26/data/new_dataset_work/n_white_top_ocr.sqlite3 "$@"
