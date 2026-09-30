#!/usr/bin/env bash
set -e

if [ -d ".venv_rebuild" ]; then
  source .venv_rebuild/bin/activate
fi

export AUTO_INDEX_ON_STARTUP=0
export CLIP_BATCH_SIZE=8
export FAISS_INDEX_TYPE=hnsw
export FAISS_HNSW_M=32
export FAISS_EF_SEARCH=96
export FAISS_SAVE_EVERY=50000
export YOLO_MODEL="$PWD/models/yolov8m.pt"
export YOLO_IMGSZ=640
export WHISPER_MODEL=small

# Full mode: OCR with EasyOCR. Slower, but useful after the fast CLIP/YOLO
# index is already validated.
export OCR_ENGINE=easyocr
export OCR_LANGS=en,vi
export OCR_GPU=1
export OCR_BATCH_SIZE=${OCR_BATCH_SIZE:-8}
export OCR_USE_ANGLE_CLS=0

python ocr_index.py
