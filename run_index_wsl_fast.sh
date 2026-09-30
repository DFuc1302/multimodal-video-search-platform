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
export YOLO_MODEL="$PWD/models/yolov8n.pt"
export YOLO_IMGSZ=640
export WHISPER_MODEL=small

# Fast preprocessing mode: no OCR at all. Use this to quickly build CLIP +
# FAISS + YOLO/object/color metadata for testing the web search UI.
export OCR_ENGINE=none
export OCR_GPU=0
export OCR_USE_ANGLE_CLS=0

python ocr_index.py
