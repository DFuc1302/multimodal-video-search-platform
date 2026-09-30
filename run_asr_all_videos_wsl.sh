#!/usr/bin/env bash
set -euo pipefail

# Local ASR runner for the full AIC video folder.
# It reuses asr_index.py, writes a resumable cache, and skips videos that
# already have current ASR segments.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VIDEO_DIR="${VIDEO_DIR:-/mnt/d/AIC/data/videos}"
DATA_ROOT="${DATA_ROOT:-/mnt/d/AIC/data}"
ASR_CACHE_FILE="${ASR_CACHE_FILE:-$DATA_ROOT/asr_cache.json}"
WHISPER_MODEL="${WHISPER_MODEL:-small}"
ASR_CHUNK_SECONDS="${ASR_CHUNK_SECONDS:-2.0}"

if [ ! -d "$VIDEO_DIR" ]; then
  echo "Không tìm thấy thư mục video: $VIDEO_DIR" >&2
  exit 1
fi

mkdir -p "$(dirname "$ASR_CACHE_FILE")"

PYTHON_BIN="${PYTHON_BIN:-}"
if [ -z "$PYTHON_BIN" ]; then
  if [ -x "$SCRIPT_DIR/.venv_rebuild/bin/python" ]; then
    PYTHON_BIN="$SCRIPT_DIR/.venv_rebuild/bin/python"
  elif [ -x "$SCRIPT_DIR/../.venv_rebuild/bin/python" ]; then
    PYTHON_BIN="$SCRIPT_DIR/../.venv_rebuild/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi

echo "=== ASR toàn bộ video ==="
echo "Video dir:       $VIDEO_DIR"
echo "Data root:       $DATA_ROOT"
echo "ASR cache:       $ASR_CACHE_FILE"
echo "Whisper model:   $WHISPER_MODEL"
echo "Chunk seconds:   $ASR_CHUNK_SECONDS"
echo "Python:          $PYTHON_BIN"
echo

export VIDEO_DIR
export DATA_ROOT
export VIDEOS_DIR="$VIDEO_DIR"
export ASR_CACHE_FILE
export WHISPER_MODEL
export ASR_CHUNK_SECONDS

"$PYTHON_BIN" asr_index.py --folder "$VIDEO_DIR"

echo
echo "Xong. Cache ASR nằm ở: $ASR_CACHE_FILE"
