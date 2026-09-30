#!/usr/bin/env bash
set -euo pipefail
PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/.venv_rebuild}"

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
  echo "Cách dùng: $0 TEN_THANH_VIEN [auto|visual|asr]" >&2
  echo "Ví dụ: $0 machine-01 auto" >&2
  exit 2
fi

member_name="$1"
mode="${2:-auto}"
output="${SHARD_OUTPUT:-$PROJECT_DIR/data/shard/$member_name}"
source_root="${SHARD_PROJECT:-${DATA_ROOT:-$PROJECT_DIR/data}}"

if [ ! -x "$VENV_DIR/bin/python" ]; then
  echo "LỖI: Không tìm thấy Python tại $VENV_DIR/bin/python" >&2
  exit 1
fi

exec "$VENV_DIR/bin/python" "$PROJECT_DIR/export_shard.py" \
  --project "$source_root" \
  --output "$output" \
  --name "$member_name" \
  --mode "$mode" \
  --asset-mode "${SHARD_ASSET_MODE:-hardlink}"
