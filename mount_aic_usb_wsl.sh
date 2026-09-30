#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
WINDOWS_DRIVE="D:"
MOUNT_POINT="/mnt/d"
WSL_LAUNCHER="/mnt/c/Windows/System32/wsl.exe"
DISTRO_NAME="${WSL_DISTRO_NAME:-Ubuntu-24.04}"

# -T reports the filesystem containing a path. For an ordinary empty /mnt/d
# that is the WSL root device (for example /dev/sdd), not an actual /mnt/d
# mount. --mountpoint only succeeds when this exact path is a mount target.
if findmnt --mountpoint "$MOUNT_POINT" -n >/dev/null 2>&1; then
  mounted_source="$(findmnt --mountpoint "$MOUNT_POINT" -n -o SOURCE)"
  mounted_fstype="$(findmnt --mountpoint "$MOUNT_POINT" -n -o FSTYPE)"
  normalized_source="${mounted_source%\\}"
  if { [ "${normalized_source^^}" != "${WINDOWS_DRIVE^^}" ] || \
       { [ "$mounted_fstype" != "9p" ] && [ "$mounted_fstype" != "drvfs" ]; }; }; then
    echo "LỖI: $MOUNT_POINT đang trỏ tới $mounted_source, không phải $WINDOWS_DRIVE" >&2
    exit 1
  fi
else
  if [ ! -x "$WSL_LAUNCHER" ]; then
    echo "LỖI: Không tìm thấy WSL launcher: $WSL_LAUNCHER" >&2
    exit 1
  fi
  mkdir -p "$MOUNT_POINT"
  echo "Đang mount USB $WINDOWS_DRIVE vào $MOUNT_POINT..."
  "$WSL_LAUNCHER" -d "$DISTRO_NAME" -u root -- \
    mount -t drvfs "$WINDOWS_DRIVE" "$MOUNT_POINT"
fi

if [ "${1:-}" = "--quiet" ]; then
  python3 "$PROJECT_DIR/preprocess_dataset.py" \
    --config "$PROJECT_DIR/dataset_roots.json" status --shard usb >/dev/null
else
  python3 "$PROJECT_DIR/preprocess_dataset.py" \
    --config "$PROJECT_DIR/dataset_roots.json" status --shard usb
fi
