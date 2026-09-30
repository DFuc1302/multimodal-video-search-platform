#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${1:-/home/dtp/project}"

if [[ ! -d "$ROOT" ]]; then
  echo "ERROR: Không tìm thấy thư mục dự án: $ROOT"
  exit 1
fi

cd "$ROOT"

echo "============================================================"
echo " AI CHALLENGE — DỌN FILE RÁC / FILE BẢN CŨ"
echo " Thư mục: $ROOT"
echo "============================================================"

# Các file lõi phải còn tồn tại sau khi dọn.
required_files=(
  "main.py"
  "index.html"
  "shared.py"
  "ocr_index.py"
  "asr_index.py"
  "kisc.py"
)

missing=0
for file in "${required_files[@]}"; do
  if [[ ! -f "$file" ]]; then
    echo "CẢNH BÁO: thiếu file lõi: $file"
    missing=1
  fi
done

if [[ "$missing" -eq 1 ]]; then
  echo
  echo "Dừng để tránh dọn nhầm một thư mục không phải project hiện tại."
  exit 1
fi

echo
echo "Các mục sẽ bị xóa:"
echo "  - mọi thư mục __pycache__ và file *.pyc"
echo "  - mọi file *:Zone.Identifier"
echo "  - asr_frames/ (ASR mới dùng ảnh có sẵn trong keyframes/)"
echo "  - backup_before_*/"
echo "  - các file backup main/index cũ"
echo "  - các patch/script sửa tạm ASR cũ"
echo "  - README sửa lỗi tạm đã hết tác dụng"
echo
echo "Các dữ liệu QUAN TRỌNG sẽ được giữ:"
echo "  .venv_rebuild/  data/ (toàn bộ dataset và index)"
echo "  main.py  index.html  shared.py  ocr_index.py"
echo "  asr_index.py  kisc.py  re_extract.py"
echo "  các run_*.sh, install_*.sh, requirements_*.txt"
echo "  yolov8n.pt và yolov8m.pt"
echo

read -r -p "Gõ DELETE để xóa vĩnh viễn các mục trên: " confirm
if [[ "$confirm" != "DELETE" ]]; then
  echo "Đã hủy. Không có file nào bị xóa."
  exit 0
fi

echo
echo "[1/7] Xóa Python cache..."
find . -type d -name "__pycache__" -prune -exec rm -rf -- {} +
find . -type f \( -name "*.pyc" -o -name "*.pyo" \) -delete

echo "[2/7] Xóa Zone.Identifier và file rác Windows..."
find . -type f \( \
  -name "*:Zone.Identifier" -o \
  -name "Thumbs.db" -o \
  -name "desktop.ini" -o \
  -name ".DS_Store" \
\) -delete

echo "[3/7] Xóa các thư mục backup tạm..."
find . -maxdepth 1 -type d -name "backup_before_*" -prune -exec rm -rf -- {} +

echo "[4/7] Xóa asr_frames cũ..."
rm -rf -- asr_frames

echo "[5/7] Xóa backup file cũ..."
rm -f -- \
  index.html.backup \
  index.html.backup_* \
  main.py.backup \
  main.py.backup_* \
  shared.py.backup \
  shared.py.backup_* \
  asr_index.py.backup \
  asr_index.py.backup_* \
  kisc.py.backup \
  kisc.py.backup_*

echo "[6/7] Xóa các file sửa tạm/phiên bản trùng đã hết tác dụng..."
rm -f -- \
  "main(3).py" \
  "main (3).py" \
  apply_asr_4_columns.sh \
  main_asr_keyframe_fix.py \
  re_extract_backup.py \
  disable_ocr_ondemand.diff

echo "[7/7] Xóa README tạm của các lần sửa trước..."
rm -f -- \
  README_ASR_2S \
  README_ASR_2S.txt \
  README_FIX \
  README_FIX.txt \
  README_TOP_CONTROLS \
  README_TOP_CONTROLS.txt

# Xóa thư mục rỗng còn sót, nhưng không đụng vào các thư mục dữ liệu quan trọng.
find . -depth -type d -empty \
  ! -path "." \
  ! -path "./data" \
  ! -path "./data/*" \
  ! -path "./videos" \
  ! -path "./keyframes" \
  ! -path "./map-keyframes" \
  ! -path "./faiss_index" \
  ! -path "./.venv_rebuild" \
  -delete 2>/dev/null || true

echo
echo "============================================================"
echo " DỌN DẸP HOÀN TẤT"
echo "============================================================"
echo
echo "Cấu trúc cấp 1 hiện tại:"
find . -maxdepth 1 -mindepth 1 -printf "%f\n" | sort
echo
echo "Kiểm tra import backend:"
if [[ -x ".venv_rebuild/bin/python" ]]; then
  if .venv_rebuild/bin/python -m py_compile main.py shared.py ocr_index.py asr_index.py kisc.py; then
    echo "OK: Python compile thành công."
  else
    echo "CẢNH BÁO: Có lỗi compile. Xem log phía trên."
    exit 2
  fi
else
  echo "Bỏ qua compile vì không tìm thấy .venv_rebuild/bin/python."
fi
