# Dedupe extracted keyframes và chuyển UI sang index tổng hợp đã lọc

## Dọn nhanh theo ảnh hiện còn (không dedupe lại)

Dừng lệnh dedupe cũ bằng Ctrl+C và dừng UI trước. Chế độ này chỉ liệt kê tên
JPEG, không đọc/giải mã/tính pHash, không xóa ảnh hoặc sửa map nguồn. Chỉ map
trong index đầu ra được lọc. Dùng work-dir mới để không tiếp tục kế hoạch xóa cũ.

```bash
cd /mnt/c/PERSONAL_PROJECT/AIC_26/CODE
source /home/dtp/project/aic_local_prep/.venv_sys/bin/activate
python dedup_active_index.py \
  --input-dir /mnt/d/AIC_26/data/extracted/keyframes \
  --output-index /mnt/c/PERSONAL_PROJECT/AIC_26/final_search_index_filtered \
  --work-dir /mnt/d/AIC_26/data/index_sync_work \
  --existing-only --faiss-mode flat --apply \
&& TEAM_VENV=/home/dtp/project/aic_local_prep/.venv_sys bash run_local_server_wsl.sh
```

Chỉ loại tham chiếu mất ảnh thuộc đúng nguồn đã chỉ định, bao gồm bảng frame
riêng của summary. Giữ các nguồn khác. Flat tránh xây graph HNSW, nhưng tìm ảnh
có thể chậm hơn. Không cam kết tổng thời gian dưới một giờ. Bỏ `--apply` để xem
số dòng cần loại; lệnh apply sau đó dùng lại inventory đã lưu và kiểm tra chưa đổi.

## Dedupe lại toàn bộ bằng pHash

Chạy trong WSL Ubuntu. Dừng UI (Ctrl+C), extractor, OCR và mọi tiến trình ghi index
trước khi chạy. Script từ chối `--apply` nếu cổng UI 8080 còn mở.

```bash
cd /mnt/c/PERSONAL_PROJECT/AIC_26/CODE
source /home/dtp/project/aic_local_prep/.venv_sys/bin/activate
python dedup_active_index.py \
  --input-dir /mnt/d/AIC_26/data/extracted/keyframes \
  --output-index /mnt/c/PERSONAL_PROJECT/AIC_26/final_search_index_dedup \
  --work-dir /mnt/d/AIC_26/data/dedup_active_work \
  --device cuda --workers 4 --batch-size 256 --threshold 4 --apply
```

Bỏ `--apply` nếu chỉ muốn lập kế hoạch. Kế hoạch được giữ để lệnh apply hoặc lần
resume không tính lại chuỗi ảnh. Nếu bị ngắt, chạy lại cùng lệnh, cùng work-dir.
Muốn một lần dedupe mới sau khi hoàn tất: chọn work-dir và output-index mới.

Nếu cần rút ngắn thời gian build, thêm `--faiss-mode flat`: tái sử dụng vector
nhưng không dựng lại đồ thị HNSW. Tìm kiếm hình ảnh sau đó có thể chậm hơn vì
phải quét toàn bộ vector. Không bảo đảm hoàn tất trong một giờ. Có thể Ctrl+C
ở bước lập kế hoạch/build và chạy lại cùng lệnh với tùy chọn này; chỉ đổi loại
FAISS khi checkpoint còn ở `plan` (chưa hoàn tất build).

Script đọc index tổng hợp từ `.active_dataset.json`, lọc ảnh liên tiếp bằng pHash
với Hamming <= 4, giữ frame đầu mỗi chuỗi. Không đổi số ảnh, `n`, `frame_idx`.
Nó tạo bản index mới với các vector có sẵn; lọc metadata/OCR, FAISS, sidecar JSON,
map CSV và bảng frame riêng trong summary; dựng lại FTS, facets, locator, thống kê.
Summary theo video/đoạn và ASR được giữ nguyên. Không chạy lại CLIP/YOLO/OCR.

Sau khi bản mới kiểm tra thành công, script xóa JPEG trùng và lọc map gốc, dọn
frame Redis trong prefix đang dùng, dựng lại danh sách/timeline, lưu Redis xuống
đĩa, rồi chuyển `.active_dataset.json` sang output-index. Redis đang dùng được
cập nhật tại chỗ; index tổng hợp gốc trên đĩa vẫn giữ để đối chiếu.
Các index trung gian `new_index`, `merged_search_index` không được cập nhật và
không nên dùng lại để phục vụ hoặc gộp lại nếu chưa đồng bộ tập frame đã xóa.

Ảnh bị xóa vĩnh viễn. Backup CSV và cấu hình nằm trong work-dir; đó không phải
backup JPEG. Cần dung lượng trống cho bản SQLite, FAISS và summary mới trên ổ C;
FAISS HNSW được dựng lại từ vector cũ, nên vẫn cần RAM và thời gian xây graph.

Chỉ khi lệnh hoàn tất thành công, khởi động UI bằng terminal sạch (không export
DATA_ROOT/REDIS_PREFIX từ lần chạy cũ):

```bash
cd /mnt/c/PERSONAL_PROJECT/AIC_26/CODE
TEAM_VENV=/home/dtp/project/aic_local_prep/.venv_sys bash run_local_server_wsl.sh
```

Script dùng cấu hình hybrid và manifest external của index hiện tại. Một lỗi ở
bước Redis giữ cấu hình UI cũ; tiếp tục cùng lệnh để hoàn thành trước khi mở UI.
