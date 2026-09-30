# Bổ sung toàn bộ dataset mới vào UI cũ

Chạy các lệnh dưới đây trong WSL Ubuntu. Script chỉ xử lý khi bạn tự chạy.
Không chạy extractor hoặc script dedup khác đồng thời với pipeline này.

## Dữ liệu và quy tắc

- Nguồn video: `/mnt/d/AIC_26/data/videos_2`, chỉ đọc.
- Keyframe: `/mnt/d/AIC_26/data/extracted/keyframes/<video>/<số>.jpg`.
- Map: `/mnt/d/AIC_26/data/extracted/map-keyframes/<video>.csv`.
- Dedupe GPU pHash, khoảng cách Hamming <= 4: giữ frame đầu của mỗi chuỗi
  liên tiếp từ 2 frame. So sánh từng cặp ảnh liền nhau trước khi xóa,
  không so tất cả ảnh với frame đầu. Không loại trùng giữa các video.
- Xóa vật lý ảnh trùng, lọc các dòng tương ứng trong map. Không đổi tên ảnh,
  không đánh lại số `n`, không tính lại `frame_idx` của video gốc.
- CLIP dùng đúng model trong manifest dataset cũ; YOLO và OCR dùng pipeline hiện có.
  Đây là metadata đối tượng YOLO, không phải bộ phân loại chủ đề riêng.
- ASR do người khác xử lý. Pipeline này không chạy Whisper/ASR, không yêu cầu
  ASR hoàn tất; dữ liệu ASR được cung cấp sẵn vẫn được nhập và tìm kiếm.
- Tái sử dụng vector/OCR/ASR cũ, tạo index gộp mới. Giữ nguyên ảnh/map/video
  tại nguồn bằng manifest đường dẫn, không tạo symlink hoặc sao chép media.
- Đồng bộ Redis riêng cổng 16380 để OCR/ASR và tìm kiếm hình ảnh cùng bộ dữ liệu.
  Summary hiện tại được lấy từ dataset cũ; pipeline không sinh summary mới.

Ảnh đã bị xóa từ lần dedup trước không thể được khôi phục bằng script này.
Backup map không phải backup ảnh. Nếu muốn làm lại từ toàn bộ ảnh ban đầu,
phải trích xuất lại trước khi bắt đầu pipeline này.

## 1. Chuẩn bị

```bash
cd /mnt/c/PERSONAL_PROJECT/AIC_26/CODE
source /home/dtp/project/aic_local_prep/.venv_sys/bin/activate
python -m pip install 'ImageHash==4.3.2'

if ! mountpoint -q /mnt/d; then
  sudo mkdir -p /mnt/d
  sudo mount -t drvfs D: /mnt/d
fi
```

Pipeline dùng cùng môi trường Python với UI. Môi trường cần có PyTorch CUDA,
Pillow, NumPy, SciPy, tqdm, faiss-cpu, EasyOCR, Transformers, Ultralytics,
Redis và OpenCV. Không cần Whisper/ffmpeg cho pipeline này. Lệnh `--stage all`
kiểm tra các module và CUDA trước khi xóa ảnh; nếu thiếu module, cài gói tương ứng
vào môi trường trên. Không cài lại PyTorch CPU đè lên bản CUDA đang dùng.
Redis Stack runtime hiện tại được lấy từ `/home/dtp/project/aic_local_prep/redis_runtime`.

## 2. Kiểm tra danh sách và xem trước dedup

```bash
python run_new_dataset_pipeline.py --stage audit
python run_new_dataset_pipeline.py --stage dedup --dry-run \
  --workers 4 --batch-size 256 --threshold 4
```

Audit yêu cầu mỗi video nguồn có thư mục ảnh và map. Nếu thiếu, hãy hoàn tất
extractor cho video còn thiếu/lỗi trước. Audit ghi báo cáo, không xóa ảnh.
Dry-run tính số ảnh sẽ xóa, không cập nhật ảnh hoặc map.

## 3. Chạy toàn bộ

```bash
python run_new_dataset_pipeline.py --stage all \
  --extracted-dir /mnt/d/AIC_26/data/extracted \
  --videos-dir /mnt/d/AIC_26/data/videos_2 \
  --old-index /home/dtp/project/aic_local_prep/search_index_ocr \
  --work-dir /mnt/d/AIC_26/data/new_dataset_work \
  --merged-index /mnt/d/AIC_26/data/merged_search_index \
  --workers 4 --batch-size 256 --threshold 4 \
  --clip-batch-size 8 --ocr-batch-size 8
```

Thứ tự: audit → dedup → CLIP/YOLO/OCR → đóng gói → gộp → kiểm tra
→ Redis → cấu hình UI. Dừng ở bước lỗi và ghi log; chạy lại **cùng lệnh** để tiếp tục.
Danh sách ảnh phải xóa được ghi trước khi xóa, tránh tính lại chuỗi sau một lần chạy
bị ngắt. Khi index gộp đã xuất bản, index mới được giữ cố định.

Nếu hết VRAM, chạy lại với `--clip-batch-size 2 --ocr-batch-size 1`.
Không đổi threshold/model/thư mục nguồn giữa các lần resume.
Giữ cùng work-dir để không dedup lần nữa trên ảnh đã lọc.

Có thể chạy từng bước bằng `--stage dedup`, `index`, `bundle`, `merge`,
`verify`, `redis`, `activate`. Chạy theo thứ tự trên. Các bước sau dedup sẽ từ chối
nếu ảnh hoặc map đã thay đổi so với trạng thái lưu.

## 4. Mở lại UI

### Chạy tiếp riêng OCR

Launcher OCR cần đường dẫn index đã tồn tại; `CODE/data` không phải index
dataset mới. Trong môi trường Python đã kích hoạt, chạy:

```bash
DATA_ROOT=/mnt/d/AIC_26/data/new_dataset_work/new_index bash run_ocr_only_wsl.sh --check
DATA_ROOT=/mnt/d/AIC_26/data/new_dataset_work/new_index bash run_ocr_only_wsl.sh
```

`--check` chỉ kiểm tra database, không chạy OCR. Launcher mặc định bật lọc chữ
trắng trên nền đen (`OCR_WHITE_ON_BLACK_ONLY=1`). Lệnh tiếp tục các ảnh đang chờ;
ảnh đã hoàn tất không tự chạy lại. Không chạy đồng thời với bước index của pipeline.

Cấu hình tốc độ mặc định: GPU, batch 12, cạnh dài tối đa 960 px, 8 luồng đọc ảnh,
`OCR_ENHANCE=0` (bỏ tối đa 8 lượt OCR tăng cường mỗi ảnh). Bộ lọc màu vẫn kiểm tra
ảnh gốc. Có thể bỏ sót chữ nhỏ/mờ hơn so với chế độ tăng cường. Mỗi phút có log
ảnh/giây, thời gian còn lại và cảnh báo nếu tốc độ không đạt `OCR_TARGET_HOURS=30`.
Đây là mục tiêu theo tốc độ đo được, không phải cam kết thời gian; ghi SQLite,
nhiệt độ GPU và tải máy có thể làm thay đổi tốc độ. Launcher dừng nếu CUDA không
sẵn sàng. Có thể đặt `OCR_BATCH_SIZE=8` khi cần giảm VRAM.

### Khởi động UI

Sau khi pipeline báo `UI configured`, dừng UI cũ và chạy:

```bash
cd /mnt/c/PERSONAL_PROJECT/AIC_26/CODE
TEAM_VENV=/home/dtp/project/aic_local_prep/.venv_sys bash run_local_server_wsl.sh
```

Mở `http://localhost:8080`. Launcher đọc `.active_dataset.json`, dùng SQLite/FAISS
gộp và Redis tương ứng. Cần gắn ổ D khi truy xuất ảnh của dataset mới.
Các biến môi trường DATA_ROOT/REDIS_* bạn tự export sẽ ưu tiên hơn cấu hình;
nên mở terminal sạch nếu trước đó có đặt những biến này.

## Vị trí kết quả

- `/mnt/d/AIC_26/data/new_dataset_work/new_index`: index riêng dataset mới.
- `/mnt/d/AIC_26/data/merged_search_index`: index gộp cũ + mới.
- `new_dataset_work/dedup_plans`: danh sách xóa và trạng thái theo video.
- `new_dataset_work/maps_before_dedup`: backup CSV trước khi lọc.
- `new_dataset_work/logs`: log index/merge/Redis.
- `new_dataset_work/audit.json`, `verification.json`: báo cáo.
- `new_dataset_work/redis_data`: dữ liệu Redis mới; cần RAM cho cả index Redis.
  Mặc định Redis giới hạn 6 GB; nếu gặp OOM, điều chỉnh REDIS_MAXMEMORY phù hợp
  RAM thực tế rồi khởi động lại instance cổng 16380 trước khi chạy lại bước Redis.

## Quay lại cấu hình UI trước khi gộp

```bash
python run_new_dataset_pipeline.py --stage rollback
TEAM_VENV=/home/dtp/project/aic_local_prep/.venv_sys bash run_local_server_wsl.sh
```

Rollback chỉ phục hồi cấu hình UI trước lần activate; không phục hồi ảnh đã xóa.


## Server với dữ liệu xử lý chưa đủ

Mặc định `ALLOW_PARTIAL_DATA=1`. Server đọc dữ liệu hiện có, không tự chạy OCR
hoặc ASR để lấp chỗ thiếu. ASR tìm kiếm vẫn dùng transcript đã được nhóm khác
cung cấp; nút Transcribe All được ẩn và API xử lý ASR mặc định bị tắt.

- OCR còn pending/error/skipped không chặn server hoặc đóng gói.
- Thiếu FAISS/CLIP: OCR và lọc metadata vẫn hoạt động; truy vấn hình ảnh dùng
  kết quả OCR phù hợp khi có, hoặc trả kết quả rỗng kèm trạng thái partial.
- Thiếu FTS/facet: dùng OCR/metadata đã lưu trong SQLite (chậm hơn chỉ mục).
- Thiếu ASR/summary: tính năng tương ứng trả kết quả rỗng; tính năng khác tiếp tục.
- Thiếu ảnh: yêu cầu ảnh trả 404, không làm hỏng truy vấn tìm kiếm tiếp theo.
- Redis chưa sẵn sàng khi khởi động hybrid: dùng SQLite/FAISS tại máy.
- Kiểm tra an toàn của bước dedup và tính nhất quán vector/metadata lúc merge
  vẫn giữ nguyên để không xuất bản ánh xạ sai.

`CODE/sever_log.json` (đúng tên yêu cầu) ghi phần thiếu, số lượng và tối đa 30
ví dụ mỗi nhóm. Server quét inventory ở nền sau khi khởi động, không sửa dataset.
Trường `scan.status` cho biết đang quét/hoàn tất/lỗi; `observed_at` ghi thời điểm
phát hiện. Lỗi phát sinh khi truy vấn cũng được ghi; các lỗi lặp lại được gộp.
Có thể đổi vị trí bằng `SERVER_LOG_PATH`. `/data-health` cho biết trạng thái quét
và khả năng tìm kiếm đang dùng được. Báo cáo là ảnh chụp tại thời điểm quét;
khởi động lại server sau khi bổ sung dữ liệu/chỉ mục để kiểm tra lại đầy đủ.

Khởi động bằng lệnh UI hiện có. Khi muốn dùng ngay index mới còn đang xử lý:

```bash
AIC_USE_ACTIVE_DATASET=0 BACKEND=sqlite \
DATA_ROOT=/mnt/d/AIC_26/data/new_dataset_work/new_index \
KEYFRAMES_DIR=/mnt/d/AIC_26/data/extracted/keyframes \
MAPFILES_DIR=/mnt/d/AIC_26/data/extracted/map-keyframes \
VIDEOS_DIR=/mnt/d/AIC_26/data/videos_2 \
TEAM_VENV=/home/dtp/project/aic_local_prep/.venv_sys \
bash run_local_server_wsl.sh
```

Để yêu cầu kiểm tra nghiêm ngặt như trước, đặt `ALLOW_PARTIAL_DATA=0`.

## Redis cho index đã gộp

Sau khi nhập index vào Redis cổng 16380, dùng launcher cố định để tránh chọn
nhầm index hoặc Redis cũ:

```bash
cd /mnt/c/PERSONAL_PROJECT/AIC_26/CODE
bash run_merged_redis_wsl.sh
```

Launcher dùng hybrid: Redis cho OCR/ASR/summary, FAISS cho vector ảnh. Namespace
`aic:merged:20260924:` được nhập với `--text-only`, không chứa vector ảnh để tiết
kiệm RAM; không dùng namespace này với backend Redis thuần. ASR chỉ nhập kết quả
đã có, không nhận dạng lại âm thanh. Redis lưu AOF ở
`/mnt/d/AIC_26/data/new_dataset_work/redis_data` và tiếp tục chạy khi dừng UI.
