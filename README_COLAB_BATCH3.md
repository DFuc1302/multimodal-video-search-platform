# Batch 3 — video đã giải nén trên Drive → dữ liệu tìm kiếm local

Notebook: `output/colab_batch3/AIC_Batch3_Colab.ipynb` (tính từ thư mục dự án).
Notebook chứa sẵn mã nguồn. Mở Google Colab → File → Upload notebook, chọn file này.
Chọn GPU runtime và chạy các ô từ trên xuống. Notebook hiện chọn `GROUPS = ALL_GROUPS`
vì cả 11 nhóm đã upload xong. Đầu vào vẫn là video rời; M04 đã hoàn tất sẽ được kiểm tra/bỏ qua.

### Theo dõi tiến trình trên Colab

Thanh tiến trình tự hiện dưới ô benchmark và xử lý: nhóm hoàn tất, video trích/lọc,
số frame đã trích, phần trăm lọc trùng, CLIP/YOLO/OCR, byte copy video, đóng gói ZIP,
CRC và SHA-256. Mỗi thanh hiển thị thời gian đã chạy; có tốc độ/ETA khi đủ dữ liệu.
Trích frame không đoán tổng số ảnh nên hiện bộ đếm frame và tốc độ thay vì phần trăm giả.
`stages` là 4 bước (trích/lọc → CLIP/YOLO/OCR → dựng/kiểm tra index → xuất/kiểm tra ZIP),
không phải phần trăm thời gian toàn bộ. Thanh nhóm chỉ hoàn tất sau kiểm tra ZIP.
Những nhóm đã xong từ notebook ngay trước bản bổ sung thanh tiến trình vẫn được
nhận diện khi recipe và nguồn khớp; thay đổi cấu hình xử lý vẫn bị từ chối khi resume.

### Nếu ô cài đặt cũ lỗi trên Python 3.13

Mở lại notebook mới hoặc thay toàn bộ ô 2 bằng nội dung `CODE/colab_batch3_setup.py`.
Không chạy lại lệnh cũ có `numpy<2` trên Python 3.13: NumPy 1.26 chỉ hỗ trợ Python 3.9–3.12.
Ô mới chọn NumPy 2.1–2.2 cho Python 3.13, cài wheel thay vì tự build,
giữ phiên bản torch/torchvision của Colab, rồi kiểm tra import và phép toán GPU.
Không cài ASR. Lỗi đầy đủ được lưu trong `/content/aic_batch3_install.log`;
`CalledProcessError` cuối ô chỉ là thông báo lệnh thất bại, không phải nguyên nhân chi tiết.
Sau khi ô 2 báo `Setup complete`, tiếp tục ô 3. Nếu Colab yêu cầu restart, restart
rồi chạy lại từ ô 1; dữ liệu video trên Drive không bị ảnh hưởng.

## Cấu hình A100 và mục tiêu 30 giờ — không gồm ASR

ASR do người khác xử lý. Notebook không cài model ASR, không đọc âm thanh và không
chạy nhận dạng tiếng nói trên bất kỳ nhóm nào; benchmark cũng không tính ASR.
Mặc định CLIP batch 64, YOLO chạy theo batch, OCR batch 32, pHash tối đa 8 CPU threads.
Cấu hình tốc độ dùng **1 frame/giây thay cho 2**, **OCR_ENHANCE=0** để bỏ lượt đọc lại
từng vùng. Giữ độ phân giải ảnh, model CLIP/YOLO và OCR tất cả frame còn lại.
Đổi lại có thể bỏ lỡ cảnh dưới 1 giây và giảm khả năng nhận chữ nhỏ/mờ/khó.
Muốn khôi phục mật độ/lượt đọc lại cũ: đặt FRAME_INTERVAL=0.5, OCR_ENHANCE=1 rồi benchmark lại.
Có fallback YOLO từng ảnh và OCR tách batch khi lỗi.
Riêng N031–N040 bật bộ lọc chỉ chữ trắng trên nền đen; nhóm M không áp bộ lọc này.
FFmpeg chỉ ghi log frame được chọn và tắt tính checksum pixel khi trích xuất;
map vẫn giữ frame ID gốc và timestamp thực. CRC/SHA-256 của gói xuất vẫn được kiểm tra.

Ô benchmark mới (report_version=2) ưu tiên số đo trọn nhóm đã hoàn tất từ file
`*_processed.timing.json`, sau khi kiểm tra ZIP/checksum, cấu hình và nguồn video.
Nhóm đã xong có 0 giờ công việc còn lại. Các nhóm M còn lại dùng tốc độ thực tế
của nhóm M đã xong (ví dụ M04); nhóm N được tính riêng, không dùng tốc độ M.
Nếu thiếu số đo trọn nhóm cùng loại, đo hai đoạn tối đa 1 phút và 10 phút để
ước tính riêng chi phí khởi động và tốc độ xử lý. Mẫu nhiễu không được suy ra tốc độ âm/0.
Bản cũ nhân cả thời gian nạp model của mẫu 2 phút với tổng thời lượng, có thể
phóng đại mạnh. Con số như 258,6 giờ vì vậy chưa phải số đo thời gian thực.

Báo cáo tách thời gian thực các nhóm đã xong, ước tính còn lại và tổng dự kiến.
Phần còn lại nhân hệ số dự phòng 1,5, cộng 4 giờ giả định cho truyền/gộp local;
tổng dự kiến cộng thời gian nhóm đã xong biết được và lần benchmark hiện tại.
Chưa tính benchmark cũ, retry, downtime; chưa đo tốc độ truyền/gộp trên máy bạn.
Dùng cùng GPU và cấu hình để đối chiếu. Nội dung, tỷ lệ ảnh trùng, OCR và Drive
có thể khác giữa các nhóm: đây vẫn là ước tính, không bảo đảm xong trong 30 giờ.
Thiếu timing nhóm đã xong thì báo chưa đủ dữ liệu kết luận tổng thời gian.
Báo cáo cũ được lưu lại với tên `budget_report_previous_*.json`.

Nếu ô 6 đang thực sự xử lý, không ngắt hoặc chạy thêm benchmark trên cùng GPU.
Nếu chỉ ô 5 dừng vì AssertionError: thay **ô 3 và ô 5** từ notebook mới, giữ
GROUPS=ALL_GROUPS, RUN_BENCHMARK=True; chạy ô 3 rồi ô 5. Không cần cài lại thư viện.
Sau khi xem báo cáo mới và qua kiểm tra ngân sách, chạy ô 6. Không đặt
ALLOW_OVER_BUDGET=True chỉ để coi cảnh báo là đã được giải quyết; tùy chọn đó
chỉ cho phép tiếp tục dù chưa chứng minh được mục tiêu thời gian.

## 1. Upload nguồn

Đưa video đã giải nén vào `MyDrive/DATA_AIC/batch 3/videos`.
Thư mục con được hỗ trợ; giữ nguyên tên video BTC, ví dụ `M01_V001.mp4` hoặc
`N031-V001.mp4`. Tên phải bắt đầu M01–M10 hoặc N031–N040, tiếp theo là `-` hoặc `_`.
Không upload cùng một video hai lần ở các thư mục khác nhau.
Hoàn tất upload các nhóm được chọn trước khi chạy. Không cần upload ZIP nguồn hay CODE lên Drive.

Notebook hỗ trợ đủ 11 nhóm; sửa GROUPS để chọn những nhóm đã upload hoàn tất.
Khi tất cả đã upload xong, đặt `GROUPS = ALL_GROUPS` để chạy/kiểm tra đủ 11 nhóm.
Không chạy hai phiên cùng nhóm. Kết quả nằm trong `batch 3/processed_data`:
11 file `*_processed.zip` và 11 file `*_processed.sha256.json`.

## 2. Những gì đã xử lý trên Colab

- Cắt frame với bước khoảng 1 giây theo FPS được probe từ video.
  Map ghi đúng số frame giải mã gốc và PTS thực; video VFR không bảo đảm khoảng cách
  thời gian đều 1 giây vì chọn theo số frame. Không hard-code FPS cho video M.
- Lọc pHash khoảng cách ≤ 4, so sánh các cặp ảnh liền kề ban đầu,
  giữ ảnh đầu chuỗi; map được lọc tương ứng và giữ nguyên số frame.
- CLIP `openai/clip-vit-large-patch14`, YOLO `yolov8m`, EasyOCR tiếng Việt/Anh.
  Nếu dataset cũ dùng CLIP khác, đổi CLIP_MODEL trong notebook cho khớp trước khi chạy.
- N031–N040: `OCR_WHITE_ON_BLACK_ONLY=1`, chỉ giữ chữ trắng trên nền đen
  bằng bộ lọc OCR hiện có của dự án. Nhóm M: `OCR_WHITE_ON_BLACK_ONLY=0`.
  Cả hai vẫn quét tất cả frame còn lại sau dedup; benchmark áp dụng cùng chính sách.
- Không chạy ASR. `asr_cache.json` là object rỗng `{}` để tương thích bundle;
  receipt ghi `asr_status=delegated_not_processed`, không coi đây là transcript đã hoàn tất.
- SQLite/FTS, FAISS, clip_text phục vụ tìm kiếm, keyframes, map và receipt.
- Kiểm tra số ảnh/map, độ đầy đủ OCR, ánh xạ FAISS, CRC ZIP và SHA-256.

ZIP xuất không chứa video, audio tạm hay cache tải model.
Video gốc trên Drive không bị sửa/xóa. Ổ tạm Colab chỉ chứa dữ liệu của một nhóm
và một video đang xử lý. Ảnh của một nhóm vẫn có thể lớn: cần theo dõi dung lượng.

Checkpoint ở cấp nhóm: nhóm hoàn tất được bỏ qua sau kiểm tra checksum và cấu hình;
nhóm bị ngắt giữa chừng sẽ chạy lại. Không có cam kết hoàn tất toàn bộ trong một phiên.
Một số model được tải lại nếu Colab cấp máy ảo mới. Giữ nguyên nguồn khi resume.
Nếu ZIP đã có nhưng thiếu checksum do ngắt lúc xuất, di chuyển cặp kết quả lỗi ra
ngoài `processed_data` rồi chạy lại nhóm đó; script không tự ghi đè gói có sẵn.

## 3. Tải về và nhập trên WSL

Tải toàn bộ **22 file** trong `processed_data` về
`D:\AIC_26\data\batch3_downloads`. Không cần tải `videos`.
Trong WSL, dùng môi trường Python đã chạy được UI hiện tại:

```bash
cd /mnt/c/PERSONAL_PROJECT/AIC_26/CODE
source /home/dtp/project/aic_local_prep/.venv_sys/bin/activate
python import_colab_batch3.py \
  --downloads /mnt/d/AIC_26/data/batch3_downloads \
  --shards /mnt/d/AIC_26/data/batch3_shards \
  --output /mnt/d/AIC_26/data/batch3_search_index
```

Lệnh tạo index riêng cho batch 3. Để gộp cùng dataset hiện có, dùng lệnh trên với
`--old-index /duong/dan/index_dang_dung` và chọn `--output` mới, chưa tồn tại.
`--old-index` phải trỏ đúng thư mục chứa `meta.db`, `faiss_index`, `serve_manifest.json`.
Manifest lưu ảnh ngoài của index cũ được kế thừa. Video ID trùng sẽ làm lệnh dừng
để tránh tự ý ghi đè. Không dùng `import_colab_batch2.py` vì map sau dedup có khoảng trống.

Máy local chỉ giải nén, kiểm tra, gộp vector/metadata và chuẩn bị phục vụ;
không chạy lại suy luận CLIP/YOLO/OCR. Gộp index vẫn cần CPU, RAM và dung lượng.
Giữ `batch3_shards` và các nguồn ảnh của dataset cũ; index gộp tham chiếu đến chúng.
Transcript batch 3 chưa có: người phụ trách ASR cần giữ nguyên tên video và trả
`asr_cache.json` đúng định dạng của dự án (key `video-name-v1::<tên file video>`,
segment cửa sổ 2 giây, cache_version=2). Gộp kết quả ASR riêng sau khi họ hoàn tất;
không cần trích frame hoặc chạy lại OCR để bổ sung transcript.

## 4. Chạy UI

Dừng UI đang dùng cổng 8080, mở terminal WSL mới và chạy:

```bash
cd /mnt/c/PERSONAL_PROJECT/AIC_26/CODE
AIC_USE_ACTIVE_DATASET=0 BACKEND=sqlite \
DATA_ROOT=/mnt/d/AIC_26/data/batch3_search_index \
KEYFRAMES_DIR=/mnt/d/AIC_26/data/batch3_search_index/keyframes \
MAPFILES_DIR=/mnt/d/AIC_26/data/batch3_search_index/map-keyframes \
VIDEOS_DIR=/mnt/d/AIC_26/data/batch3_shards/videos \
TEAM_VENV=/home/dtp/project/aic_local_prep/.venv_sys \
bash run_local_server_wsl.sh
```

Mở `http://localhost:8080`. Nếu dùng output khác, sửa DATA_ROOT/KEYFRAMES_DIR/MAPFILES_DIR.
Ảnh và map được resolver đọc từ shard. Chế độ SQLite dùng OCR/ASR trong bundle;
không cần nhập lại Redis để bắt đầu tìm kiếm. Không thay cấu hình UI đang lưu.
Video gốc không có trên máy thì không phát được video đầy đủ; xem keyframe và tìm kiếm vẫn dùng được.
Nếu cần video playback sau này, phải cấu hình thêm nguồn video có thể truy cập trên máy.

## Kiểm thử và giới hạn

Notebook cần Colab GPU và quyền mount Drive của bạn để kiểm thử toàn bộ.
Các kiểm thử local chỉ bao phủ logic đóng gói, đường dẫn ZIP, phân nhóm và trích frame.
Số video được kiểm kê theo file đã upload; chưa có manifest của BTC để chứng minh
bạn upload đủ mọi video bên trong từng nhóm.

Sau khi sửa mã nguồn, chạy `python build_colab_batch3_notebook.py` để cập nhật mã nhúng.


## Bản tăng tốc A100 (24/09)

Nếu đang dừng ở lỗi ngân sách: lấy ô **3, 5, 6** từ notebook mới, chạy 3 → 5 → 6
(ô 6 sau khi ước tính đạt ngân sách). Giữ `GROUPS = ALL_GROUPS`, `RUN_BENCHMARK = True`.
Không cần cài lại ô 2. Nếu ô 6 đang thực sự chạy, chờ nhóm/phiên hoàn tất rồi cập nhật;
không thay mã hoặc benchmark tranh GPU trong khi đang xử lý.

Profile `a100-fast` giữ interval 1 giây và model CLIP/YOLO, nhưng thu ảnh **đầu vào OCR**
xuống cạnh dài tối đa 1280px, có thể mất chữ nhỏ. Ảnh xuất giữ nguyên. Dò chữ theo lô 8
ảnh và nhận dạng theo lô 64 vùng giúp tránh dồn 32 ảnh lớn vào detector. N vẫn lọc
trên pixel ảnh gốc để chỉ giữ chữ trắng nền đen. Không bỏ OCR frame nào, không ASR.

Bỏ lượt backfill YOLO trên các kết quả rỗng của lần chạy mới; lỗi metadata vẫn bị
kiểm tra thiếu frame khi xác minh bundle. FAISS IndexIDMap2 tự cập nhật bảng tra cứu
ở add_with_ids nên không dựng lại toàn bộ mỗi batch. Tính SHA-256 ngay trong lượt copy video.

M04 và nhóm hoàn tất từ các bản notebook đã biết vẫn được kiểm tra checksum/nguồn/model/
interval trước khi bỏ qua. Receipt ghi rõ profile mới; nhóm cũ giữ OCR gốc, không ghi đè.
Benchmark v3 không dùng tốc độ M04 profile cũ để dự đoán profile mới; đo lại nhóm còn lại.
Không có kết quả đo A100 tại môi trường local; không khẳng định tăng 1,7 lần hay đạt 30h
trước khi đo bằng ô 5. Báo cáo trước được lưu lại để so sánh.
