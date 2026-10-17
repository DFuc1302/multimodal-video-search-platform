# Cập nhật tìm kiếm theo báo cáo hạn chế

Các thay đổi nằm trong `CODE`. CLIP/FAISS, OCR Search, ASR, Video Match, Find Similar, KIS, VQA và xuất Temporal Alignment tiếp tục dùng các luồng hiện có. Cần khởi động lại backend để nạp mã mới.

## ASR: sửa lỗi và giảm chi phí ánh xạ

`/asr/search` sử dụng timeline từ `frame_locator` và CSV `n,pts_time,frame_idx`, thay cho hai lời gọi hàm không tồn tại. Khoảng thời gian là `[start,end)`, nên frame tại đúng cuối một đoạn thuộc đoạn kế tiếp. Phản hồi giữ `results`, `frames`, `total`, đồng thời trả `offset`, `limit`, `next_offset`, `has_more`, `video_id` và `mapping_mode`. Tham số `top_k` cũ vẫn được chấp nhận khi chưa có `limit`.

- Giao diện ASR mặc định 100 segment/trang và chỉ yêu cầu trang đang xem.
- Nhánh cache JSON loại segment trùng khi nhiều điều kiện cùng khớp, sắp xếp ổn định và giữ ngữ nghĩa AND ở cấp video như FTS.
- Federated mode dùng cache hợp nhất thay vì vô tình tìm FTS của một nguồn duy nhất. Đây vẫn là nhánh quét cache; chưa có FTS liên nguồn.
- Timeline được cache có giới hạn, có khóa bảo vệ khi nhiều request cùng chạy; tìm khoảng thời gian dùng tìm kiếm nhị phân.
- Serving read-only tin vào locator đã kiểm tra lúc dựng bundle, không kiểm tra hàng nghìn file trên USB ở mỗi lần nạp timeline. `/image` vẫn kiểm tra file khi phục vụ ảnh. Nếu ổ ngoài offline, transcript vẫn trả được nhưng ảnh có thể không mở được.

Ví dụ:

```bash
curl -X POST http://localhost:8080/asr/search \
  -H 'Content-Type: application/json' \
  -d '{"phrase":"thành phố","operator":"and","offset":0,"limit":100}'
```

## VLM và timeline

VLM là bước tùy chọn sau lọc và loại frame gần trùng. Chỉ top-N được chấm lại; các kết quả còn lại giữ thứ tự. Lỗi, timeout, mô hình chưa cấu hình hoặc tác vụ VLM đang bận đều trả lại thứ tự tìm kiếm ban đầu. Kết quả và trạng thái chấm lại được trả trong `reranking`; điểm gốc `score` vẫn được giữ, điểm bổ sung ở `vlm_score` và `rerank_score`.

Backend dùng một checkpoint **Qwen2-VL Instruct đã có trên máy**. Không tự tải trọng số, không gọi dịch vụ bên ngoài. Cách truyền nhiều ảnh và chat template dựa trên [tài liệu Transformers 4.46.3](https://huggingface.co/docs/transformers/v4.46.3/en/model_doc/qwen2_vl), phù hợp với phiên bản khóa trong dự án.

```bash
cd /home/dtp/project/CODE
VLM_MODEL_PATH=/duong/dan/checkpoint/Qwen2-VL-Instruct \
VLM_DEVICE=cuda:0 VLM_TOP_N=12 VLM_TIMEOUT_MS=5000 \
./run_central_search_server_wsl.sh
```

`VLM_MODEL_PATH` phải là thư mục trọng số, tokenizer và processor hoàn chỉnh. Chọn `VLM_DEVICE=cpu` nếu cần chạy CPU. Lần nạp đầu có thể vượt ngân sách và trả lại xếp hạng gốc; request sau có thể dùng mô hình đã nạp. Chỉ có một tác vụ VLM đang xử lý, không xếp hàng vô hạn. Kiểm tra độ trễ trên phần cứng thực tế trước khi tăng top-N.

| Cấu hình | Mặc định | Tác dụng |
|---|---|---|
| `VLM_MODEL_PATH` | trống | Chưa bật VLM khi chưa có checkpoint |
| `VLM_DEVICE` | `cpu` | Thiết bị suy luận |
| `VLM_TOP_N` | `12` | Tối đa 100 ứng viên được chấm lại |
| `VLM_TIMEOUT_MS` | `3000` | Ngân sách chờ chấm lại của mỗi request |
| `VLM_WEIGHT` | `0.65` | Trọng số VLM so với điểm tìm kiếm chuẩn hóa |

Trên giao diện, chọn **Refine relationships**; tùy chọn này chỉ mở khi đã cấu hình thư mục mô hình. **Follow timeline** cho VLM xem tối đa 3 keyframe trong cửa sổ ±10 giây, có thứ tự và nhãn thời gian. Lightbox cũng có **Timeline ±10 seconds** để duyệt tối đa 9 ảnh trước/sau. API `/frame/timeline` cho phép `window_seconds` từ 0.5 đến 120 và `max_frames` từ 3 đến 31. Nếu không có CSV, `mapping_mode=inferred_interval` cho biết thời gian chỉ được suy ra, không phải mốc đã xác minh.

```bash
curl --get http://localhost:8080/search \
  --data-urlencode 'query=người đặt chiếc cốc lên bàn rồi quay đi' \
  --data-urlencode 'rerank=true' --data-urlencode 'timeline=true' \
  --data-urlencode 'top_k=100'
```

Điểm Yes/No của VLM là tín hiệu xếp hạng, không phải xác suất đã hiệu chuẩn. Cách này bổ sung ngữ cảnh thời gian cho top-N; chưa phải bộ theo dõi vật thể liên tục và không thể phục hồi một đáp án bị bỏ khỏi tập ứng viên ban đầu. Video Match tiếp tục chạy luồng CLIP nhanh như trước.

## OCR: nhận dạng theo vùng và nâng version trên bản sao

Pipeline mới giữ ảnh gốc cho bước đọc các vùng chữ nhỏ/độ tin cậy thấp, bổ sung các tile chồng nhau để tìm chữ lượt đầu bỏ sót, tăng tương phản CLAHE và phóng lớn có giới hạn. Các box được quy đổi về tọa độ chuẩn hóa trên toàn ảnh. Kết quả trùng hoặc mảnh chữ nằm trong một dòng được gộp; Unicode được chuẩn hóa NFC, bỏ ký tự vô hình và thu gọn khoảng trắng. FTS vẫn có cả chữ có dấu và dạng bỏ dấu, bao gồm `đ → d`.

Version OCR gồm tên recipe và hash cấu hình: engine, phiên bản thư viện engine, ngôn ngữ, giới hạn ảnh, số vùng và chế độ tăng cường. Cùng cấu hình trên các shard tạo cùng version. Dữ liệu legacy không bị gắn nhãn hoàn tất của pipeline mới nếu chưa nhận dạng lại. Với frame lỗi, version ghi nhận lần thử nhưng trạng thái vẫn là `error`; chúng không được coi là đã nâng xong.

| Cấu hình | Mặc định | Tác dụng |
|---|---|---|
| `OCR_ENHANCE` | `1` | Bật đọc bổ sung theo vùng |
| `OCR_REGION_LIMIT` | `8` | Tối đa 12 vùng bổ sung mỗi ảnh |
| `OCR_MAX_SIDE` | `0` | Không thu nhỏ lượt cơ sở nếu không đặt biến |
| `OCR_PIPELINE_VERSION` | `ocr-regions-v3` | Tiền tố; hash recipe được nối tự động |

Đọc nhiều vùng tăng thời gian xử lý offline; không chạy OCR mới trong tìm kiếm read-only. Có thể đặt `OCR_ENHANCE=0` để dùng tốc độ của luồng cơ sở. Khi đổi các thiết lập chất lượng, version thay đổi và cần nhận dạng lại dữ liệu muốn đồng nhất.

Công cụ sau tạo bản sao SQLite độc lập, dùng lại ảnh/vector/ASR bằng symlink, rồi chỉ cập nhật OCR và FTS ở bản sao. Nguồn phải được giữ nguyên và còn truy cập được trong quá trình nâng cấp. Output cần đủ dung lượng cho SQLite và các file tạm; không dùng thư mục đang phục vụ.

```bash
cd /home/dtp/project
# Xem kế hoạch; chưa chép hoặc nhận dạng ảnh.
.venv_rebuild/bin/python CODE/reindex_ocr.py \
  --source data/search_index --output data/search_index.ocr-v3

# Chạy nâng cấp. Dùng cùng cấu hình khi tiếp tục hoặc xử lý các shard khác.
OCR_GPU=1 OCR_MAX_SIDE=960 OCR_REGION_LIMIT=8 \
.venv_rebuild/bin/python CODE/reindex_ocr.py \
  --source data/search_index --output data/search_index.ocr-v3 --apply

# Tiếp tục nếu đã staging xong nhưng nhận dạng bị gián đoạn.
OCR_GPU=1 OCR_MAX_SIDE=960 OCR_REGION_LIMIT=8 \
.venv_rebuild/bin/python CODE/reindex_ocr.py \
  --output data/search_index.ocr-v3 --resume --apply
```

`ocr_upgrade_report.json` chỉ báo `ready_for_bundle=true` khi mọi frame có version đích và trạng thái `done`, `no_text` hoặc `reused`; các frame thiếu ảnh/lỗi cần xử lý trước. Công cụ rebuild và kiểm tra toàn vẹn FTS. Nó không tự thay serving index và không tạo serving manifest cho dữ liệu còn dở.

Sau khi báo cáo đạt, dựng lại bundle và kiểm tra ở cổng riêng:

```bash
.venv_rebuild/bin/python CODE/prepare_serve_bundle.py \
  --data-root data/search_index.ocr-v3 --skip-clip-text-export

cd CODE
CENTRAL_SEARCH_ROOT=/home/dtp/project/data/search_index.ocr-v3 \
SERVER_PORT=18088 ./run_central_search_server_wsl.sh
```

## Log ẩn danh và đánh giá theo phiên

Mặc định log ở `/home/dtp/project/runtime_logs`, tách khỏi serving dataset. Đặt `SEARCH_LOG_DIR` để đổi thư mục, hoặc `SEARCH_LOG_ENABLED=0` để tắt. Trong Settings, **Start new competition session** tạo mã phiên ngẫu nhiên mới và xóa shortlist hiện tại; lịch sử Saved Queries vẫn giữ nguyên.

- Log ghi endpoint, fingerprint query/bộ lọc, thứ tự kết quả, trạng thái, độ trễ backend, cấu hình, shortlist và đáp án đã xuất.
- Query, bộ lọc dạng chữ, tên câu hỏi và nội dung đáp án được HMAC-SHA256 bằng khóa riêng; không lưu nguyên văn, IP, cookie hay danh tính người dùng. Log không thể tự khôi phục câu tìm kiếm để chạy lại. Saved Queries trong trình duyệt vẫn có nội dung như chức năng cũ.
- File/frame_idx, thứ tự shortlist và trạng thái ánh xạ được giữ để đối chiếu; nội dung tự do của đáp án chỉ lưu fingerprint.
- Các phản hồi tìm kiếm có header `X-Search-Id`, `X-Query-Id`, `X-Search-Session`. Frontend gắn thao tác chọn/xuất với lượt tìm đã tạo ra ảnh.
- Log JSONL tách theo ngày UTC và xoay file khi khoảng 20 MB. Khóa `.hmac-key` và chỉ mục liên kết `.receipts.db` chỉ nằm cục bộ. Giữ khóa nếu muốn so khớp fingerprint giữa các lần khởi động; không đưa khóa vào báo cáo chia sẻ.
- Lỗi ghi log không làm endpoint tìm kiếm thất bại. `/evaluation/status` cho biết log có bật và lần ghi gần nhất có lỗi hay không. Launcher tắt access log URL của Uvicorn để không ghi lại query nguyên văn. Nếu tự chạy Uvicorn, dùng `--no-access-log`; reverse proxy riêng cần có chính sách tương ứng.

Tạo file đáp án chuẩn độc lập, ví dụ `qrels.json`:

```json
[
  {
    "query_id": "hash X-Query-Id hoặc query_id từ log",
    "unit": "frame",
    "relevant": ["L01_V001/000123.jpg", "L01_V001/000124.jpg"]
  }
]
```

Có thể thêm `session_id` để áp dụng đáp án chuẩn cho riêng một phiên. Đơn vị `frame` dùng đường dẫn ảnh tương đối; `video` dùng mã như `L01_V001`; `segment` của ASR dùng định dạng `L01_V001@2.00-4.00`. Các ID phải đúng như trường `ranking` trong log, không nhầm số keyframe với `frame_idx` gốc.

```bash
cd /home/dtp/project
.venv_rebuild/bin/python CODE/evaluate_search.py \
  --logs runtime_logs --qrels qrels.json --output evaluation-report.json
```

Báo cáo tính Recall@1/5/10/50/100, MRR trên thứ tự đã trả, độ trễ p50/p95 và số request lỗi, tách theo phiên, đơn vị kết quả và cấu hình. Recall@K là số đáp án chuẩn khác nhau xuất hiện trong top-K chia tổng số đáp án chuẩn; MRR dùng vị trí đáp án chuẩn đầu tiên. Các trang ASR được ghép theo offset, không chấm chuỗi bị khuyết trang. Lượt tìm lặp lại dùng kết quả thành công mới nhất để tránh làm tăng trọng số một câu. Không có qrels thì số đo chất lượng để trống; shortlist và đáp án tự xuất không bị tự coi là ground truth.

## Kiểm chứng và giới hạn hiện tại

Đã kiểm thử ASR HTTP → FTS/cache → CSV → ảnh thật, AND/OR, phân trang, không tìm thấy, ảnh từ nguồn ngoài, filter trước rerank, dự phòng timeout/lỗi VLM, cache VLM, OCR batch → FTS, bản sao nâng OCR, quyền riêng tư và tính đúng Recall/MRR. Bộ unittest hiện có 99 test đạt. Kiểm thử trình duyệt cũng đạt cho tìm kiếm, phân trang, giữ thứ tự rerank khi nhóm video, chọn/xuất CSV, ASR, timeline và tạo phiên mới.

Trên chỉ mục hiện có: 199.228 segment ở 865 video có lời thoại; cache chứa 873 video, gồm 8 video không có segment. Truy vấn mẫu trả HTTP 200 và 4 keyframe cho mỗi đoạn 2 giây; sau tối ưu, đo khoảng 35 ms lượt đầu và 7 ms lượt lặp trong phép thử cục bộ. Đây là smoke test một truy vấn, không phải benchmark toàn bộ corpus hay bảo đảm độ trễ.

Checkpoint VLM chưa có trong máy ở lần sửa này nên chưa đo chất lượng hoặc tốc độ suy luận VLM thật. OCR mới đã chạy với mô hình EasyOCR cục bộ trên dữ liệu mẫu; chưa tái nhận dạng toàn bộ corpus và chưa có bộ đáp án chuẩn để kết luận Recall/MRR tăng bao nhiêu.

```bash
PYTHONPATH=CODE .venv_rebuild/bin/python -m unittest discover -s tests -p 'test_*.py'
# Khi môi trường Node có Playwright và Chromium:
node tests/test_ui_browser.cjs
# Có thể truyền đường dẫn package Playwright và trình duyệt đã cài ở tham số 1, 2.
```
