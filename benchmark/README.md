# Bộ benchmark retrieval

Benchmark được đóng băng theo nội dung tệp qrels. Lưu qrels đã duyệt ở một bản phát hành riêng, ghi lại SHA-256 trong báo cáo, và không cập nhật nhãn test sau khi xem kết quả model. Mỗi truy vấn cần `query_id`, `unit`, `query_type`, `topic` và danh sách `relevant` do người gán nhãn xác nhận độc lập. Có thể thêm `relevant_intervals` cho nhãn thời gian:

```json
{
  "query_id": "<query-id-from-telemetry>",
  "unit": "frame",
  "query_type": "temporal",
  "topic": "health",
  "relevant": ["L01_V001/000123.jpg"],
  "relevant_intervals": [
    {"video_id": "L01_V001", "start_seconds": 12.0, "end_seconds": 18.5}
  ],
  "temporal_tolerance_seconds": 1.0
}
```

Để tính temporal recall/MRR, ranking trong log phải có item dạng `{ "file": "...", "video_id": "...", "timestamp_seconds": 14.2 }`. Log phiên bản hiện tại lưu đường dẫn frame dạng chuỗi, nên temporal metrics sẽ được bỏ qua cho đến khi ranking xuất dữ liệu timestamp. Không quy đổi frame number sang giây theo giả định FPS cố định.

Phân tầng đề xuất: `query_type` (`visual`, `text-in-scene`, `speech`, `summary`, `temporal`, `multimodal`) và `topic` theo chủ đề video. Không đưa thao tác shortlist của thí sinh thành nhãn đúng; shortlist chỉ có thể tạo ứng viên để chuyên gia duyệt. Tách hẳn split phát triển và test; không dùng query hoặc video test để fine-tune, expansion tuning hay chọn ngưỡng.

Chạy báo cáo hiện có từ log và qrels đã đóng băng:

```bash
python evaluate_search.py --logs runtime_logs --qrels benchmark/qrels-test.json --output benchmark/report.json
```

Báo cáo gồm Recall@K, MRR, P50/P95, lỗi, tỷ lệ không kết quả và phân tầng theo loại truy vấn/chủ đề. Chạy mỗi treatment trên cùng một query set và so sánh theo `configuration_id`. Đặt model, seed, index manifest và cờ tính năng vào metadata chạy để không gộp cấu hình khác nhau.

## Ma trận ablation cần ghi nhận

Giữ nguyên tập truy vấn, index, `top_k` và bước gộp kết quả. Đo độc lập BM25 và dense trước; sau đó thêm lần lượt RRF, OCR, ASR, summary, reranker và temporal context. Báo cáo cả chất lượng, latency và coverage từng tầng. Repo hiện có BM25 cho ASR/summary, dense CLIP và VLM reranker, nhưng chưa có giao diện thí nghiệm chuẩn để bật/tắt mọi nhánh độc lập; tránh gọi kết quả hybrid hiện tại là baseline BM25 thuần.

## Quyền riêng tư và học chủ động

Telemetry hiện ghi hash HMAC cho truy vấn/câu trả lời, mã phiên giả danh và lựa chọn shortlist, không ghi văn bản thô. Giữ mapping khóa riêng, giới hạn thời hạn lưu, chỉ xuất feedback đã khử định danh, và chỉ nhập vào tập train sau khi được duyệt. Không đọc feedback của test thành ground truth. Tạo split theo video trước khi dùng feedback để tránh rò các frame lân cận giữa train/test.
