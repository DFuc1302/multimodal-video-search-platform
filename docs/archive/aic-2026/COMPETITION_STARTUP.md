# Khởi động khi thi và nộp bài độc lập

Trong Ubuntu/WSL, dừng phiên server cũ bằng Ctrl+C rồi chạy:

```bash
cd /mnt/c/PERSONAL_PROJECT/AIC_26/CODE
bash run_final_hybrid_wsl.sh run
```

Mở http://localhost:8080/#dres ngay sau khi giao diện báo cổng đã mở. Phần UI/nộp bài là tiến trình nhẹ ở cổng 8080; phần tìm kiếm tải riêng trên 127.0.0.1:8081. Tab DRES vẫn hoạt động khi tìm kiếm đang tải hoặc khởi động lỗi. Nếu cổng 8081 đang được dùng, đổi trước khi chạy bằng `SEARCH_SERVER_PORT=8082 bash run_final_hybrid_wsl.sh run`.

Để giảm chờ lúc thi:
- Chạy `prepare` chỉ khi cần nạp/cập nhật bộ dữ liệu Redis, hoàn tất trước giờ thi. Không chạy lại mỗi lần mở UI.
- Mở server trước giờ thi và giữ terminal chạy. CLIP/FAISS được tải và kiểm tra trước khi nút tìm kiếm bật.
- Redis tải song song với quá trình khởi tạo CLIP/FAISS; kiểm tra đúng nguồn dữ liệu và số lượng chỉ mục vẫn thực hiện trước khi cho tìm kiếm.
- Redis mặc định tiếp tục chạy sau khi đóng UI để lần mở lại không phải nạp snapshot. Redis vẫn chiếm RAM; khởi động lại máy vẫn cần tải lại dữ liệu. Có thể đặt `STOP_REDIS_ON_EXIT=1` trước lệnh run để giữ cách tắt Redis cùng server.
- Chế độ metadata lazy, FAISS mmap và chỉ tải CLIP text vẫn được giữ. Không tự chạy lại OCR/ASR/CLIP indexing khi mở server.

Nộp thủ công:
1. Nhập Video ID (hỗ trợ dạng `N001-V001` hoặc `N001_V001`).
2. Chọn **idx gốc (frame_idx)** hoặc **ID ảnh (n trong map-keyframes)**, rồi nhập số. ID ảnh được đổi bằng CSV, không nhầm với frame gốc.
3. Dùng `−/+` với bước tùy chọn. Với TRAKE, chỉnh số tại vị trí con trỏ trong danh sách.
4. Nhấn **Kiểm tra đáp án**, rồi **Mô phỏng nộp** hoặc bật chế độ nộp thật sau khi đăng nhập/chọn phiên thi.

Các số chưa có ánh xạ CSV sẽ được báo; không nội suy thời gian. Mỗi lần sửa số sẽ hủy bản kiểm tra cũ, tránh gửi nhầm đáp án. Chỉ bấm nộp mới gửi; tìm kiếm không tự nộp.

Query BTC:
- Dán nguyên văn vào ô **Đề bài / query của BTC** ở tab DRES.
- Chọn Visual, Visual + Summary hoặc Summary, bấm **Tìm đề này** khi tìm kiếm sẵn sàng.
- Query được giữ trong session của tab trình duyệt khi chuyển tab/tải lại trang. Query không được ghép vào payload nộp DRES.
- Khi tìm kiếm đang xử lý, có thể chuyển lại tab DRES để nộp thủ công.

Kiểm tra đã thực hiện: kiểm thử gateway/DRES giả lập, proxy video dạng Range, chuyển ID ảnh sang idx, bảo vệ gửi trùng, JavaScript và kiểm tra trình duyệt với dữ liệu map thực tế trong chế độ mô phỏng. Chưa đo thời gian cold-start toàn bộ CLIP/FAISS/Redis trên phiên thi thật; không cam kết số giây cố định.

TRAKE tự đọc `/mnt/d/AIC_26/data/aic26-b2-map-keyframes.zip`. Có thể đổi nguồn bằng `DRES_OFFICIAL_MAPS=/path/to/official-maps.zip bash run_final_hybrid_wsl.sh run`. Preview hiển thị idx local → frame_idx BTC và độ lệch thời gian. Không cần dựng lại search index cho thay đổi này.


### OCR/ASR trước CLIP

Với `run_final_hybrid_wsl.sh run`, server ưu tiên OCR/ASR: kiểm tra SQLite/Redis FTS và transcript trước, đặt `/ready.text_ready=true`, rồi mới mở metadata cache, FAISS và tải CLIP trong nền. Truy vấn OCR thuần không mở FAISS. UI mở tab OCR Search và tab Audio ngay ở mốc này. Tab OCR có request, trạng thái, kết quả và phân trang riêng nên có thể chạy song song với tìm Visual ở tab Frame Search. Visual, Visual + Summary, tìm tương tự và Video Match chỉ nhận truy vấn sau khi CLIP đã tải và qua phép thử tìm kiếm. Nếu CLIP lỗi, OCR/ASR đã xác minh vẫn dùng được. Tab Nộp DRES hoạt động độc lập từ đầu.

Không cần dựng lại chỉ mục. Dừng server đang chạy bằng Ctrl+C trong terminal của nó rồi chạy lại:

```bash
cd /mnt/c/PERSONAL_PROJECT/AIC_26/CODE
bash run_final_hybrid_wsl.sh run
```

Mặc định gateway chạy hai worker riêng: Text ở cổng nội bộ 8081 và Visual ở 8082. Kết quả đo cold-load trên bộ hiện tại: Redis text 0,44 giây/65 MB; FAISS 18,16 giây/3,14 GB; CLIP 6,32 giây/1,57 GB. Vì FAISS/CLIP là điểm nghẽn và rủi ro sập lớn nhất, chúng chỉ nằm trong worker Visual. Nếu worker này sập, kiểm tra và khởi động riêng mà không dừng OCR/ASR/DRES:

```bash
curl -s http://localhost:8080/workers
curl -s -X POST http://localhost:8080/workers/visual/restart
```

Kiểm thử mô phỏng CLIP đang tải/lỗi đã chạy qua API thật với SQLite FTS; chưa đo thời gian cold boot của toàn bộ dataset trên máy thi.
