# Nộp bài chung kết bằng tab DRES

Mở server như bình thường, chọn **Nộp DRES**. Tab dùng chung FastAPI và bộ dữ
liệu đang cấu hình. Không cần triển khai DRES hay cài thêm thư viện.

## Cách dùng

1. Để **Nộp thật lên DRES** tắt khi diễn tập.
2. Chọn Textual KIS, Video KIS, Q&A hoặc TRAKE.
3. Nhập Video ID đúng tên gốc, không có đuôi `.mp4`/`.mov`.
4. Nhập **frame_idx gốc**, không nhập số thứ tự file ảnh `n`.
   KIS và Q&A dùng một frame; Q&A thêm câu trả lời trên một dòng.
   TRAKE dùng danh sách frame tăng dần theo sự kiện, cùng một video.
5. Nhấn **Kiểm tra đáp án**, kiểm tra video, frame, PTS và mili giây.
6. Nhấn **Mô phỏng nộp**. Mô phỏng không gửi request nào tới DRES.

Có thể chọn frame ở Frame Search rồi sang tab DRES và nhấn **Lấy từ frame đã
chọn**. KIS/Q&A yêu cầu danh sách chỉ có một frame.

Khi thi thật: đăng nhập trong tab, chọn evaluation ACTIVE, bật **Nộp thật lên
DRES**, kiểm tra lại đáp án, xác nhận tên câu trong bản xem trước rồi nhấn
**Nộp đáp án lên DRES**. Bản xem trước hết hạn sau 2 phút; đổi nội dung sẽ vô
hiệu hóa bản xem trước. Server đọc lại câu đang thi trước mỗi lần gửi và gắn
`taskName` vào answerSet để hạn chế nộp nhầm khi chuyển câu.

Video KIS không được chụp/ghi video đề bài để đưa vào công cụ.

## Ánh xạ và giới hạn

- Đọc CSV đúng nguồn của video qua dataset/federation/external storage.
- `time_ms = round_half_up(pts_time * 1000)`. Không suy thời gian bằng FPS cố
  định, tên ảnh hay khoảng cách keyframe.
- KIS: `mediaItemName`, `start`, `end`; hai thời điểm bằng frame đã chọn.
- Q&A: `QA-<ANSWER>-<VIDEO_ID>-<TIME(ms)>`.
- TRAKE: `TR-<VIDEO_ID>-<FRAME_ID1>,<FRAME_ID2>,...`.
- Giữ cách viết Video ID theo tên CSV, gồm dấu gạch nối/gạch dưới.
- Frame không có trong CSV hoặc có nhiều `pts_time` mâu thuẫn bị chặn. Chưa
  hỗ trợ frame tùy ý ngoài bảng keyframe: cần xác minh timestamp từ video gốc
  trước khi mở rộng. Không tự sửa dữ liệu nguồn.
- Câu QA có dấu `-` được giữ nguyên và cảnh báo; cần xác nhận quy tắc phân cách
  với BTC. Giới hạn 2.000 ký tự là giới hạn ứng dụng, không phải quy định PDF.

## Kết nối và lịch sử

Mật khẩu chỉ dùng cho request đăng nhập, không lưu file. Session DRES giữ trong
bộ nhớ server. Browser giữ mã kết nối nội bộ trong sessionStorage của tab;
khởi động lại server cần đăng nhập lại. Không có mật khẩu mặc định trong code.
Server DRES phải dùng HTTPS; không tự theo redirect mang credentials.

Lịch sử nộp thật nằm ở `CODE/runtime/dres_history.sqlite`, gồm đáp án và phản
hồi, không gồm mật khẩu/session. Đổi vị trí qua `DRES_HISTORY_PATH` nếu cần.
Không xóa lịch sử giữa cuộc thi: lịch sử giúp chống nộp trùng sau khi khởi động
lại. Dùng một tiến trình server thông thường, không dùng nhiều worker vì
session và bản xem trước nằm trong bộ nhớ tiến trình.

- **DRES đã nhận** không có nghĩa đúng. Xem `submission`/verdict.
- **Chưa rõ kết quả** hoặc **Đang gửi / chưa rõ kết quả**: kiểm tra lịch sử
  DRES/BTC. Cùng đáp án trong câu đó bị chặn; không tự gửi lại.
- Request bị từ chối rõ ràng (HTTP 400/401/403/404/412) có thể thử lại sau khi
  xử lý nguyên nhân và tạo bản xem trước mới.
- Lịch sử mô phỏng chỉ ở trang hiện tại, không chiếm khóa chống trùng nộp thật.
- HTTP 401 khi đăng nhập: kiểm tra tài khoản/mật khẩu và trạng thái với BTC.

## Kiểm thử

Trong Python environment của dự án:

```sh
python -m unittest test_dres -v
python -m unittest discover -v
node test_ui.cjs
python audit_dres_frames.py --data-root /mnt/d/AIC_26/data/merged_search_index --output ../output/dres_frame_audit.json
```

Test DRES dùng transport giả lập, không gửi đáp án lên BTC. Cần phiên tập huấn
ACTIVE để xác nhận `taskName`, `start=end`, quy tắc QA và verdict của bản DRES
BTC đang dùng.

Tham chiếu: `HD-ChungKet-2026.pdf`, trang 4–6;
https://github.com/dres-dev/Client-Examples ;
https://raw.githubusercontent.com/dres-dev/DRES/master/doc/oas-client.json .
