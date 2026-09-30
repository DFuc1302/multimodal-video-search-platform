# Server chỉ nộp bài DRES

Server này chỉ tải giao diện nộp bài, bảng ánh xạ frame và lịch sử nộp. Nó không
khởi động Redis, FAISS, CLIP, OCR, ASR hoặc server tìm kiếm.

## Chạy trực tiếp trên Windows

Nhấp đúp `CHAY_APP_NOP_BAI_WINDOWS.cmd`. Lần đầu chương trình sẽ tạo môi trường
Python nhẹ và cài hai thư viện web cần thiết. Các lần sau sẽ mở ngay. Truy cập
`http://localhost:8091` và giữ cửa sổ lệnh đang chạy.

Hoặc chạy trong PowerShell:

```powershell
cd C:\PERSONAL_PROJECT\AIC_26\CODE
.\run_submission_server_windows.ps1
```

Đây là cách nên dùng khi WSL mất kết nối Internet.

## Chạy trong WSL

Trong WSL chạy:

```bash
cd /mnt/c/PERSONAL_PROJECT/AIC_26/CODE
bash run_submission_server_wsl.sh
```

Sau đó mở `http://localhost:8080`. Giao diện tự đăng nhập bằng tài khoản được
cấu hình cục bộ trong `.dres_credentials.env`, nên không cần nhập lại mật khẩu.

Mặc định luôn ở chế độ mô phỏng. Muốn nộp thật phải chọn phiên ACTIVE, bật
**Nộp thật lên DRES**, nhấn **Kiểm tra đáp án**, kiểm tra payload rồi mới nhấn
**Nộp đáp án lên DRES**. Server giữ nguyên cơ chế preview hết hạn sau 2 phút,
đối chiếu lại câu hiện tại và chặn gửi trùng.

Không chia sẻ file `.dres_credentials.env`. Nếu đổi mật khẩu, sửa file này rồi
khởi động lại server.
