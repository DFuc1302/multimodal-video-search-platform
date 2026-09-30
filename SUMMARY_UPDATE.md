# Cập nhật summary cho UI thi đấu

Chạy trong Ubuntu/WSL:

```bash
cd /mnt/c/PERSONAL_PROJECT/AIC_26/CODE
bash run_final_hybrid_wsl.sh summary --allow-issues
```

`--allow-issues` bỏ qua đoạn lỗi đã liệt kê trong report, giữ các đoạn hợp lệ. Bỏ cờ này nếu muốn dừng khi có bất kỳ lỗi nguồn/ánh xạ nào. Dữ liệu kiểm tra ngày 25/09/2026 có 66 đoạn lỗi, 660 đoạn không có ảnh trong khoảng thời gian; không có lỗi ánh xạ nguồn. Không sửa JSON gốc.

Chỉ kiểm tra nguồn, chưa tạo vector hoặc kích hoạt:

```bash
bash run_final_hybrid_wsl.sh summary --audit-only
```

Nguồn mặc định: `filtered_data_all_873_moved/*/video_summaries_json` và `D:/AIC_26/data/classified_MS/*/video_summaries_json`. Nguồn khác có thể thêm bằng `--extra-summary-root /mnt/d/path/to/summaries`. JSON trong thư mục gốc hoặc thư mục con `video_summaries_json` được nhận. Chỉ nhận tên video L/M/S/N. Hai file cùng video nhưng nội dung khác nhau sẽ dừng để tránh chọn nhầm.

Quy tắc: M→10, S→11, N→12; L thuộc nhóm 09→10; các L khác giữ nhóm đã được duyệt qua thư mục phân loại. Không chạy lại CLIP/OCR/ASR, không di chuyển media. Chỉ tạo embedding văn bản summary bằng multilingual-e5-small (CPU mặc định; thêm `--device cuda` nếu muốn dùng GPU).

Mỗi lần chạy tạo một thư mục `final_search_index/summary_releases/<id>/` gồm manifest JSON, SQLite, vector FAISS khi build, report.json và unmapped_segments.json. Report liệt kê số video theo nhóm, lỗi từng đoạn và số đoạn thiếu ảnh. Bản mới chỉ được kích hoạt sau khi SQLite/FAISS qua kiểm tra; `active_summary.json` được thay nguyên tử. Bản cũ giữ nguyên. Nếu lỗi hoặc bị ngắt trước khi kích hoạt, UI tiếp tục dùng bản cũ. Các release chỉ audit không được kích hoạt.

Sau khi lệnh báo `Validated and activated`, nhấn Ctrl+C ở terminal server cũ, rồi:

```bash
bash run_final_hybrid_wsl.sh run
```

Mở http://localhost:8080, làm mới trang, chọn Summaries hoặc Summaries + Visual và nhóm 10/11/12. Summary thiếu keyframe vẫn có nội dung văn bản trong danh sách mở rộng. `/summary/status` báo số video theo nhóm. Không cần chạy `prepare` để cập nhật summary; summary dùng SQLite/FAISS của release mới, OCR/ASR vẫn dùng Redis theo cấu hình hybrid. Các tiến trình đang mở chỉ mục cũ cần khởi động lại.

Kiểm thử:

```bash
source /home/dtp/project/aic_local_prep/.venv_sys/bin/activate
python -m unittest discover -s . -p test_summary_integration.py -v
```

Kiểm thử sử dụng FAISS/SQLite thật với vector mẫu, giả lập bước embedding. Chưa thay thế kiểm tra chất lượng tìm kiếm bằng model trên toàn bộ dữ liệu.
