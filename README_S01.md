# S01: pipeline WSL, GPU RTX 4070

S01 là video đua xe đạp theo note dataset. Pipeline chỉ chọn mã S01-Vxxx
(hoặc S01_Vxxx), không lấy video giao thông Nxxx. Tái sử dụng pipeline hiện có:
pHash dedup → CLIP/YOLO → EasyOCR tiếng Việt/Anh → Whisper ASR → SQLite/FAISS.
Không tạo bản tóm tắt nội dung bằng LLM; đầu ra tìm kiếm gồm hình ảnh, OCR và ASR.

## Cài thư viện

```bash
source ~/.venvs/aic-keyframes-gpu/bin/activate
cd /mnt/c/PERSONAL_PROJECT/AIC_26/CODE
python -m pip install -r requirements_s01.txt
python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'
```

CUDA phải hiển thị `True`. Cần ffmpeg và ổ D được mount đúng tại `/mnt/d`.
Model tải lần đầu cần internet. Cache model mới đặt dưới output trên ổ D.

## Chạy toàn bộ

```bash
python run_s01_pipeline.py \
  --extracted-dir /mnt/d/AIC_26/data/extracted \
  --videos-dir /mnt/d/AIC_26/data/videos_2 \
  --output-dir /mnt/d/AIC_26/data/s01_processed \
  --workers 4 --batch-size 256 \
  --clip-batch-size 8 --ocr-batch-size 8
```

Chạy lại cùng lệnh để tiếp tục. Giữ nguyên threshold/model cho cùng output.
Thư mục đích phải tách riêng nguồn và dataset cũ. Script sử dụng hardlink JPEG
khi cùng filesystem, nếu không hỗ trợ thì copy. Xóa ảnh trùng chỉ xóa liên kết
trong bộ S01, không xóa ảnh gốc trong extracted và không sửa video nguồn.

Map giữ nguyên `n` và `frame_idx` gốc, lọc các dòng không còn ảnh sau dedup.
Dedup chỉ giữ frame đầu tiên của mỗi chuỗi trùng liên tiếp từ 2 ảnh trở lên.
So khớp từng cặp liền kề theo pHash và threshold; không so toàn video với nhau.
Các file có thể là `001.jpg`, `004.jpg`, ...; không đánh số lại sau khử trùng.

## Chạy từng bước

```bash
python run_s01_pipeline.py --stage prepare
python run_s01_pipeline.py --stage dedup
python run_s01_pipeline.py --stage index
python run_s01_pipeline.py --stage asr
python run_s01_pipeline.py --stage bundle
```

Mỗi lệnh dùng cùng các đường dẫn mặc định như ví dụ trên. Nếu thay đường dẫn,
phải truyền lại các tham số đó cho từng lệnh, bao gồm lệnh mở server.
`prepare` và `dedup` có checkpoint theo video. CLIP/OCR/ASR dùng cơ chế resume
của pipeline cũ. Bundle chỉ thành công khi kiểm tra SQLite/FAISS/OCR đạt yêu cầu.
Lỗi trả exit code khác 0; xem logs trước khi tiếp tục.

## Tìm kiếm chung dataset cũ và S01

```bash
python run_s01_pipeline.py --stage serve --port 8081
```

Mở http://127.0.0.1:8081. Dataset cũ mặc định lấy từ
`/home/dtp/project/aic_local_prep/search_index_ocr`; thay bằng `--old-data-root`
nếu dùng nơi khác. Code đọc tên model CLIP từ manifest cũ để giữ cùng không gian
vector. Tìm kiếm dùng federation SQLite/FAISS, không cần nhập lại Redis.
Nếu muốn phát video gốc của dataset cũ, truyền đúng `--old-videos-dir`.

## Đầu ra

```text
/mnt/d/AIC_26/data/s01_processed/
  search_index/
    keyframes/
      05_The_thao_dua_xe_dap_va_thi_dau_duong_pho/
        keyframes/S01-V001/001.jpg
    map-keyframes/S01-V001.csv
    transcripts_asr/S01-V001.json
    meta.db
    faiss_index/
    asr_cache.json
    clip_text/
    serve_manifest.json
  s01_videos.json                # danh sách S01, đọc trực tiếp video nguồn
  model_cache/
  pipeline_state.json
  search_roots.json
  logs/index.log
  logs/asr.log
  logs/bundle.log
```

Pipeline xử lý tất cả keyframe còn lại; không giảm số ảnh bằng sampling thêm.
Không cần symlink hay quyền tạo liên kết thư mục trên ổ D. Thư mục dataset/videos
rỗng do phiên bản cũ tạo có thể để nguyên; phiên bản mới không sử dụng chúng.
Nhiều video dài hàng giờ nên OCR/ASR có thể chạy lâu. Không chạy đồng thời
extract/dedup khác trên bộ S01 này khi đang build index.
