"""Build a self-contained Colab notebook with this project's preprocessing code."""
import base64
import io
import json
from pathlib import Path
import uuid
import zipfile

CODE = Path(__file__).resolve().parent
OUT = CODE.parent / "output" / "colab_batch3"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", zipfile.ZIP_DEFLATED) as z:
        for path in sorted(CODE.glob("*.py")):
            if not path.name.startswith(("test_", "build_colab_batch3_notebook")):
                z.write(path, path.name)
    encoded = base64.b64encode(stream.getvalue()).decode()
    cells = []

    def cell(kind, source, metadata=None):
        value = {"cell_type": kind, "id": uuid.uuid4().hex[:8],
                 "metadata": metadata or {}, "source": source.splitlines(keepends=True)}
        if kind == "code":
            value.update(execution_count=None, outputs=[])
        cells.append(value)

    cell("markdown", """# AIC — Batch 3: M01–M10 và N031–N040
Video đã giải nén → keyframes + map frame gốc + lọc trùng pHash + CLIP/YOLO/OCR.
**ASR do người khác xử lý; notebook này không chạy ASR cho bất kỳ nhóm nào.**
Mặc định hiện chọn **toàn bộ 11 nhóm đã upload xong**. Mỗi nhóm tạo một ZIP dữ liệu,
**không chứa video**, lưu trong `DATA_AIC/batch 3/processed_data`.

Cấu hình ưu tiên tốc độ cho mục tiêu **30 giờ trên A100**: 1 frame/giây, tắt lượt
OCR tăng cường đọc lại từng vùng, giữ nguyên ảnh xuất và model CLIP/YOLO.
Profile `a100-fast` thu nhỏ **ảnh đầu vào OCR** tới cạnh dài tối đa 1280px; chữ rất nhỏ
có thể bị bỏ sót. Dò chữ theo lô 8 ảnh, nhận dạng theo lô 64 vùng; không dò YOLO lần
hai trên ảnh đã có kết quả rỗng. FAISS không dựng lại bảng tra cứu sau mỗi lô.
Có thể bỏ lỡ cảnh dưới 1 giây và chữ khó nhận hơn cấu hình trước. Không bỏ bước OCR.

Chọn **Runtime → Change runtime type → GPU**. Chạy các ô theo thứ tự.
Notebook đã chứa mã nguồn cần thiết, không cần upload thêm thư mục CODE.

**Đầu vào:** upload xong video của các nhóm được chọn vào `DATA_AIC/batch 3/videos` trước khi chạy.
Nhóm đã xử lý xong sẽ được kiểm tra và bỏ qua nếu nguồn/cấu hình không đổi.
Có thể giữ thư mục con. Giữ nguyên tên như `M01_V001.mp4`, `M01-V001.mp4`, `N031_V001.mp4`.
Notebook sẽ báo lỗi nếu gặp tên không phân nhóm được hoặc trùng ID.

**Chạy tiếp:** các nhóm hoàn tất được kiểm tra SHA-256 và bỏ qua. Nhóm đang chạy dở
sẽ làm lại nếu phiên bị ngắt. Không chạy hai phiên trên cùng nhóm/processed_data.
Không thay đổi video hoặc cấu hình khi chạy tiếp. pHash chỉ lọc trong từng video,
giữ frame đầu của chuỗi ảnh liền kề có khoảng cách ≤ 4; không đánh lại số frame.
""")
    cell("markdown", "## 1. Mount Google Drive và chọn cấu hình")
    cell("code", """from google.colab import drive
drive.mount('/content/drive')
from pathlib import Path
BATCH_ROOT = Path('/content/drive/MyDrive/DATA_AIC/batch 3')
CLIP_MODEL = 'openai/clip-vit-large-patch14'  # Phải giống serve_manifest.json của dataset cũ nếu gộp.
CLIP_BATCH_SIZE = 64  # Cấu hình A100; nếu OOM giảm 32/16.
OCR_BATCH_SIZE = 32   # Nếu OOM giảm 16/8.
FRAME_INTERVAL = 1.0  # 1 frame/giây. Đặt 0.5 nếu cần giữ mật độ cũ, rồi benchmark lại.
SPEED_PROFILE = 'a100-fast'  # OCR input <=1280px, detector batches 8, recognition batches 64.
OCR_ENHANCE = 0       # Tắt lượt đọc lại từng vùng; N vẫn lọc chữ trắng nền đen.
TARGET_HOURS = 30
RUN_BENCHMARK = True
ALLOW_OVER_BUDGET = False
ALL_GROUPS = [f'Videos_M{i:02}.zip' for i in range(1, 11)] + ['Video_N031-N040.zip']
GROUPS = ALL_GROUPS  # Tất cả đã upload xong; M04 hoàn tất sẽ được kiểm tra/bỏ qua.
# GROUPS là tên nhóm xử lý, KHÔNG phải yêu cầu upload ZIP.
# Chỉ chọn các nhóm đã upload xong. Khi đủ cả 11 nhóm, có thể đặt GROUPS = ALL_GROUPS.
assert (BATCH_ROOT / 'videos').is_dir(), BATCH_ROOT / 'videos'
""")
    cell("markdown", "## 2. Cài thư viện và kiểm tra GPU\nTự chọn NumPy theo Python (Python 3.13 dùng NumPy 2), giữ PyTorch/CUDA sẵn có của Colab. Chỉ cài wheel, không biên dịch gói cũ. Log đầy đủ: `/content/aic_batch3_install.log`. Kiểm tra import và GPU trong tiến trình mới trước khi tiếp tục. Nếu Colab yêu cầu restart sau cài đặt, restart rồi chạy lại từ ô 1.")
    cell("code", (CODE / "colab_batch3_setup.py").read_text(encoding="utf-8"))
    cell("markdown", "## 3. Nạp mã nguồn đã đóng gói")
    cell("code", "import base64, io, zipfile\nCODE_ROOT = Path('/content/aic_batch3_code')\nCODE_ROOT.mkdir(exist_ok=True)\nSPEED_PROFILE = 'a100-fast'\nEMBEDDED_CODE = " + repr(encoded) + "\nwith zipfile.ZipFile(io.BytesIO(base64.b64decode(EMBEDDED_CODE))) as z:\n    z.extractall(CODE_ROOT)\nprint('Đã nạp mã nguồn:', CODE_ROOT)\n", {"cellView": "form"})
    cell("markdown", "## 4. Kiểm tra đủ video và dung lượng\nỔ tạm cần chứa ảnh của một nhóm và một video đang xử lý. Drive cần đủ chỗ cho toàn bộ kết quả. Số lượng video dưới đây là số thực tế đã upload, không phải kiểm chứng với danh sách gốc của BTC.")
    cell("code", """import shutil
sys.path.insert(0, str(CODE_ROOT))
from colab_batch3 import loose_videos
for group in GROUPS:
    videos = loose_videos(BATCH_ROOT / 'videos', group)
    print(group, len(videos), 'videos', round(sum(p.stat().st_size for p in videos)/2**30, 2), 'GiB')
print('Ổ tạm còn:', round(shutil.disk_usage('/content').free / 2**30, 1), 'GiB')
(BATCH_ROOT / 'processed_data').mkdir(exist_ok=True)
""")
    cell("markdown", """## 5. Đo thử trên A100 và ước tính mốc 30 giờ
Đọc thời gian thực của nhóm đã hoàn tất, ví dụ M04, từ file `.timing.json` cạnh ZIP.
Nhóm đã xong có **0 giờ công việc còn lại**, không benchmark lại. Các nhóm M còn lại
chỉ được ước tính từ nhóm M đã xong **cùng speed_profile**; nhóm N tính riêng.
M04 chạy bằng mã cũ được giữ lại nhưng không dùng tốc độ cũ để đánh giá tối ưu mới.
Nếu chưa có nhóm hoàn tất cùng loại, đo hai đoạn tối đa 1 phút và 10 phút để tách
chi phí khởi động model khỏi tốc độ xử lý. Không nhân chi phí nạp model mỗi 2 phút.
Cộng hệ số dự phòng 1,5 cho phần còn lại và 4 giờ dự phòng truyền/gộp local.
Đây **không phải bảo đảm**: tốc độ nhóm khác có thể khác vì chữ, ảnh trùng và Drive.
Tổng dự kiến gồm thời gian nhóm đã xong biết được, lần benchmark này và phần còn lại;
chưa tính các lần benchmark trước, chạy lại hoặc thời gian gián đoạn. ASR không nằm trong phạm vi.
Notebook dừng trước lượt chạy toàn bộ nếu ước tính vượt 30 giờ. Đọc báo cáo để quyết định
chia nhóm sang GPU khác hoặc đổi cấu hình rồi benchmark lại. Không tự dừng giữa chừng
và báo hoàn tất khi còn video chưa xử lý.
""")
    cell("code", """import json
if RUN_BENCHMARK:
    subprocess.run([sys.executable, '-u', str(CODE_ROOT / 'benchmark_colab_batch3.py'),
        '--batch-root', str(BATCH_ROOT), '--clip-model', CLIP_MODEL,
        '--clip-batch-size', str(CLIP_BATCH_SIZE), '--ocr-batch-size', str(OCR_BATCH_SIZE),
        '--interval', str(FRAME_INTERVAL), '--ocr-enhance', str(OCR_ENHANCE),
        '--target-hours', str(TARGET_HOURS),
        '--speed-profile', SPEED_PROFILE, '--archives', *GROUPS], check=True)
budget = json.loads((BATCH_ROOT / 'processed_data' / 'budget_report.json').read_text())
cfg = budget['configuration']
assert cfg.get('speed_profile') == SPEED_PROFILE, 'Hãy benchmark lại với cấu hình tăng tốc.'
assert budget.get('scope') == 'visual_only_asr_delegated', 'Hãy benchmark lại bằng notebook mới.'
assert budget.get('report_version') == 3, 'Báo cáo cũ chưa đo profile tăng tốc: chạy lại ô 5.'
assert (cfg['clip_model'], cfg['clip_batch_size'], cfg['ocr_batch_size'], cfg['interval'],
        cfg['ocr_enhance'], cfg['archives'], cfg['target_hours']) == (CLIP_MODEL, CLIP_BATCH_SIZE,
        OCR_BATCH_SIZE, FRAME_INTERVAL, OCR_ENHANCE, GROUPS, TARGET_HOURS), 'Cấu hình đã đổi: hãy benchmark lại.'
print('Nhóm đã hoàn tất:', ', '.join(budget['completed_groups']) or 'chưa có')
print('Thời gian thực các nhóm đã xong:', round(budget['known_completed_hours'], 2), 'giờ')
print('Phần còn lại gồm dự phòng:', round(budget['estimated_remaining_upper_hours'], 2), 'giờ')
print('Tổng dự kiến biết được:', round(budget['estimated_upper_hours'], 2), 'giờ')
for group in budget['groups']:
    print(group['group'], group['status'], round(group['estimated_hours'], 2), 'giờ còn lại;', group['method'])
if any(g['status'] != 'completed' for g in budget['groups']):
    assert budget['within_budget_estimate'] is True or ALLOW_OVER_BUDGET, (
        'Chưa chứng minh đạt 30h: ước tính còn vượt hoặc thiếu timing nhóm đã xong. '
        'Xem budget_report.json; không chỉ tắt cảnh báo rồi coi như đạt mục tiêu.')
""")
    cell("markdown", "## 6. Xử lý toàn bộ phần hình ảnh/OCR\nKhông chạy ASR. OCR quét tất cả frame còn lại: **N031–N040 chỉ giữ chữ trắng trên nền đen**, dùng bộ lọc hiện có của dự án; nhóm M không áp bộ lọc này. Benchmark cũng dùng đúng cấu hình riêng từng nhóm. Khi một nhóm hoàn tất, ZIP và file checksum được ghi lên Drive. `asr_cache.json` rỗng chỉ để tương thích định dạng bundle; receipt ghi ASR chưa xử lý, giao cho người khác.\n\n**Thanh tiến trình tự hiện dưới ô:** nhóm hoàn tất; video trích frame/lọc trùng; số frame đã trích; phần trăm lọc trùng; CLIP/YOLO/OCR; số byte copy, đóng ZIP và kiểm tra checksum. Nhãn `stages` hiển thị 4 bước cùng thời gian đã chạy, không phải phần trăm thời gian toàn bộ. Video đạt 100% ở bước trích/lọc chưa có nghĩa đã xong CLIP/OCR. Thanh nhóm chỉ tăng sau khi ZIP đã xuất và kiểm tra xong. ETA nếu có chỉ là ước tính của thanh hiện tại.")
    cell("code", """command = [sys.executable, '-u', str(CODE_ROOT / 'colab_batch3.py'),
    '--batch-root', str(BATCH_ROOT), '--input-mode', 'videos',
    '--clip-model', CLIP_MODEL,
    '--clip-batch-size', str(CLIP_BATCH_SIZE), '--ocr-batch-size', str(OCR_BATCH_SIZE),
    '--interval', str(FRAME_INTERVAL), '--ocr-enhance', str(OCR_ENHANCE),
    '--speed-profile', SPEED_PROFILE, '--archives', *GROUPS]
subprocess.run(command, check=True)
""")
    cell("markdown", "## 7. Kiểm tra và tải kết quả\nTải **cả ZIP và `.sha256.json`** từ `processed_data` bằng giao diện Drive. Không tải thư mục `videos`. ZIP dùng ZIP64 cho gói lớn. Máy local sẽ kiểm tra checksum trước khi nhập.")
    cell("code", """import json
from colab_batch3 import digest
for group in GROUPS:
    archive = BATCH_ROOT / 'processed_data' / (Path(group).stem + '_processed.zip')
    checksum = archive.with_suffix('.sha256.json')
    assert archive.is_file() and checksum.is_file(), f'Chưa hoàn tất: {group}'
    assert digest(archive, desc=f'Check {archive.name}') == json.loads(checksum.read_text())['sha256'], archive
    print('OK', archive.name, round(archive.stat().st_size / 2**30, 2), 'GiB')
print(f'Đã kiểm tra {len(GROUPS)} nhóm: {len(GROUPS)} ZIP + {len(GROUPS)} checksum.')
""")
    cell("markdown", """## Chạy trên máy local
Xem `docs/archive/aic-2026/README_COLAB_BATCH3.md`. Dùng `import_colab_batch3.py` để giải nén,
kiểm tra và gộp chỉ mục. Có thể thêm `--old-index` để gộp dataset hiện có.
Máy local không cần chạy lại CLIP/OCR/YOLO; bước gộp FAISS vẫn cần RAM và CPU.
Batch 3 chưa có transcript ASR. Kết quả ASR từ người phụ trách cần được gộp riêng sau đó.
Giữ các thư mục shard sau khi nhập vì ảnh/map được đọc trực tiếp từ đó.
Tìm kiếm và xem keyframe dùng được không có video gốc; phát video đầy đủ cần nguồn video riêng.

**Trạng thái kiểm thử:** notebook cần được chạy trên Colab GPU có Drive của bạn.
Không có kết quả xử lý GPU thật được lưu sẵn trong notebook này.
""")
    notebook = {"nbformat": 4, "nbformat_minor": 5, "metadata": {
        "accelerator": "GPU", "colab": {"name": "AIC_Batch3_Colab.ipynb"},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}, "cells": cells}
    path = OUT / "AIC_Batch3_Colab.ipynb"
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding="utf-8")
    for c in cells:
        if c['cell_type'] == 'code':
            compile(''.join(c['source']), '<notebook-cell>', 'exec')
    print(path)


if __name__ == '__main__':
    main()
