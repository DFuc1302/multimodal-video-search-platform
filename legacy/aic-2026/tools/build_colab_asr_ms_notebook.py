"""Generate a self-contained Colab notebook without local package requirements."""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
runner = (ROOT / 'colab_asr_ms.py').read_text(encoding='utf-8')
cells = []
def cell(kind, source):
    item = dict(cell_type=kind, metadata={}, source=source.splitlines(keepends=True))
    if kind == 'code':
        ast.parse(source)
        item.update(execution_count=None, outputs=[])
    cells.append(item)

cell('markdown', '''# ASR video M + S trên Colab
Chạy trên **runtime GPU riêng**, lần lượt các ô từ trên xuống. Chỉ đọc video M ở batch 3/videos và S ở batch 2 (quét cả thư mục con); bỏ qua N. Video phải giải nén và tên dạng M01_V001.mp4 / S01_V001.mp4.

Mục tiêu **12 giờ trên A100 40 GB**, không benchmark và không chặn chạy theo ngân sách. Faster-Whisper large-v3-turbo, FP16, batch 32, beam 1, VAD bỏ khoảng im lặng, tiếng Việt. Copy trước một video song song với GPU để giảm chờ Drive. Có thể đổi LANGUAGE='auto' cho video nhiều ngôn ngữ. Không thể bảo đảm 12 giờ khi chưa biết tổng thời lượng và tốc độ Drive. Giữ word timestamps để tạo cửa sổ tìm kiếm 2 giây.

Lưu từng video trên Drive; khi ngắt runtime, chạy lại các ô, chỉ video dở cần làm lại. **TỔNG ASR** hiển thị phần trăm theo thời lượng video, cập nhật ngay trong video đang chạy. **Video đã lưu** đếm video hoàn tất; thanh tên video hiển thị tiến trình video hiện tại. Phần trăm không phải tỷ lệ thời gian tính toán; ETA chỉ ước lượng, nhất là sau resume. Đọc metadata thời lượng đầu phiên không chạy nhận dạng thử. Không chạy hai bản notebook vào cùng OUTPUT. Đĩa Colab cần đủ chỗ cho hai video lớn nhất cùng lúc.

Tài liệu thư viện: https://github.com/SYSTRAN/faster-whisper
''')
cell('markdown', '## 1. Kết nối Drive và cài thư viện')
cell('code', '''from google.colab import drive
drive.mount('/content/drive')
import subprocess, sys
subprocess.run([sys.executable, '-m', 'pip', 'install', 'faster-whisper==1.2.1', 'tqdm', 'nvidia-cublas-cu12', 'nvidia-cudnn-cu12==9.*'], check=True)
subprocess.run(['nvidia-smi'], check=True)
''')
cell('markdown', '## 2. Cấu hình')
cell('code', '''M_ROOT = '/content/drive/MyDrive/DATA_AIC/batch 3/videos'
S_ROOT = '/content/drive/MyDrive/DATA_AIC/batch 2'
OUTPUT = '/content/drive/MyDrive/DATA_AIC/asr_MS'
MODEL = 'large-v3-turbo'
BATCH_SIZE = 32  # A100 40 GB; nếu CUDA out of memory, giảm xuống 16 hoặc 8
LANGUAGE = 'vi'
''')
cell('markdown', '## 3. Tạo chương trình và chạy ASR — thanh tiến trình hiện bên dưới')
cell('code', 'from pathlib import Path\nPath("/content/colab_asr_ms.py").write_text(' + repr(runner) + ', encoding="utf-8")\n' + '''import os, site
env = os.environ.copy()
libs = [str(p) for root in site.getsitepackages() for p in Path(root).glob('nvidia/*/lib')]
env['LD_LIBRARY_PATH'] = ':'.join(libs + [env.get('LD_LIBRARY_PATH', '')])
cmd = [sys.executable, '-u', '/content/colab_asr_ms.py', '--m-root', M_ROOT,
       '--s-root', S_ROOT, '--output', OUTPUT, '--model', MODEL,
       '--batch-size', str(BATCH_SIZE), '--language', LANGUAGE]
with subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                      text=True, encoding='utf-8', errors='replace', bufsize=1) as process:
    for line in process.stdout:
        print(line, end='', flush=True)
    result = process.wait()
if result:
    raise subprocess.CalledProcessError(result, cmd)
''')
cell('markdown', '## 4. Tải kết quả — chỉ chạy sau khi ô 3 hoàn tất\nFile được giữ trên Drive tại DATA_AIC/asr_MS/asr_cache.json. Kết quả riêng, không ghi đè dữ liệu CLIP/OCR. Có thể tải bằng Drive nếu trình duyệt tải chậm.')
cell('code', '''import json
from google.colab import files
status = json.loads((Path(OUTPUT) / 'status.json').read_text())
assert status['complete'], status
files.download(str(Path(OUTPUT) / 'asr_cache.json'))
''')
notebook = dict(nbformat=4, nbformat_minor=4, cells=cells,
                metadata={'colab': {'name': 'AIC_ASR_M_S_Colab.ipynb'},
                          'accelerator': 'GPU',
                          'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}})
out = ROOT.parent / 'output' / 'colab_asr' / 'AIC_ASR_M_S_Colab.ipynb'
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding='utf-8')
print(out)
