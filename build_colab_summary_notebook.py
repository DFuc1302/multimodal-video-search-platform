import ast
import json
from pathlib import Path

root = Path(__file__).resolve().parent
cells = []
def add(kind, text):
    cell = dict(cell_type=kind, metadata={}, source=text.splitlines(keepends=True))
    if kind == 'code':
        ast.parse(text)
        cell.update(outputs=[], execution_count=None)
    cells.append(cell)

add('markdown', '''# Tiếp tục summary M trong ngân sách 3 giờ
Chỉ chạy M còn thiếu trong OUTPUT cũ. Giữ nguyên toàn bộ đoạn đã lưu, kể cả video đang dở; tiếp tục từ mốc kết thúc cuối cùng. Không cần tạo OUTPUT mới. Không chạy đồng thời phiên cũ và phiên mới.

**Đổi lại tốc độ:** phần mới dùng 2 ảnh cỡ 336px, tối đa 128 token, batch 2 và đoạn 120 giây. Tự tăng độ dài đoạn lên tối đa 300 giây theo tốc độ thực đo. Ảnh lấy mẫu thưa có thể bỏ lỡ cảnh ngắn; đây không phải chất lượng/mật độ tương đương 3 ảnh mỗi 30 giây. ASR đầy đủ vẫn được giữ cho từng đoạn. Các đoạn cũ không bị gộp hoặc viết lại.

Ngân sách 3 giờ tính từ lúc gọi run, gồm đọc nguồn và nạp model, chưa gồm cài thư viện/mount Drive. Dành 3 phút cuối để xuất kết quả; thao tác Drive hoặc một lô GPU đang chạy có thể vượt mốc dừng. Không thể cam kết xong toàn bộ trước khi đo tốc độ trên dữ liệu/GPU thực tế. Code in tên GPU thực tế (không giả định tên G4), ETA và cảnh báo nếu không đủ thời gian.

Dừng phiên summary cũ trước khi chạy. An toàn nhất: Restart session, chạy lại notebook này và giữ nguyên đường dẫn OUTPUT/ASR. File status.json ghi complete=false và danh sách còn thiếu nếu hết giờ hoặc gặp lỗi. ZIP summary_M.zip chỉ chứa các video M hoàn tất; checkpoint của video đang dở vẫn ở Drive. Phần trăm tiến độ mới tính theo thời lượng video đã phủ, không so trực tiếp với số đoạn của bản cũ.

JSON giữ video_metadata và phan_doan_thoi_gian cho summary_search.py. Chưa tạo tóm tắt toàn video hoặc nhập OCR chuyên dụng. Model: https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct
''')
add('markdown', '## 1. Drive và thư viện')
add('code', '''from google.colab import drive
drive.mount('/content/drive')
import subprocess, sys
subprocess.run(['apt-get','-qq','install','-y','ffmpeg'], check=True)
subprocess.run([sys.executable,'-m','pip','install','transformers==4.51.3','accelerate','av','pillow','tqdm'], check=True)
''')
add('markdown', '## 2. Cấu hình')
add('code', '''M_ROOT = '/content/drive/MyDrive/DATA_AIC/batch 3/videos'
S_ROOT = '/content/drive/MyDrive/DATA_AIC/batch 2'
ASR_PATH = '/content/drive/MyDrive/DATA_AIC/asr_MS/asr_cache.json'
OUTPUT = '/content/drive/MyDrive/DATA_AIC/summary_MS'
WINDOW_SECONDS = 120
FAMILIES = ['M']
BUDGET_HOURS = 3
BATCH_SIZE = 2
ADAPTIVE_WINDOW = True
MAX_WINDOW_SECONDS = 300
''')
add('markdown', '## 3. Chạy summary — tiếp tục từ checkpoint nếu bị ngắt')
source = 'from pathlib import Path\nimport subprocess, sys\n'
for name in ('colab_asr_ms.py', 'colab_summary_ms.py'):
    source += f'Path("/content/{name}").write_text({(root/name).read_text(encoding="utf-8")!r}, encoding="utf-8")\n'
source += '''import json
config = dict(m_root=M_ROOT, s_root=S_ROOT, asr_path=ASR_PATH, output=OUTPUT,
              window=WINDOW_SECONDS, families=FAMILIES, budget_hours=BUDGET_HOURS,
              batch_size=BATCH_SIZE, adaptive_window=ADAPTIVE_WINDOW, max_window=MAX_WINDOW_SECONDS)
Path('/content/summary_config.json').write_text(json.dumps(config))
entry = "import json; from colab_summary_ms import run; run(**json.load(open('/content/summary_config.json')))"
with subprocess.Popen([sys.executable, '-u', '-c', entry], cwd='/content',
                      stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1) as proc:
    try:
        for line in proc.stdout:
            print(line, end='', flush=True)
        code = proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        proc.wait()
        raise
if code:
    raise RuntimeError('Summary bị lỗi. Xem phía trên; checkpoint đã lưu vẫn được giữ.')
status = json.loads((Path(OUTPUT) / 'status.json').read_text())
print('HOÀN TẤT' if status['complete'] else 'CHƯA XONG — chạy lại để tiếp tục', status)
'''
add('code', source)
add('markdown', '## 4. Tải JSON summary, không kèm video')
add('code', '''from google.colab import files
files.download(str(Path(OUTPUT) / 'summary_M.zip'))
''')
out = root.parent / 'output/colab_summary/AIC_Summary_M_S_Colab.ipynb'
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(dict(nbformat=4, nbformat_minor=4, cells=cells,
    metadata=dict(accelerator='GPU', kernelspec=dict(name='python3', display_name='Python 3', language='python'))), ensure_ascii=False), encoding='utf-8')
print(out)
resume = '\n'.join(''.join(cell['source']) for cell in cells[4:7] if cell['cell_type'] == 'code')
# A single replacement cell for an already mounted/configured Colab runtime.
(out.parent / 'Resume_M_3h.py').write_text(resume, encoding='utf-8')
