"""Launch the resumable N OCR worker detached from the terminal."""
from pathlib import Path
import subprocess

root = Path(__file__).resolve().parent
log_path = root.parent / 'runtime_logs' / 'n_white_top_ocr.log'
log_path.parent.mkdir(exist_ok=True)
with log_path.open('a') as log:
    proc = subprocess.Popen(['bash',str(root/'run_n_top_ocr_wsl.sh')],
                            cwd=root,stdin=subprocess.DEVNULL,stdout=log,stderr=log,
                            start_new_session=True)
log_path.with_suffix('.pid').write_text(str(proc.pid))
print(f'PID={proc.pid} log={log_path}')
