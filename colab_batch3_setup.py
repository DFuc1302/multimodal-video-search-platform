"""Colab setup cell: support Python 3.13 without replacing its CUDA PyTorch."""
import importlib.metadata as metadata
import os
from pathlib import Path
import shutil
import subprocess
import sys


def requirements_for(version):
    if version >= (3, 14):
        raise RuntimeError("This setup supports Python 3.10–3.13; choose a matching Colab runtime.")
    if version < (3, 10):
        raise RuntimeError("Python >=3.10 is required.")
    numpy = "numpy>=2.1,<2.3" if version >= (3, 13) else "numpy>=1.26,<2"
    return [numpy, "scipy>=1.14,<2", "pillow>=10", "tqdm",
            "opencv-python==4.11.0.86", "opencv-python-headless==4.11.0.86",
            "scikit-image>=0.25,<1", "scikit-learn>=1.6,<2", "pandas>=2.2.3,<3",
            "matplotlib>=3.9,<4", "faiss-cpu>=1.10,<2", "easyocr==1.7.2",
            "transformers==4.46.3", "ultralytics>=8.3.100,<9", "redis", "ImageHash==4.3.2"]


def main():
    print("Python:", sys.version, flush=True)
    packages = requirements_for(sys.version_info[:2])
    log_path = Path('/content/aic_batch3_install.log')
    constraints = Path('/content/aic_batch3_torch_constraints.txt')
    installed_torch = {}
    for name in ('torch', 'torchvision', 'torchaudio'):
        try:
            installed_torch[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            if name != 'torchaudio':
                raise RuntimeError('Choose a Colab GPU runtime with torch and torchvision installed.')
    constraints.write_text(''.join(f'{k}=={v}\n' for k, v in installed_torch.items()))
    print('Preserving Colab CUDA packages:', installed_torch, flush=True)
    with log_path.open('w', encoding='utf-8') as log:
        def run_logged(command):
            print('Running:', ' '.join(map(str, command)), flush=True)
            with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                  text=True, errors='replace', bufsize=1) as proc:
                for line in proc.stdout:
                    print(line, end='', flush=True)
                    log.write(line)
                    log.flush()
                if proc.wait():
                    raise RuntimeError(f'Installation/check failed (exit {proc.returncode}). '
                                       f'Full log: {log_path}; inspect the first ERROR above.')
        if not shutil.which('ffmpeg'):
            run_logged(['apt-get', '-qq', 'update'])
            run_logged(['apt-get', '-qq', 'install', '-y', 'ffmpeg'])
        # Wheels only: never attempt unsupported NumPy/Rust/C++ builds in Colab.
        run_logged([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip', 'wheel'])
        run_logged([sys.executable, '-m', 'pip', 'install', '--upgrade',
                    '--only-binary=:all:', '--constraint', str(constraints), *packages])
        # Fresh subprocess avoids packages imported by the notebook before pip ran.
        smoke = '''import numpy as np, scipy, cv2, torch, torchvision, faiss, easyocr
from transformers import CLIPModel, CLIPProcessor, CLIPTextModelWithProjection
from ultralytics import YOLO
import imagehash, redis, skimage, sklearn, pandas
assert torch.cuda.is_available(), 'GPU not available: choose A100 GPU runtime'
a = np.zeros((2, 3), dtype=np.float32)
assert torch.from_numpy(a).numpy().shape == (2, 3)
index = faiss.IndexFlatL2(3)
index.add(a)
assert index.ntotal == 2
from torchvision.ops import nms
boxes = torch.tensor([[0., 0., 1., 1.]], device='cuda')
assert nms(boxes, torch.tensor([0.9], device='cuda'), 0.5).numel() == 1
print('IMPORT/GPU OK:', torch.cuda.get_device_name(0), 'NumPy', np.__version__, 'Torch', torch.__version__)
'''
        # Transformers is used for PyTorch CLIP only; do not import unrelated TF/Keras.
        os.environ.update(USE_TF='0', USE_FLAX='0')
        run_logged([sys.executable, '-c', smoke])
        for name, version in installed_torch.items():
            if metadata.version(name) != version:
                raise RuntimeError(f'{name} unexpectedly changed; restart with a clean GPU runtime.')
        with Path('/content/aic_batch3_environment.txt').open('w') as env_file:
            subprocess.run([sys.executable, '-m', 'pip', 'freeze'], stdout=env_file, check=True)
    print('Setup complete. Continue with cell 3. No ASR packages installed by this cell.')


if __name__ == '__main__':
    main()
