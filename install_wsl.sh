#!/usr/bin/env bash
set -e

sudo apt update
sudo apt install -y ffmpeg python3-venv python3-dev build-essential libglib2.0-0 libgl1

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel

# Install CUDA PyTorch first so pip does not accidentally choose the CPU wheel.
# This works well on WSL when the Windows NVIDIA driver already exposes CUDA.
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements_wsl.txt

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")
PY
