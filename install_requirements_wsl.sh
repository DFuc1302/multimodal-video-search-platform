#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
sudo apt-get update
sudo apt-get install -y python3.12-venv python3.12-dev build-essential ffmpeg libgl1 libglib2.0-0
VENV_DIR="${TEAM_VENV:-$HOME/.venvs/aic-team}"
python3.12 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip
python -m pip install setuptools==69.5.1 wheel==0.43.0
python -m pip install -r requirements_wsl_locked.txt --no-build-isolation
echo 'Cai xong. Chay: bash run_local_server_wsl.sh'
