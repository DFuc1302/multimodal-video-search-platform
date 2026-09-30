#!/usr/bin/env bash
set -e

python -m pip install --upgrade pip
python -m pip install "setuptools==69.5.1" "wheel==0.43.0"

# Disable build isolation so openai-whisper can see setuptools/pkg_resources.
python -m pip install --no-cache-dir --no-build-isolation -r requirements_wsl_locked.txt
