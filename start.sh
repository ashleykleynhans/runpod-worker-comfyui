#!/usr/bin/env bash

echo "Worker Initiated"

echo "Symlinking files from Network Volume"
rm -rf /workspace && \
  ln -s /runpod-volume /workspace

echo "Starting ComfyUI API"
source /workspace/venv/bin/activate
TCMALLOC="$(ldconfig -p | grep -Po "libtcmalloc.so.\d" | head -n 1)"
export LD_PRELOAD="${TCMALLOC}"
export PYTHONUNBUFFERED=true
export HF_HOME="/workspace"

# Set InSPyReNet background-removal model path to the model downloaded
# from Google drive into the Docker container
export TRANSPARENT_BACKGROUND_FILE_PATH=/root/.transparent-background

cd /workspace/ComfyUI
python main.py --port 3000 --temp-directory /tmp > /workspace/logs/comfyui-serverless.log 2>&1 &
deactivate

echo "Starting RunPod Handler"
python3 -u /handler.py
