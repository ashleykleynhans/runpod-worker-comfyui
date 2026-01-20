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

# Disable xformers for CUDA 12.8
EXTRA_ARGS=""
if [ "${CUDA_SHORT}" = "cu128" ]; then
    EXTRA_ARGS="--disable-xformers"
fi

python main.py --port 3000 --temp-directory /tmp ${EXTRA_ARGS} > /workspace/logs/comfyui-serverless.log 2>&1 &
deactivate

echo "Starting Runpod Handler"
python3 -u /handler.py
