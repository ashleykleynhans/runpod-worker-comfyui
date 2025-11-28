FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_PREFER_BINARY=1 \
    PYTHONUNBUFFERED=1

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

WORKDIR /

# Upgrade apt packages and install required dependencies
RUN apt update && \
    apt upgrade -y && \
    apt install -y \
      python3-dev \
      python3-pip \
      fonts-dejavu-core \
      rsync \
      git \
      jq \
      moreutils \
      aria2 \
      wget \
      curl \
      libglib2.0-0 \
      libsm6 \
      libgl1 \
      libxrender1 \
      libxext6 \
      ffmpeg \
      libgoogle-perftools4 \
      libtcmalloc-minimal4 \
      procps && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/* && \
    apt-get clean -y

# Set Python
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Install Worker dependencies
RUN pip install requests runpod==1.7.10

# Install InSPyReNet transparent background model used by the transparent-background Python
# module (https://github.com/plemeri/transparent-background) so it doesn't have to be
# downloaded from Google Drive at run time - this increases stability and performance.
RUN pip install gdown && \
  mkdir -p /root/.transparent-background && \
  gdown 13oBl5MTVcWER3YU4fSxW3ATlVfueFQPY -O /root/.transparent-background/ckpt_base.pth

# Add RunPod Handler and Docker container start script
COPY start.sh handler.py ./

# Add validation schemas
COPY schemas /schemas

# Add workflows
COPY workflows /workflows

# Start the container
RUN chmod +x /start.sh
ENTRYPOINT /start.sh
