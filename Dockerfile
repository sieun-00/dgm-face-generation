# Reproducible environment for the top-10 verification submission.
# Matches the host (driver 535 / CUDA 12.2). Devel image provides nvcc for custom ops.
FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev git ninja-build build-essential \
    libgl1 libglib2.0-0 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3 /usr/bin/python
WORKDIR /workspace

COPY requirements.txt .
# torch 2.0.1+cu118 — the exact build used to produce the submission
RUN pip3 install --no-cache-dir --index-url https://download.pytorch.org/whl/cu118 \
        torch==2.0.1 torchvision==0.15.2 && \
    pip3 install --no-cache-dir -r requirements.txt

COPY . .
# clone the StyleGAN repo at build time if not vendored
RUN test -d external/stylegan2-ada-pytorch || \
    git clone https://github.com/NVlabs/stylegan2-ada-pytorch.git external/stylegan2-ada-pytorch

CMD ["/bin/bash"]
