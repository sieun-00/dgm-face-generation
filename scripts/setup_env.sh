#!/usr/bin/env bash
# One-time environment setup for StyleGAN2-ADA fine-tuning.
# This server HAS system CUDA toolkits: we use /usr/local/cuda-12.1 and match
# PyTorch to cu121. (Default `nvcc` is 11.5, so we point CUDA_HOME at 12.1
# explicitly.) No conda CUDA toolkit needed.
#
# IMPORTANT: run these by COPY-PASTE, not `bash setup_env.sh`, because
# `conda activate` only works in an interactive shell.
#
# This file doubles as the reproducible env record for the top-10 submission.

ENV=facegen

# 0) make conda activatable in this shell
source "$(conda info --base)/etc/profile.d/conda.sh"

# 1) create + activate the env
conda create -n "$ENV" python=3.9 -y
conda activate "$ENV"

# 3) PyTorch — cu118 (CUDA 11.8 runtime). NOTE: install torch BEFORE picking the
#    build CUDA, then match the build toolkit's major version to torch (11.x).
pip install torch==2.0.1 torchvision==0.15.2 \
    --index-url https://download.pytorch.org/whl/cu118

# 2) point the build at a COMPLETE CUDA toolkit (must have BOTH bin/nvcc AND
#    include/cuda_runtime_api.h). On this server /usr/local/cuda -> 12.6 and is
#    complete; cuda-12.1 is runtime-only (no nvcc/headers) so it CANNOT compile.
#    VERIFIED: nvcc 12.6 compiles the ops fine even with cu118 torch.
export CUDA_HOME=/usr/local/cuda            # -> 12.6, complete (has nvcc + headers)
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
export TORCH_CUDA_ARCH_LIST="8.6"           # A5000 + 3090 are Ampere (sm_86)
hash -r                                     # clear bash's cached nvcc path!
nvcc --version                              # must report 11.x (here 11.5)
ls "$CUDA_HOME/include/cuda_runtime_api.h"  # headers must exist
# FALLBACK if the system toolkit is incomplete or version-mismatches torch:
#   conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit -y
#   export CUDA_HOME=$CONDA_PREFIX; export PATH="$CUDA_HOME/bin:$PATH"; hash -r

# 4) python deps (ninja compiles the StyleGAN custom CUDA ops)
pip install -r requirements.txt

# 5) clone the StyleGAN repo (dnnlib / torch_utils / custom ops)
mkdir -p external
test -d external/stylegan2-ada-pytorch || \
  git clone https://github.com/NVlabs/stylegan2-ada-pytorch.git \
            external/stylegan2-ada-pytorch

# 6) SMOKE TEST — forces the custom CUDA ops to compile + checks the GPU.
rm -rf ~/.cache/torch_extensions/*          # wipe any failed build cache first
echo "=== smoke test: compiling custom ops + 1 sample (first run is slow) ==="
FFHQ256="https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/transfer-learning-source-nets/ffhq-res256-mirror-paper256-noaug.pkl"
CUDA_VISIBLE_DEVICES=0 python external/stylegan2-ada-pytorch/generate.py \
    --outdir=/tmp/sg_smoke --seeds=0 --network="$FFHQ256"
echo "If /tmp/sg_smoke/seed0000.png exists, the env is READY."

# 7) persist the CUDA env for future shells (so training picks the right nvcc):
#   cat >> ~/.bashrc <<'RC'
#   export CUDA_HOME=/usr/local/cuda-12.1
#   export PATH=$CUDA_HOME/bin:$PATH
#   export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
#   export TORCH_CUDA_ARCH_LIST="8.6"
#   RC
