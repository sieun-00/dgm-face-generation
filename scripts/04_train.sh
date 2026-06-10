#!/usr/bin/env bash
# Fine-tune StyleGAN2-ADA on CelebV-HQ from FFHQ pretrained weights.
# Trains DIRECTLY from the image folder (no dataset zip needed -> saves disk).
#
# IMPORTANT compatibility notes (this server / PyTorch 2.0.1+cu118):
#   * --aug noaug : ADA uses grid_sample, whose 2nd-order grad is unimplemented
#     in PyTorch 2.x, which crashes the R1 path. We disable ADA. (20k images +
#     FFHQ transfer is enough without it.)
#   * --metrics none : in-training fid50k_full is slow AND memory-heavy on a
#     shared GPU. We snapshot often and compute FID post-hoc on chosen ckpts
#     (calc_metrics.py) / use the leaderboard to select.
set -e
cd "$(dirname "$0")/.."

# Deterministic GPU selection (nvidia-smi GPU0 = A5000 by PCI bus). Both GPUs are
# shared on this box — check nvidia-smi and pick the freer one.
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0
export PYTHONWARNINGS=ignore           # silence conv2d_gradfix fallback spam
WORK="${FACEGEN_WORK:-$PWD}"           # snapshots are large; keep them on a big partition

REPO=external/stylegan2-ada-pytorch
DATA="$WORK/data/faces256"             # a folder of 256x256 images works directly
OUTDIR="$WORK/runs"
RESUME=ffhq256                         # repo shortcut for the FFHQ-256 pretrained pkl

mkdir -p "$OUTDIR"

# disk sanity check before a long run
FREE_GB=$(df -BG --output=avail "$OUTDIR" | tail -1 | tr -dc '0-9')
echo "free space on runs partition: ${FREE_GB} GB"
if [ "${FREE_GB:-0}" -lt 10 ]; then
  echo "WARNING: <10 GB free. Each snapshot pkl is ~300 MB; set FACEGEN_WORK to a bigger disk."
fi

python "$REPO/train.py" \
  --outdir "$OUTDIR" \
  --data "$DATA" \
  --gpus 1 \
  --cfg paper256 \
  --aug noaug \
  --mirror 1 \
  --resume "$RESUME" \
  --snap 10 \
  --kimg 1000 \
  --batch 32 \
  --metrics none

echo "Monitor tick speed:  grep -aE '^tick' $OUTDIR/*/log.txt | tail"
echo "Snapshots land as $OUTDIR/<run>/network-snapshot-XXXXXX.pkl every ~40 kimg."
echo "Score candidates on the leaderboard; FID for the report via calc_metrics.py."
