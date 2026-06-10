# Face Generation Challenge — DGM Spring 2026

Reproducible pipeline for the CelebV-HQ face-generation challenge. We transfer-learn
**StyleGAN2-ADA** from an FFHQ-256 checkpoint onto CelebV-HQ frames and deterministically
generate 1,000 face images at 256×256 for the leaderboard.

**Final result:** FID 33.32 · IS 4.22 · KID 0.0036 · TopPR 0.8505.

## Approach

The leaderboard compares generated images to 32,550 CelebV-HQ frames with FID↓, IS↑, KID↓, TopPR↑,
ranked by the average rank. Since FID/KID are distribution distances, the largest lever is matching the
target distribution, so we fine-tune directly on CelebV-HQ rather than on a generic face set.
StyleGAN2-ADA is used because the FFHQ-256 prior lets transfer learning converge in hours on one 24 GB
GPU, the final weights are ~300 MB (< 5 GB), sampling 1,000 images takes seconds, and generation is
fully seed-deterministic.

A key empirical finding: the model is **coverage-limited**, so lowering truncation ψ (which trims
diversity) *raises* FID. FID decreases monotonically toward ψ = 1.0 (49.2 → 40.1 → 33.3 at ψ = 0.7,
0.8, 1.0), so the final submission uses **ψ = 1.0** (no truncation).

## Environment

Full, pinned setup is in `scripts/setup_env.sh` (Python 3.9, PyTorch 2.0.1+cu118, system CUDA for the
custom-op build). In short:

```bash
git clone https://github.com/NVlabs/stylegan2-ada-pytorch.git external/stylegan2-ada-pytorch
conda create -n facegen python=3.9 -y && conda activate facegen
pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
export CUDA_HOME=/usr/local/cuda          # a complete toolkit (nvcc + headers)
export PATH=$CUDA_HOME/bin:$PATH
export TORCH_CUDA_ARCH_LIST="8.6"         # Ampere (A5000 / 3090)
```

Container build: `docker build -t facegen . && docker run --gpus all -it facegen`.

Heavy outputs (data, checkpoints) are written under `$FACEGEN_WORK` if set, otherwise next to the repo:

```bash
export FACEGEN_WORK=/path/to/large/disk/facegen
```

## Pipeline (run in order)

```bash
# 1) data: download a CelebV-HQ mirror tar, stream-extract a subset of clips
bash scripts/01_download_data.sh

# 2) preprocess -> 256x256 JPEG faces (<=8 frames/clip, capped at --max-images)
python scripts/02_preprocess.py \
  --in "$FACEGEN_WORK/data/raw" --out "$FACEGEN_WORK/data/faces256_big" \
  --size 256 --max-images 60000 --max-per-video 8 --delete-source

# 3) fine-tune from FFHQ-256 (ADA disabled; see report Sec. 5), trains from the image folder
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
python external/stylegan2-ada-pytorch/train.py \
  --outdir "$FACEGEN_WORK/runs" --data "$FACEGEN_WORK/data/faces256_big" \
  --gpus 1 --cfg paper256 --aug noaug --mirror 1 --resume ffhq256 \
  --snap 20 --kimg 1000 --batch 32 --metrics none

# 4) deterministic generation: seeds 0..999, 256x256, psi=1.0  (reproducible inference code)
python scripts/05_generate.py \
  --network "$FACEGEN_WORK"/runs/00003-*/network-snapshot-001000.pkl \
  --outdir out/final --seeds 0-999 --trunc 1.0 --size 256

# 5) package the leaderboard zip (validates count / size / flat structure)
python scripts/06_package_submission.py --indir out/final --zip submission.zip --jpeg-quality 95
```

`scripts/03_make_dataset.sh` (optional dataset-zip packing) and `scripts/07_psi_sweep.sh`
(truncation A/B helper) are provided but not required for the final result.

## Final submission (exact settings)

| Setting | Value |
|---|---|
| Dataset | CelebV-HQ — 60k frames from 10,000 clips, 256×256 |
| Model | StyleGAN2-ADA, `paper256`, transfer from FFHQ-256, `--aug noaug`, batch 32, 1000 kimg |
| Weights (≤ 5 GB) | `runs/00003-.../network-snapshot-001000.pkl` (~300 MB) |
| Seeds | 0–999 (one latent per image) |
| Truncation ψ | 1.0 (no truncation — model is coverage-limited) |
| Leaderboard | FID 33.32 · IS 4.22 · KID 0.0036 · TopPR 0.8505 |

The submitted images are reproduced bit-for-bit by step 4 above from the released checkpoint and seeds.

## Reproducibility notes

- `config.py` records the exact seeds (0–999) and ψ (1.0) used for the submission.
- `requirements.txt` and `Dockerfile` pin the environment; the custom CUDA ops are JIT-compiled at runtime.
- All images come from the fine-tuned model only; no crawled real images and no external generative services.
