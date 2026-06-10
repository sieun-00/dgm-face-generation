"""Central configuration. Edit once; every script imports from here.

These values ARE the reproducibility record for the top-10 verification:
keep SEEDS and TRUNC exactly equal to what produced the submitted zip.

DISK: by default everything lives next to this repo. If your HOME has a small
quota, point all heavy outputs to a big partition with ONE env var:

    export FACEGEN_WORK=/scratch/$USER/facegen      # or /data/... , /mnt/...

Find a partition with free space:  bash scripts/00_check_disk.sh
"""
import os
from pathlib import Path

# ---- work root (data / runs / out live here) ---------------------------
ROOT = Path(__file__).resolve().parent
WORK = Path(os.environ.get("FACEGEN_WORK", ROOT)).resolve()   # heavy outputs go here

# ---- paths -------------------------------------------------------------
DATA_RAW = WORK / "data" / "raw"           # downloaded clips / images
DATA_FACES = WORK / "data" / "faces256"    # preprocessed 256x256 faces
DATASET_ZIP = WORK / "data" / "celebvhq256.zip"   # OPTIONAL StyleGAN zip (can skip)
RUNS = WORK / "runs"                       # training output (snapshots are large)
OUT = WORK / "out"                         # generated images
STYLEGAN_REPO = ROOT / "external" / "stylegan2-ada-pytorch"

# pretrained FFHQ-256 start point. 04_train.sh passes --resume ffhq256 (repo shortcut).

# ---- disk safety -------------------------------------------------------
MIN_FREE_GB = 5            # 02_preprocess aborts early if free space < this

# ---- FINAL SUBMISSION (REPRODUCIBILITY-CRITICAL) -----------------------
# Winning config: 60k-frame (v2) fine-tune, last snapshot, NO truncation.
# Leaderboard: FID 33.32 | IS 4.22 | KID 0.0036 | TopPR 0.8505
WINNING_NETWORK = "runs/00003-faces256_big-mirror-paper256-kimg1000-batch32-noaug-resumeffhq256/network-snapshot-001000.pkl"
NUM_IMAGES = 1000          # leaderboard requires exactly 1000
SEEDS = list(range(0, NUM_IMAGES))   # seeds 0..999, one per image
TRUNC = 1.0                # truncation psi = 1.0 (no truncation) -> best FID/KID/IS
IMG_SIZE = 256             # match leaderboard reference resolution
IMG_FORMAT = "png"         # final submission images (png or jpg both accepted)

# Exact reproduction command (top-10 verification):
#   python scripts/05_generate.py --network <WINNING_NETWORK> \
#       --outdir out/final --seeds 0-999 --trunc 1.0 --size 256

# ---- preprocessing (DISK-FRUGAL DEFAULTS) ------------------------------
PREP_FORMAT = "jpg"        # store the *training* set as JPEG -> ~10x smaller than PNG
PREP_QUALITY = 95          # JPEG quality for the training set
MAX_IMAGES = 20000         # transfer learning needs ~10-30k; hard cap to bound disk
MAX_PER_VIDEO = 12         # frames per clip (limits near-duplicates AND disk use)

# ---- training ----------------------------------------------------------
GPU = "0"                  # CUDA_VISIBLE_DEVICES — RTX A5000
TRAIN_KIMG = 1000          # transfer learning usually converges ~1000 kimg
SNAP = 100                 # snapshot/metric every N kimg (higher = fewer big pkls)
CFG = "paper256"           # stylegan2-ada config tuned for 256x256
