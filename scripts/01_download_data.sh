#!/usr/bin/env bash
# Get CelebV-HQ source data from a HuggingFace mirror, then extract a SUBSET of
# clips (we only need ~2k clips to build 20k training frames — not all 41 GB).
set -e
cd "$(dirname "$0")/.."
WORK="${FACEGEN_WORK:?set FACEGEN_WORK to a big partition, e.g. /media/usr/DATA/sieun/facegen}"
mkdir -p "$WORK/data"

echo "=============================================================="
echo " CelebV-HQ acquisition  ->  $WORK/data"
echo " Both HF mirrors are ~41 GB video archives. We download one,"
echo " then stream-extract only the first ~2500 clips."
echo "=============================================================="

# 1) download the plain tar (resumable; ~41.9 GB). Lands in $WORK/data/videos.tar
huggingface-cli download SwayStar123/CelebV-HQ videos.tar \
    --repo-type dataset --local-dir "$WORK/data"

# 2) extract a subset of clips (fast: stops after --max-videos)
python scripts/extract_tar_subset.py \
    --tar "$WORK/data/videos.tar" \
    --out "$WORK/data/raw" \
    --max-videos 2500

# 3) (optional) free 41 GB once you've extracted enough clips
# rm -f "$WORK/data/videos.tar"

echo "Next: extract faces ->"
echo "  python scripts/02_preprocess.py --in $WORK/data/raw \\"
echo "      --out $WORK/data/faces256 --size 256 --max-images 20000 --delete-source"
