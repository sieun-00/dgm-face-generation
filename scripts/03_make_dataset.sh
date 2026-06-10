#!/usr/bin/env bash
# OPTIONAL — and SKIPPED by default to save disk.
# StyleGAN2-ADA's train.py reads a directory of equal-size images directly via
# --data, so you do NOT need to build a zip (which would duplicate the dataset).
#
# Only build the zip if you specifically want the packed format (e.g. faster I/O
# on a networked filesystem). It needs as much free space as data/faces256.
set -e
cd "$(dirname "$0")/.."
WORK="${FACEGEN_WORK:-$PWD}"

REPO=external/stylegan2-ada-pytorch
SRC="$WORK/data/faces256"
DST="$WORK/data/celebvhq256.zip"

echo "NOTE: this step is optional. 04_train.sh trains directly from $SRC."
echo "Building a zip anyway will use ~as much disk as the image folder."
read -p "Build the dataset zip? [y/N] " yn
[ "$yn" = "y" ] || { echo "skipped (recommended)."; exit 0; }

python "$REPO/dataset_tool.py" --source "$SRC" --dest "$DST" --width 256 --height 256
echo "dataset zip -> $DST"
