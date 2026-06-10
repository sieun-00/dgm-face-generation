#!/usr/bin/env bash
# Generate submissions across truncation psi values to A/B test on the leaderboard.
# psi down  -> better FID/fidelity, worse diversity (TopPR/recall).
# Use the 4 daily submissions to find the psi with the best *Total* (avg rank).
set -e
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=0

NET=${1:?usage: 07_psi_sweep.sh <network.pkl>}

for PSI in 0.7 0.8 0.9 1.0; do
  echo "=== psi=$PSI ==="
  python scripts/05_generate.py --network "$NET" \
      --outdir "out/psi$PSI" --seeds 0-999 --trunc "$PSI" --size 256
  python scripts/06_package_submission.py \
      --indir "out/psi$PSI" --zip "submission_psi$PSI.zip" --jpeg-quality 95
done

echo "Built submission_psi{0.7,0.8,0.9,1.0}.zip — upload and compare Total."
