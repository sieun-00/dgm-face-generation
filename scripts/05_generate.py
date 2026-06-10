#!/usr/bin/env python3
"""Deterministic image generation — THE reproducible inference code for top-10 check.

Loads a trained StyleGAN2-ADA snapshot (.pkl) and generates one image per seed.
Same network + same seeds + same truncation  =>  bit-identical images every run.

Usage:
  CUDA_VISIBLE_DEVICES=0 python scripts/05_generate.py \
      --network runs/00000-.../network-snapshot-001000.pkl \
      --outdir out/psi0.8 --seeds 0-999 --trunc 0.8 --size 256

Requires the StyleGAN repo on PYTHONPATH (this script adds it automatically).
"""
import argparse, os, re, sys
from pathlib import Path

import numpy as np
import PIL.Image
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "external" / "stylegan2-ada-pytorch"))
import dnnlib          # noqa: E402  (from the StyleGAN repo)
import legacy          # noqa: E402


def parse_seeds(spec: str):
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--network", required=True, help="path/URL to .pkl snapshot")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--seeds", default="0-999")
    ap.add_argument("--trunc", type=float, default=0.8, help="truncation psi")
    ap.add_argument("--size", type=int, default=256, help="output resolution")
    ap.add_argument("--format", default="png", choices=["png", "jpg"])
    args = ap.parse_args()

    seeds = parse_seeds(args.seeds)
    os.makedirs(args.outdir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"loading {args.network}")
    with dnnlib.util.open_url(args.network) as f:
        G = legacy.load_network_pkl(f)["G_ema"].to(device).eval()

    label = torch.zeros([1, G.c_dim], device=device)   # unconditional faces

    for i, seed in enumerate(seeds):
        # per-seed RNG => fully reproducible latent
        z = torch.from_numpy(
            np.random.RandomState(seed).randn(1, G.z_dim)
        ).to(device)
        with torch.no_grad():
            img = G(z, label, truncation_psi=args.trunc, noise_mode="const")
        img = (img.clamp(-1, 1) + 1) * (255 / 2)
        img = img.permute(0, 2, 3, 1).round().clamp(0, 255).to(torch.uint8)[0].cpu().numpy()
        pil = PIL.Image.fromarray(img, "RGB")
        if pil.size != (args.size, args.size):
            pil = pil.resize((args.size, args.size), PIL.Image.LANCZOS)
        # leaderboard expects img_0000.. naming with no subfolders
        pil.save(Path(args.outdir) / f"img_{i:04d}.{args.format}")
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(seeds)}")

    print(f"DONE: {len(seeds)} imgs -> {args.outdir}  (psi={args.trunc}, size={args.size})")
    print("Record this exact command + network pkl for the reproducibility submission.")


if __name__ == "__main__":
    main()
