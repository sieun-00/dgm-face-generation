#!/usr/bin/env python3
"""Package generated images into a leaderboard-valid submission zip.

Enforces the rules so you don't waste a daily submission on a -1 error:
  - exactly <=1000 images (1000 recommended), images at zip ROOT (no subfolders)
  - JPEG/PNG only, total < 200 MB
  - re-encodes to clean JPEG if --jpeg-quality is set (smaller zip)
"""
import argparse, sys, zipfile
from pathlib import Path

from PIL import Image

MAX_BYTES = 200 * 1024 * 1024
EXTS = {".png", ".jpg", ".jpeg"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", required=True, type=Path)
    ap.add_argument("--zip", required=True, type=Path)
    ap.add_argument("--max", type=int, default=1000)
    ap.add_argument("--jpeg-quality", type=int, default=0,
                    help="if >0, re-encode all to JPEG at this quality to shrink zip")
    args = ap.parse_args()

    imgs = sorted(p for p in args.indir.iterdir() if p.suffix.lower() in EXTS)
    if not imgs:
        sys.exit(f"ERROR: no images in {args.indir}")
    if len(imgs) > args.max:
        print(f"trimming {len(imgs)} -> {args.max}")
        imgs = imgs[: args.max]
    if len(imgs) < 1000:
        print(f"WARNING: only {len(imgs)} images (<1000). Leaderboard recommends 1000.")

    args.zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.zip, "w", zipfile.ZIP_STORED) as zf:
        for i, p in enumerate(imgs):
            arcname = f"img_{i:04d}"  # flat root, sequential names
            if args.jpeg_quality > 0:
                tmp = p.with_suffix(".repack.jpg")
                Image.open(p).convert("RGB").save(
                    tmp, "JPEG", quality=args.jpeg_quality)
                zf.write(tmp, arcname + ".jpg")
                tmp.unlink()
            else:
                zf.write(p, arcname + p.suffix.lower())

    size = args.zip.stat().st_size
    print(f"zip: {args.zip}  ({size/1e6:.1f} MB, {len(imgs)} images)")
    if size > MAX_BYTES:
        print("ERROR: zip > 200 MB. Re-run with --jpeg-quality 92 (or lower).")
        sys.exit(1)
    print("OK — ready to upload to https://leaderboard.lait-lab.com")


if __name__ == "__main__":
    main()
