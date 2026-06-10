#!/usr/bin/env python3
"""Turn raw CelebV-HQ clips/images into a clean folder of 256x256 face images.

DISK-FRUGAL by default:
  - writes JPEG (q=95), ~10x smaller than PNG, for the *training* set
  - hard cap via --max-images (stops early; transfer learning needs only ~10-30k)
  - --delete-source frees each video right after it is processed
  - aborts early if the target disk has < --min-free-gb free, with a clear message

Handles two inputs transparently:
  - a folder of videos  (.mp4/.mov/...) -> samples frames at an interval
  - a folder of images  (.jpg/.png/...) -> uses them directly

Tip: if HOME is full, write somewhere with space:
  export FACEGEN_WORK=/scratch/$USER/facegen   (config picks it up)
  python scripts/02_preprocess.py --in data/raw --out $FACEGEN_WORK/data/faces256
"""
import argparse, shutil, sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
try:
    from tqdm import tqdm
except ImportError:                       # progress bar is optional
    def tqdm(it, **kw):
        return it

VID_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_files(root: Path, exts):
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in exts)


def free_gb(path: Path) -> float:
    path = path if path.exists() else path.parent
    return shutil.disk_usage(path).free / (1024 ** 3)


def get_detector():
    import torch
    from facenet_pytorch import MTCNN
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    return MTCNN(image_size=256, margin=40, post_process=False, device=dev)


def crop_face(det, pil_img):
    face = det(pil_img)
    if face is None:
        return None
    arr = face.permute(1, 2, 0).clamp(0, 255).byte().cpu().numpy()
    return Image.fromarray(arr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, type=Path)
    ap.add_argument("--out", dest="out", required=True, type=Path)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--format", choices=["jpg", "png"], default="jpg",
                    help="training-set image format (jpg is far smaller)")
    ap.add_argument("--quality", type=int, default=95, help="JPEG quality")
    ap.add_argument("--frame-interval", type=int, default=10,
                    help="sample every Nth frame from videos")
    ap.add_argument("--max-per-video", type=int, default=12,
                    help="cap frames per clip (limits near-duplicates + disk)")
    ap.add_argument("--max-images", type=int, default=20000,
                    help="HARD cap; stop once this many faces are written")
    ap.add_argument("--min-free-gb", type=float, default=5.0,
                    help="abort if target disk free space drops below this")
    ap.add_argument("--delete-source", action="store_true",
                    help="delete each video after processing to free space")
    ap.add_argument("--detect", action="store_true",
                    help="MTCNN face crop (default: assume already cropped)")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    # --- disk precheck: fail fast with an actionable message ---
    avail = free_gb(args.out)
    print(f"free space on target ({args.out}): {avail:.1f} GB")
    if avail < args.min_free_gb:
        sys.exit(
            f"ERROR: only {avail:.1f} GB free (< {args.min_free_gb} GB).\n"
            f"  Fix: point output to a bigger partition:\n"
            f"    export FACEGEN_WORK=/scratch/$USER/facegen\n"
            f"    python scripts/02_preprocess.py --in {args.inp} "
            f"--out $FACEGEN_WORK/data/faces256\n"
            f"  Locate space with: bash scripts/00_check_disk.sh"
        )

    det = get_detector() if args.detect else None
    ext = args.format
    save_kw = {"quality": args.quality} if ext == "jpg" else {}

    videos = list_files(args.inp, VID_EXT)
    images = list_files(args.inp, IMG_EXT)
    print(f"found {len(videos)} videos, {len(images)} images under {args.inp}")
    print(f"writing up to {args.max_images} {ext.upper()} faces -> {args.out}")

    idx = 0
    check_every = 200   # re-check disk periodically

    def emit(pil) -> bool:
        """Returns False to signal 'stop' (cap hit or disk low)."""
        nonlocal idx
        if idx >= args.max_images:
            return False
        if idx % check_every == 0 and free_gb(args.out) < args.min_free_gb:
            print(f"\nSTOP: free space < {args.min_free_gb} GB at {idx} images.")
            return False
        if det is not None:
            pil = crop_face(det, pil)
            if pil is None:
                return True
        pil = pil.convert("RGB").resize((args.size, args.size), Image.LANCZOS)
        pil.save(args.out / f"face_{idx:06d}.{ext}", **save_kw)
        idx += 1
        return True

    # images first
    for p in tqdm(images, desc="images"):
        try:
            if not emit(Image.open(p)):
                break
        except Exception as e:
            print(f"skip {p}: {e}", file=sys.stderr)

    # then videos
    stop = idx >= args.max_images
    for vp in tqdm(videos, desc="videos", disable=stop):
        if stop:
            break
        cap = cv2.VideoCapture(str(vp))
        fno, kept = 0, 0
        while kept < args.max_per_video:
            ok, frame = cap.read()
            if not ok:
                break
            if fno % args.frame_interval == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                if not emit(Image.fromarray(rgb)):
                    stop = True
                    break
                kept += 1
            fno += 1
        cap.release()
        if args.delete_source:
            try:
                vp.unlink()
            except OSError:
                pass

    print(f"\nDONE: wrote {idx} face images -> {args.out}")
    print(f"remaining free space: {free_gb(args.out):.1f} GB")
    if idx < 1000:
        print("WARNING: very few images; transfer learning wants a few thousand+.")


if __name__ == "__main__":
    main()
