#!/usr/bin/env python3
"""Stream-extract the first N video clips from a (large) CelebV-HQ tar.

We only need ~2k clips to make 20k training frames, so there's no point
un-taring all 41 GB. This iterates the tar sequentially and stops early.
Works on a plain .tar; for .tar.gz pass the path too (tarfile auto-detects).
"""
import argparse, tarfile
from pathlib import Path

VID_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tar", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--max-videos", type=int, default=2500)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    n = 0
    # streaming mode: 'r|*' reads sequentially without building a full index
    with tarfile.open(args.tar, "r|*") as tar:
        for m in tar:
            if not m.isfile():
                continue
            if Path(m.name).suffix.lower() not in VID_EXT:
                continue
            # flatten into out/ to avoid odd nested paths
            m.name = Path(m.name).name
            tar.extract(m, args.out)
            n += 1
            if n % 200 == 0:
                print(f"  extracted {n} clips")
            if n >= args.max_videos:
                break
    print(f"DONE: extracted {n} clips -> {args.out}")


if __name__ == "__main__":
    main()
