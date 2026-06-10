#!/usr/bin/env bash
# Find where you have free space and what is eating your disk.
# Run this FIRST when you hit "no space left on device".
echo "=================== MOUNTED FILESYSTEMS (free space) ==================="
df -h | grep -vE '^(tmpfs|udev|overlay)' | sort -k4 -h
echo
echo ">>> Pick a mount with lots of 'Avail' (often /data, /scratch, /mnt/...)."
echo ">>> Set it as the work root, e.g.:"
echo "      export FACEGEN_WORK=/scratch/$USER/facegen"
echo

echo "=================== HOME QUOTA (if any) ==============================="
quota -s 2>/dev/null || echo "(no quota command / no quota set)"
echo

echo "=================== BIGGEST DIRS UNDER HOME =========================="
du -h -d1 "$HOME" 2>/dev/null | sort -rh | head -15
echo

echo "=================== THIS REPO'S DATA DIRS ============================"
cd "$(dirname "$0")/.."
du -sh data runs out external 2>/dev/null
echo
echo "Tip: the CelebV-HQ raw video download is the usual culprit (100s of GB)."
echo "You do NOT need it all — a few thousand clips/frames is plenty."
