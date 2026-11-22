#!/usr/bin/env bash
# Fetch everything that is not stored in git: upstream code, checkpoints, datasets.
# Safe to re-run; finished steps are skipped.
set -euo pipefail
cd "$(dirname "$0")/.."

DA_COMMIT=a561b84
HF=https://huggingface.co

# 1. Official Depth Anything V2 code, pinned.
if [ ! -d Depth-Anything-V2 ]; then
  git clone https://github.com/DepthAnything/Depth-Anything-V2.git
  git -C Depth-Anything-V2 checkout -q "$DA_COMMIT"
fi

# 2. Checkpoints (ViT-S relative + indoor metric).
mkdir -p checkpoints
get() { [ -s "$2" ] || curl -L --fail -o "$2" "$1"; }
get "$HF/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth" \
    checkpoints/depth_anything_v2_vits.pth
get "$HF/depth-anything/Depth-Anything-V2-Metric-Hypersim-Small/resolve/main/depth_anything_v2_metric_hypersim_vits.pth" \
    checkpoints/depth_anything_v2_metric_hypersim_vits.pth

# 3. DA-2K relative-depth benchmark (~680 MB).
if [ ! -f data/DA-2K/DA-2K/annotations.json ]; then
  mkdir -p data/DA-2K
  get "$HF/datasets/depth-anything/DA-2K/resolve/main/DA-2K.zip" data/DA-2K.zip
  unzip -q data/DA-2K.zip -d data/DA-2K && rm -rf data/DA-2K.zip data/DA-2K/__MACOSX
fi

# 4. NYU Depth V2 labeled set (~2.9 GB): filled + raw depth, instances, scene ids.
mkdir -p data/nyu_mat
get http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat \
    data/nyu_mat/nyu_depth_v2_labeled.mat

echo "Setup done. NYU PNGs (data/nyu/{rgb,depth,masks}) will be generated from the .mat by tools/prepare_nyu.py (planned)."
