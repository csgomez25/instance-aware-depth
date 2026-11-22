#!/usr/bin/env bash
# Fetch and prepare everything that is not stored in git. Safe to re-run; finished steps are skipped.
#
#   bash scripts/setup.sh            # everything (~4 GB download, a few minutes of GPU for masks)
#   DATA_ROOT=/some/dir bash scripts/setup.sh   # put datasets somewhere else
#   SKIP_MASKS=1 bash scripts/setup.sh          # no YOLO mask generation
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PYTHON:-python}
DATA_ROOT=${DATA_ROOT:-$PWD/data}
CKPT_DIR=${CKPT_DIR:-$PWD/checkpoints}
export DATA_ROOT CKPT_DIR
DA_COMMIT=a561b84
HF=https://huggingface.co
get() { [ -s "$2" ] || curl -L --fail -o "$2" "$1"; }
mkdir -p "$DATA_ROOT" "$CKPT_DIR"

echo "== 1/6 Depth Anything V2 code (pinned to $DA_COMMIT)"
if [ ! -d Depth-Anything-V2 ]; then
  git clone -q https://github.com/DepthAnything/Depth-Anything-V2.git
  git -C Depth-Anything-V2 checkout -q "$DA_COMMIT"
fi

echo "== 2/6 Checkpoints"
get "$HF/depth-anything/Depth-Anything-V2-Small/resolve/main/depth_anything_v2_vits.pth" \
    "$CKPT_DIR/depth_anything_v2_vits.pth"
get "$HF/depth-anything/Depth-Anything-V2-Metric-Hypersim-Small/resolve/main/depth_anything_v2_metric_hypersim_vits.pth" \
    "$CKPT_DIR/depth_anything_v2_metric_hypersim_vits.pth"

echo "== 3/6 DA-2K (~680 MB)"
if [ ! -f "$DATA_ROOT/DA-2K/DA-2K/annotations.json" ]; then
  mkdir -p "$DATA_ROOT/DA-2K"
  get "$HF/datasets/depth-anything/DA-2K/resolve/main/DA-2K.zip" "$DATA_ROOT/DA-2K.zip"
  unzip -q "$DATA_ROOT/DA-2K.zip" -d "$DATA_ROOT/DA-2K" && rm -rf "$DATA_ROOT/DA-2K.zip" "$DATA_ROOT/DA-2K/__MACOSX"
fi

echo "== 4/6 iBims-1 core (~55 MB, rsync from TUM; public password m1455541)"
if [ ! -d "$DATA_ROOT/ibims/ibims1_core_raw" ]; then
  command -v rsync >/dev/null || { echo "rsync missing: apt-get install rsync"; exit 1; }
  mkdir -p "$DATA_ROOT/ibims" && cd "$DATA_ROOT/ibims"
  RSYNC_PASSWORD=m1455541 rsync -q \
    rsync://m1455541@dataserv.ub.tum.de/m1455541/{ibims1_core_raw.zip,evaluation_scripts.zip,imagelist.txt,Ibims_Dataset.sha512} .
  grep -E "core_raw|evaluation_scripts" Ibims_Dataset.sha512 | sha512sum -c -
  unzip -qo ibims1_core_raw.zip && unzip -qo evaluation_scripts.zip -d evaluation_scripts
  rm -f ibims1_core_raw.zip evaluation_scripts.zip
  cd - >/dev/null
fi

echo "== 5/6 NYU Depth V2 labeled (~2.9 GB) -> image files"
if [ "$(ls "$DATA_ROOT/nyu/instances" 2>/dev/null | wc -l)" -ne 1449 ]; then
  mkdir -p "$DATA_ROOT/nyu_mat"
  get http://horatio.cs.nyu.edu/mit/silberman/nyu_depth_v2/nyu_depth_v2_labeled.mat \
      "$DATA_ROOT/nyu_mat/nyu_depth_v2_labeled.mat"
  (cd tools && "$PY" prepare_nyu.py)
fi

echo "== 6/6 YOLO11s-seg masks"
if [ -z "${SKIP_MASKS:-}" ]; then (cd tools && "$PY" make_masks.py); fi

echo "Setup done. Data in $DATA_ROOT, checkpoints in $CKPT_DIR."
