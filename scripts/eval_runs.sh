#!/usr/bin/env bash
# Score finished training runs on NYU val (depth + boundary metrics), one summary row per run.
#   bash scripts/eval_runs.sh h3_silog_lr5e-6_bs4 h2_edge_yolo_l0.5 ...
# Runs come from $RUNS_DIR (default runs/). A run already scored (per-image CSV present) is skipped.
set -euo pipefail
cd "$(dirname "$0")/../tools"
RUNS_DIR=${RUNS_DIR:-../runs}
SPLIT=${SPLIT:-val}
for run in "$@"; do
  if [ -f "../results/nyu/${run}_${SPLIT}_filled_per_image.csv" ]; then echo "$run already scored"; continue; fi
  python eval_nyu.py --split "$SPLIT" --gt filled --ckpt "$RUNS_DIR/$run/best.pth" --tag "$run" 2>&1 \
    | grep -v -e xFormers -e 'it/s' | tail -1
done
