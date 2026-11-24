#!/usr/bin/env bash
# Score finished training runs, one summary row per run.
#   bash scripts/eval_runs.sh h3_silog_lr5e-6_bs4 h2_edge_yolo_l0.5 ...     NYU val (depth + boundary metrics)
#   SPLIT=test bash scripts/eval_runs.sh ...                                NYU test (final numbers only)
#   DATASET=ibims bash scripts/eval_runs.sh ...                             iBims-1 zero-shot
# Runs come from $RUNS_DIR (default runs/). A run already scored (per-image CSV present) is skipped.
set -euo pipefail
cd "$(dirname "$0")/../tools"
RUNS_DIR=${RUNS_DIR:-../runs}
DATASET=${DATASET:-nyu}
SPLIT=${SPLIT:-val}
for run in "$@"; do
  if [ "$DATASET" = ibims ]; then
    out="../results/ibims/${run}_per_image.csv"
    cmd=(python eval_ibims.py --ckpt "$RUNS_DIR/$run/best.pth" --tag "$run")
  else
    out="../results/nyu/${run}_${SPLIT}_filled_per_image.csv"
    cmd=(python eval_nyu.py --split "$SPLIT" --gt filled --ckpt "$RUNS_DIR/$run/best.pth" --tag "$run")
  fi
  if [ -f "$out" ]; then echo "$run already scored"; continue; fi
  "${cmd[@]}" 2>&1 | grep -v -e xFormers -e 'it/s' | tail -1
done
