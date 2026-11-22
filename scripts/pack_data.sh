#!/usr/bin/env bash
# Bundle the prepared datasets into one tar (~2 GB) so Colab sessions can restore them
# from Google Drive in a minute instead of re-downloading and re-extracting.
#
#   bash scripts/pack_data.sh [OUT.tar]     # default: $DATA_ROOT/iad_data.tar
# The .mat is left out: everything needed is already extracted into nyu/.
set -euo pipefail
cd "$(dirname "$0")/.."
DATA_ROOT=${DATA_ROOT:-$PWD/data}
OUT=${1:-$DATA_ROOT/iad_data.tar}
tar -C "$DATA_ROOT" -cf "$OUT" --dereference nyu DA-2K ibims
ls -lh "$OUT"
