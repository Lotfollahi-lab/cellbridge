#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
CONFIG="${1:-$ROOT/configs/baselines/uni_selective_ft.yaml}"

module load ISG/conda
conda activate hest
cd "$ROOT"
PYTHONPATH=src python -m terra_multi.baselines.uni_selective_ft \
  --config "$CONFIG" check
