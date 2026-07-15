#!/bin/bash
# Report cell-type classification metrics + figures (runs locally, not via bsub).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
[ -f "$ROOT/scripts/cluster_env.sh" ] && . "$ROOT/scripts/cluster_env.sh"

CONFIG="${CONFIG:-$ROOT/configs/exploration/celltype_tenx116.yaml}"
OUTPUT_DIR="$(sed -nE 's/^[[:space:]]*output_dir:[[:space:]]*([^#[:space:]]+).*/\1/p' "$CONFIG" | head -n1)"
case "$OUTPUT_DIR" in
  /*) : ;;
  *) OUTPUT_DIR="$ROOT/$OUTPUT_DIR" ;;
esac

module load cellgen/uv
source "${ENV_PATH:-/lustre/scratch126/cellgen/lotfollahi/so25/envs/terra-dev}/bin/activate"
cd "$ROOT"
PYTHONPATH=src python3 -m terra_multi.exploration.compare_celltype --results_dir "$OUTPUT_DIR"
