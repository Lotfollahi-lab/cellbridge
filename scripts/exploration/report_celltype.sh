#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
[ ! -f "$ROOT/scripts/cluster_env.sh" ] || . "$ROOT/scripts/cluster_env.sh"
CONFIG="${CONFIG:-$ROOT/configs/exploration/celltype_tenx116.yaml}"
OUTPUT="$($ROOT/scripts/submit_job.sh value "$CONFIG" output_dir)"
case "$OUTPUT" in /*) ;; *) OUTPUT="$ROOT/$OUTPUT" ;; esac
module load cellgen/uv
ENV_ROOT="${CELLBRIDGE_ENV_ROOT:-$ROOT/../../envs}"
source "${ENV_PATH:-$ENV_ROOT/terra-dev}/bin/activate"
cd "$ROOT"
PYTHONPATH=src python3 -m cellbridge.exploration.compare_celltype --results_dir "$OUTPUT"
