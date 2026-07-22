#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
[ ! -f "$ROOT/scripts/cluster_env.sh" ] || . "$ROOT/scripts/cluster_env.sh"
CONFIG="${1:-$ROOT/configs/evaluation/phenocell.yaml}"
case "$CONFIG" in /*) ;; *) CONFIG="$ROOT/$CONFIG" ;; esac
DS="$(basename "$CONFIG" .yaml)"
DEP="${DEPENDENCY:-}"; [[ -z "$DEP" || "$DEP" == *"("* ]] || DEP="done($DEP)"
DEP_ARGS=(); [ -z "$DEP" ] || DEP_ARGS=(--dependency "$DEP")
"$ROOT/scripts/submit_job.sh" submit --job "${DS}_project" \
  --log-prefix project --queue "${OOD_PROJECT_QUEUE:-training-normal}" \
  --gpu "${OOD_PROJECT_GPU_SPEC:-num=1:gmem=10000}" \
  --memory "${OOD_PROJECT_MEM_GB:-32}" --cores "${OOD_PROJECT_NCORES:-4}" \
  "${DEP_ARGS[@]}" "$CONFIG" -- python3 -m terra_multi.evaluation.project_ood \
  --config "$CONFIG"
