#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
[ ! -f "$ROOT/scripts/cluster_env.sh" ] || . "$ROOT/scripts/cluster_env.sh"
CONFIG="${1:-$ROOT/configs/evaluation/phenocell.yaml}"
case "$CONFIG" in /*) ;; *) CONFIG="$ROOT/$CONFIG" ;; esac
DS="$(basename "$CONFIG" .yaml)"
DEP="${DEPENDENCY:-}"; [[ -z "$DEP" || "$DEP" == *"("* ]] || DEP="done($DEP)"
DEP_ARGS=(); [ -z "$DEP" ] || DEP_ARGS=(--dependency "$DEP")
"$ROOT/scripts/submit_job.sh" submit --job "${DS}_embed" \
  --log-prefix embed --output-key embedding.output_dir \
  --queue "${OOD_QUEUE:-training-normal}" \
  --gpu "${OOD_GPU_SPEC:-num=1:gmem=30000}" \
  --memory "${OOD_EMBED_MEM_GB:-64}" --cores "${OOD_EMBED_NCORES:-8}" \
  --environment hest "${DEP_ARGS[@]}" "$CONFIG" -- python \
  -m cellbridge.evaluation.embed_ood --config "$CONFIG"
