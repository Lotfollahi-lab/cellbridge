#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
[ ! -f "$ROOT/scripts/cluster_env.sh" ] || . "$ROOT/scripts/cluster_env.sh"
CONFIG="${1:-$ROOT/configs/evaluation/phenocell.yaml}"
case "$CONFIG" in /*) ;; *) CONFIG="$ROOT/$CONFIG" ;; esac
DEP="${DEPENDENCY:-}"; [[ -z "$DEP" || "$DEP" == *"("* ]] || DEP="done($DEP)"
DEP_ARGS=(); [ -z "$DEP" ] || DEP_ARGS=(--dependency "$DEP")
"$ROOT/scripts/submit_job.sh" submit "$CONFIG" --job phenocell_embed \
  --log-prefix embed --output-key embedding.output_dir \
  --queue "${PHENOCELL_QUEUE:-training-normal}" \
  --gpu "${PHENOCELL_GPU_SPEC:-num=1:gmem=30000}" \
  --memory "${PHENOCELL_EMBED_MEM_GB:-64}" --cores "${PHENOCELL_EMBED_NCORES:-8}" \
  --environment hest "${DEP_ARGS[@]}" -- python \
  -m terra_multi.evaluation.embed_phenocell --config "$CONFIG"
