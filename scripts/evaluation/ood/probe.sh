#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
[ ! -f "$ROOT/scripts/cluster_env.sh" ] || . "$ROOT/scripts/cluster_env.sh"
CONFIG="${1:-$ROOT/configs/evaluation/phenocell.yaml}"
ARM="${2:-}"; case "$ARM" in ""|uni|distilled) ;; *) echo "arm must be uni or distilled" >&2; exit 1 ;; esac
case "$CONFIG" in /*) ;; *) CONFIG="$ROOT/$CONFIG" ;; esac
DS="$(basename "$CONFIG" .yaml)"
DEP="${DEPENDENCY:-}"; [[ -z "$DEP" || "$DEP" == *"("* ]] || DEP="done($DEP)"
DEP_ARGS=(); [ -z "$DEP" ] || DEP_ARGS=(--dependency "$DEP")
ARM_ARGS=(); [ -z "$ARM" ] || ARM_ARGS=(--arm "$ARM")
"$ROOT/scripts/submit_job.sh" submit \
  --job "${DS}_probe${ARM:+_$ARM}" --log-prefix "probe${ARM:+_$ARM}" \
  --queue "${OOD_PROBE_QUEUE:-training-normal}" --gpu - \
  --memory "${OOD_PROBE_MEM_GB:-64}" --cores "${OOD_PROBE_NCORES:-8}" \
  "${DEP_ARGS[@]}" "$CONFIG" -- python3 -m terra_multi.evaluation.probe_ood \
  --config "$CONFIG" "${ARM_ARGS[@]}"
