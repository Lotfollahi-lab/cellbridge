#!/bin/bash
set -euo pipefail

# Usage: submit_probes.sh [config]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG="${1:-$ROOT/configs/distill/distill_v0_vicreg_mild.yaml}"
case "$CONFIG" in
  /*) ;;
  *) CONFIG="$ROOT/$CONFIG" ;;
esac

DISTILL_VERSION="$(sed -nE 's/^[[:space:]]*distill_version:[[:space:]]*([^#[:space:]]+).*/\1/p' "$CONFIG" | head -n1)"
if [ "${DISTILL_VERSION:-v0}" = "v1" ]; then
  FEAT_DIM="$(sed -nE 's/^[[:space:]]*feat_dim:[[:space:]]*([0-9]+).*/\1/p' "$CONFIG" | head -n1)"
  NUM_PROTOTYPES="$(sed -nE 's/^[[:space:]]*num_prototypes:[[:space:]]*([0-9]+).*/\1/p' "$CONFIG" | head -n1)"
  ARMS=("prototype_penultimate_${FEAT_DIM:-512}" "prototype_probs_${NUM_PROTOTYPES:-256}")
else
  ARMS=(uni_baseline terra_teacher distilled concat_oracle)
fi
job_ids=()
for arm in "${ARMS[@]}"; do
  if ! output="$("$SCRIPT_DIR/submit_probe.sh" "$CONFIG" "$arm" 2>&1)"; then
    printf '%s\n' "$output" >&2
    exit 1
  fi
  printf '%s\n' "$output"
  job_id="$(printf '%s\n' "$output" | sed -nE 's/.*Job <([0-9]+)>.*/\1/p' | tail -n1)"
  [ -n "$job_id" ] || {
    echo "could not parse job ID for $arm" >&2
    exit 1
  }
  job_ids+=("$job_id")
done

dependency="$(printf 'done(%s)&&' "${job_ids[@]}")"
dependency="${dependency%&&}"
DEPENDENCY="$dependency" "$SCRIPT_DIR/submit_probe.sh" "$CONFIG" aggregate
