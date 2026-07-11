#!/bin/bash
set -euo pipefail

# Usage: submit_probes.sh [config]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG="${1:-$ROOT/configs/distill/distill_v0.yaml}"
case "$CONFIG" in
  /*) ;;
  *) CONFIG="$ROOT/$CONFIG" ;;
esac

ARMS=(uni_baseline terra_teacher distilled concat_oracle)
job_ids=()
for arm in "${ARMS[@]}"; do
  output="$("$SCRIPT_DIR/submit_probe.sh" "$CONFIG" "$arm")"
  printf '%s\n' "$output"
  job_id="$(printf '%s\n' "$output" | sed -nE 's/.*Job <([0-9]+)>.*/\1/p' | tail -n1)"
  [ -n "$job_id" ] || {
    echo "could not parse job ID for $arm" >&2
    exit 1
  }
  job_ids+=("$job_id")
done

dependency="$(printf 'ended(%s)&&' "${job_ids[@]}")"
dependency="${dependency%&&}"
DEPENDENCY="$dependency" "$SCRIPT_DIR/submit_probe.sh" "$CONFIG" aggregate
