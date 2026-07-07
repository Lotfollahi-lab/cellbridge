#!/bin/bash
set -euo pipefail

SAMPLE_ID="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

submit_step() {
  local dependency="$1"
  shift
  local out job
  if [ -n "$dependency" ]; then
    out="$(DEPENDENCY="$dependency" "$@" 2>&1)"
  else
    out="$("$@" 2>&1)"
  fi
  printf '%s\n' "$out" >&2
  job="$(printf '%s\n' "$out" | sed -nE 's/Job <([0-9]+)>.*/\1/p' | tail -n1)"
  [ -n "$job" ] || { echo "Could not parse bsub job id for $*" >&2; exit 1; }
  printf '%s' "$job"
}

qc_job="$(submit_step "" "$SCRIPT_DIR/02_quality_check.sh" "$SAMPLE_ID")"
labels_job="$(submit_step "$qc_job" "$SCRIPT_DIR/03_cell_labels.sh" "$SAMPLE_ID")"
terra_job="$(submit_step "$labels_job" "$SCRIPT_DIR/04_terra_embed.sh" "$SAMPLE_ID")"
uni_job="$(submit_step "$terra_job" "$SCRIPT_DIR/05_uni_embed.sh" "$SAMPLE_ID")"

printf '%s: qc=%s labels=%s terra=%s uni=%s\n' \
  "$SAMPLE_ID" "$qc_job" "$labels_job" "$terra_job" "$uni_job"
