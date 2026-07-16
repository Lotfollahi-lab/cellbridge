#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG="${1:-$ROOT/configs/jepa/spatial_block.yaml}"
case "$CONFIG" in /*) ;; *) CONFIG="$ROOT/$CONFIG" ;; esac
[ -f "$CONFIG" ] || { echo "config not found: $CONFIG" >&2; exit 1; }

train_job="${JEPA_TRAIN_JOB_ID:-}"
if [ -n "$train_job" ]; then
  [[ "$train_job" =~ ^[0-9]+$ ]] || {
    echo "JEPA_TRAIN_JOB_ID must be a numeric LSF job ID" >&2
    exit 1
  }
  echo "reusing training job ${train_job}"
else
  if ! train_output="$("$SCRIPT_DIR/submit_train.sh" "$CONFIG" 2>&1)"; then
    printf '%s\n' "$train_output" >&2
    exit 1
  fi
  printf '%s\n' "$train_output"
  train_job="$(printf '%s\n' "$train_output" | sed -nE 's/.*Job <([0-9]+)>.*/\1/p' | tail -n1)"
  [ -n "$train_job" ] || {
    echo "could not parse training job ID" >&2
    exit 1
  }
fi

FAST_PATH="$(sed -nE 's/^[[:space:]]*label_test_fast_path:[[:space:]]*([^#[:space:]]+).*/\1/p' "$CONFIG" | head -n1)"
[ -f "$FAST_PATH" ] || { echo "fast split not found: $FAST_PATH" >&2; exit 1; }
mapfile -t SAMPLES < <(tail -n +2 "$FAST_PATH" | cut -d, -f1 | grep -v '^$')
[ "${#SAMPLES[@]}" -gt 0 ] || { echo "fast split is empty" >&2; exit 1; }

probe_job_ids=()
for sample in "${SAMPLES[@]}"; do
  if ! extract_output="$(DEPENDENCY="done(${train_job})" \
    "$SCRIPT_DIR/submit_extract.sh" "$CONFIG" "$sample" 2>&1)"; then
    printf '%s\n' "$extract_output" >&2
    exit 1
  fi
  printf '%s\n' "$extract_output"
  extract_job="$(printf '%s\n' "$extract_output" | sed -nE 's/.*Job <([0-9]+)>.*/\1/p' | tail -n1)"
  [ -n "$extract_job" ] || {
    echo "could not parse extraction job ID for $sample" >&2
    exit 1
  }

  if ! probe_output="$(DEPENDENCY="done(${extract_job})" \
    "$SCRIPT_DIR/submit_probe.sh" "$CONFIG" "$sample" 2>&1)"; then
    printf '%s\n' "$probe_output" >&2
    exit 1
  fi
  printf '%s\n' "$probe_output"
  probe_job="$(printf '%s\n' "$probe_output" | sed -nE 's/.*Job <([0-9]+)>.*/\1/p' | tail -n1)"
  [ -n "$probe_job" ] || {
    echo "could not parse probe job ID for $sample" >&2
    exit 1
  }
  probe_job_ids+=("$probe_job")
done

dependency="$(printf 'done(%s)&&' "${probe_job_ids[@]}")"
dependency="${dependency%&&}"
DEPENDENCY="$dependency" "$SCRIPT_DIR/submit_aggregate.sh" "$CONFIG"
