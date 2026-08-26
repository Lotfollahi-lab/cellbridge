#!/bin/bash
# Submit one or more cell-count scaling arms.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
SUBMIT_TRAIN="$ROOT/scripts/train/submit_train.sh"

arms=("$@")
if [ "${#arms[@]}" -eq 0 ]; then
  arms=(cells_5k cells_10k cells_50k cells_all)
fi

for arm in "${arms[@]}"; do
  config="$ROOT/configs/train/lora/scaling/${arm}.yaml"
  if [ ! -f "$config" ]; then
    echo "missing scaling config: $config" >&2
    exit 1
  fi

  job_id="$(LORA_QUEUE="${LORA_QUEUE:-training-normal}" \
    "$SUBMIT_TRAIN" "$config")"
  echo "$arm $job_id"
done
