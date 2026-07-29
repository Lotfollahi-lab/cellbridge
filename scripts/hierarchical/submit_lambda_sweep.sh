#!/bin/bash
# Training only: one independent job per lambda_hier setting.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

for lambda in 1 10; do
  config="$ROOT/configs/hierarchical/lora_combined_lambda${lambda}.yaml"
  echo "lambda_hier=${lambda}.0 -> job $("$ROOT/scripts/lora/submit_train.sh" "$config")" >&2
done
