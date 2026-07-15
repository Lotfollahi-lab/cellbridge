#!/bin/bash
set -euo pipefail

# Fan out the evaluation of every LoRA config at once.
# Usage: submit_eval_suite_parallel.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

CONFIGS=(
  frozen_head_only
  contrastive_k0
  contrastive_last4
  contrastive_gex_last4
)

for name in "${CONFIGS[@]}"; do
  echo "=== ${name} ==="
  "$SCRIPT_DIR/submit_eval_parallel.sh" "$ROOT/configs/lora/${name}.yaml"
done
