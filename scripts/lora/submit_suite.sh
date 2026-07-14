#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

configs=(
  "$ROOT/configs/lora/frozen_head_only.yaml"
  "$ROOT/configs/lora/contrastive_k0.yaml"
  "$ROOT/configs/lora/contrastive_last4.yaml"
  "$ROOT/configs/lora/contrastive_gex_last4.yaml"
)

for config in "${configs[@]}"; do
  "$SCRIPT_DIR/submit_experiment.sh" "$config"
done
