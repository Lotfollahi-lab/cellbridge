#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

for name in baseline negative_downweighting hierarchy_ranking combined; do
  "$SCRIPT_DIR/submit_experiment.sh" \
    "$ROOT/configs/hierarchical/lora_${name}.yaml"
done
