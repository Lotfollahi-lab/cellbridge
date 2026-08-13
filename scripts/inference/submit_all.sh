#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

datasets=(pannuke lizard phenocell immunocto nucls)
for dataset in "${datasets[@]}"; do
  echo "Submitting $dataset" >&2
  "$SCRIPT_DIR/submit.sh" "$ROOT/configs/inference/$dataset.yaml" "$@"
done
