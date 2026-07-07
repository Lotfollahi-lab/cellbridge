#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG="${CONFIG:-$ROOT/configs/pipeline/pipeline.yaml}"

CONFIG="$CONFIG" "$SCRIPT_DIR/01_find_samples.sh"

OUTPUT_ROOT="$(sed -nE 's/^[[:space:]]*output_root:[[:space:]]*([^#[:space:]]+).*/\1/p' "$CONFIG" | head -n1)"
case "$OUTPUT_ROOT" in
  /*) ;;
  *) OUTPUT_ROOT="$ROOT/$OUTPUT_ROOT" ;;
esac
MANIFEST="$OUTPUT_ROOT/manifest.csv"

tail -n +2 "$MANIFEST" | cut -d, -f1 | while read -r sample_id; do
  [ -n "$sample_id" ] || continue
  CONFIG="$CONFIG" "$SCRIPT_DIR/run_sample_pipeline.sh" "$sample_id"
done
