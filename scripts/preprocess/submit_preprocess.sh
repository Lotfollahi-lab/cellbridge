#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG="${CONFIG:-$ROOT/configs/preprocess/preprocess.yaml}"
CONFIG="$CONFIG" "$SCRIPT_DIR/01_find_samples.sh"
OUTPUT="$($ROOT/scripts/submit_job.sh value "$CONFIG" output_root)"
case "$OUTPUT" in /*) ;; *) OUTPUT="$ROOT/$OUTPUT" ;; esac
while IFS=, read -r sample _; do
  [ -z "$sample" ] || CONFIG="$CONFIG" "$SCRIPT_DIR/run_sample_preprocess.sh" "$sample"
done < <(tail -n +2 "$OUTPUT/manifest.csv")
