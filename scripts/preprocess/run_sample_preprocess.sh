#!/bin/bash
set -euo pipefail
SAMPLE="${1:?usage: run_sample_preprocess.sh <sample-id>}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
qc="$($SCRIPT_DIR/02_quality_check.sh "$SAMPLE")"
labels="$(DEPENDENCY="$qc" "$SCRIPT_DIR/03_cell_labels.sh" "$SAMPLE")"
terra="$(DEPENDENCY="$labels" "$SCRIPT_DIR/04_terra_embed.sh" "$SAMPLE")"
uni="$(DEPENDENCY="$terra" "$SCRIPT_DIR/05_uni_embed.sh" "$SAMPLE")"
printf '%s: qc=%s labels=%s terra=%s uni=%s\n' "$SAMPLE" "$qc" "$labels" "$terra" "$uni"
