#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG="${1:-$ROOT/configs/exploration/novae_phenotype_structure.yaml}"
case "$CONFIG" in /*) ;; *) CONFIG="$ROOT/$CONFIG" ;; esac
SAMPLES=(
  TENX191 TENX192 TENX193 TENX194 TENX195 TENX196
  TENX197 TENX198 TENX199 TENX200 TENX201 TENX202
)
JOBS=()
for sample in "${SAMPLES[@]}"; do
  job="$($SCRIPT_DIR/submit_extract.sh "$CONFIG" "$sample")"
  echo "NOVAE $sample -> job $job" >&2
  JOBS+=("$job")
done
dependency="$(printf 'done(%s)&&' "${JOBS[@]}")"
DEPENDENCY="${dependency%&&}" "$SCRIPT_DIR/submit_analysis.sh" "$CONFIG"
