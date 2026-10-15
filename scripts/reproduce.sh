#!/usr/bin/env bash
# Run one paper model with the directory layout consumed by notebooks 05–08.
set -euo pipefail

method="${1:-cellbridge}"
stage="${2:-check}"
case "$method" in
  cellbridge) version=uni_terra_four_layers_adapted_all_cells ;;
  lemon) version=vits8 ;;
  uni|virchow2|volta|past|spatialwhisperer) version=all ;;
  *) echo "Unknown paper method: $method" >&2; exit 2 ;;
esac
case "$stage" in
  check|extract|evaluate|conformal|run) ;;
  *) echo "Stage must be check, extract, evaluate, conformal, or run" >&2; exit 2 ;;
esac
root="${CELLBRIDGE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
: "${CELLBRIDGE_OUTPUTS_ROOT:?Set CELLBRIDGE_OUTPUTS_ROOT to an absolute output directory}"
version="${CELLBRIDGE_RUN_VERSION:-$version}"
run_root="$CELLBRIDGE_OUTPUTS_ROOT/reproducibility/inference/$method/$version"
extra=()
if [ "$method" = cellbridge ]; then
  extra+=(--arm lora_adapted)
  if [ -n "${CELLBRIDGE_CHECKPOINT:-}" ]; then
    extra+=(--checkpoint "$CELLBRIDGE_CHECKPOINT")
  fi
fi
if [ -n "${CELLBRIDGE_DEVICE:-}" ]; then extra+=(--device "$CELLBRIDGE_DEVICE"); fi

for dataset in nucls pannuke lizard phenocell; do
  python -m cellbridge.inference.run "$stage" \
    --config "$root/configs/inference/$dataset.yaml" --method "$method" \
    --output-dir "$run_root/$dataset" "${extra[@]}"
done

# These are the reviewed paper sections (Appendix A.1 excludes TENX143).
for sample in TENX116 TENX187 TENX191 TENX192 TENX193 TENX194 TENX195 TENX196 \
              TENX197 TENX198 TENX199 TENX200 TENX201 TENX202; do
  python -m cellbridge.inference.run "$stage" \
    --config "$root/configs/inference/hest.yaml" --method "$method" \
    --sample-id "$sample" --output-dir "$run_root/hest/$sample" "${extra[@]}"
done
