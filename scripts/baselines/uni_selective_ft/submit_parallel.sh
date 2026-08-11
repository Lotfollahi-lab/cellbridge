#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
CONFIG="${1:-$ROOT/configs/baselines/uni_selective_ft.yaml}"
case "$CONFIG" in /*) ;; *) CONFIG="$ROOT/$CONFIG" ;; esac
SUBMIT="$ROOT/scripts/submit_job.sh"

mapfile -t SAMPLES < <("$SUBMIT" cohort "$CONFIG" \
  hest.label_test_fast_path hest.splits_path label_test)

submit_job() {
  local suffix="$1" queue="$2" memory="$3" cores="$4" dependency="$5" gmem="$6"
  shift 6
  local dependency_args=()
  [ -z "$dependency" ] || dependency_args=(--dependency "$dependency")
  local gpu_args=(); [ -z "$gmem" ] || gpu_args=(--gmem "$gmem")
  "$SUBMIT" submit --job "uft_${suffix}" --log-prefix "$suffix" \
    --queue "$queue" "${gpu_args[@]}" --memory "$memory" --cores "$cores" \
    --environment hest "${dependency_args[@]}" "$CONFIG" -- \
    python -m cellbridge.baselines.uni_selective_ft --config "$CONFIG" "$@"
}

PREP_QUEUE="${UNI_FT_PREP_QUEUE:-auto}"
prep_pheno="$(submit_job prep_pheno "$PREP_QUEUE" \
  "${UNI_FT_PREP_MEM_GB:-32}" "${UNI_FT_PREP_NCORES:-4}" "" "" \
  prepare-ood --dataset phenocell)"
echo "  PREP PhenoCell -> job $prep_pheno" >&2
prep_pan="$(submit_job prep_pan "$PREP_QUEUE" \
  "${UNI_FT_PREP_MEM_GB:-32}" "${UNI_FT_PREP_NCORES:-4}" "" "" \
  prepare-ood --dataset pannuke)"
echo "  PREP PanNuke -> job $prep_pan" >&2

GPU_QUEUE="${UNI_FT_GPU_QUEUE:-auto}"
GPU_GMEM="${UNI_FT_GPU_GMEM:-40000}"
GPU_MEM="${UNI_FT_GPU_MEM_GB:-96}"
GPU_CORES="${UNI_FT_GPU_NCORES:-12}"
IDS=()
for sample in "${SAMPLES[@]}"; do
  id="$(submit_job "hest_${sample}" "$GPU_QUEUE" "$GPU_MEM" "$GPU_CORES" \
    "" "$GPU_GMEM" train-hest --sample "$sample")"
  echo "  HEST $sample -> job $id" >&2
  IDS+=("$id")
done

for task in all immune; do
  id="$(submit_job "pheno_${task}" "$GPU_QUEUE" "$GPU_MEM" "$GPU_CORES" \
    "done($prep_pheno)" "$GPU_GMEM" train-ood --dataset phenocell --task "$task")"
  echo "  PhenoCell $task -> job $id" >&2
  IDS+=("$id")
done

id="$(submit_job pan_all "$GPU_QUEUE" "$GPU_MEM" "$GPU_CORES" \
  "done($prep_pan)" "$GPU_GMEM" train-ood --dataset pannuke --task all)"
echo "  PanNuke all -> job $id" >&2
IDS+=("$id")

dependency="$(printf 'done(%s)&&' "${IDS[@]}")"
"$SUBMIT" submit --job uft_aggregate --log-prefix aggregate \
  --queue "${UNI_FT_SUMMARY_QUEUE:-auto}" --memory 8 --cores 1 \
  --environment hest --dependency "${dependency%&&}" "$CONFIG" -- \
  python -m cellbridge.baselines.uni_selective_ft --config "$CONFIG" aggregate
