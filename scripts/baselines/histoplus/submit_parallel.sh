#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
CONFIG="${1:-$ROOT/configs/baselines/histoplus.yaml}"
case "$CONFIG" in /*) ;; *) CONFIG="$ROOT/$CONFIG" ;; esac
SUBMIT="$ROOT/scripts/submit_job.sh"
mapfile -t HEST_SAMPLES < <("$SUBMIT" cohort "$CONFIG" \
  hest.label_test_fast_path hest.splits_path label_test)

submit_stage() {
  local suffix="$1" queue="$2" memory="$3" cores="$4" dependency="$5" gmem="$6"
  shift 6
  local dep_args=(); [ -z "$dependency" ] || dep_args=(--dependency "$dependency")
  local gpu_args=(); [ -z "$gmem" ] || gpu_args=(--gmem "$gmem")
  "$SUBMIT" submit --job "hp_${suffix}" --log-prefix "$suffix" \
    --queue "$queue" "${gpu_args[@]}" --memory "$memory" --cores "$cores" \
    --environment none "${dep_args[@]}" "$CONFIG" -- "$SCRIPT_DIR/run_in_env.sh" "$@"
}

GPU_QUEUE="${HISTOPLUS_GPU_QUEUE:-auto}"
GPU_GMEM="${HISTOPLUS_GPU_GMEM:-30000}"
GPU_MEM="${HISTOPLUS_GPU_MEM_GB:-96}"
GPU_CORES="${HISTOPLUS_GPU_NCORES:-12}"

prep="$(submit_stage prepare "$GPU_QUEUE" "$GPU_MEM" "$GPU_CORES" "" \
  "$GPU_GMEM" prepare --config "$CONFIG")"
echo "  PREPARE -> job $prep" >&2

IDS=()
for sample in "${HEST_SAMPLES[@]}"; do
  id="$(submit_stage "hest_${sample}" "$GPU_QUEUE" "$GPU_MEM" "$GPU_CORES" \
    "done($prep)" "$GPU_GMEM" infer-hest --config "$CONFIG" --samples "$sample")"
  echo "  HEST $sample -> job $id" >&2
  IDS+=("$id")
done

PHENO_SHARDS="${HISTOPLUS_PHENOCELL_SHARDS:-1}"
PANNUKE_SHARDS="${HISTOPLUS_PANNUKE_SHARDS:-4}"
for dataset in phenocell pannuke; do
  shards="$PHENO_SHARDS"; [ "$dataset" != pannuke ] || shards="$PANNUKE_SHARDS"
  for ((shard=0; shard<shards; shard++)); do
    id="$(submit_stage "${dataset}_${shard}" "$GPU_QUEUE" "$GPU_MEM" \
      "$GPU_CORES" "done($prep)" "$GPU_GMEM" infer-ood --config "$CONFIG" \
      --dataset "$dataset" --shard-index "$shard" --num-shards "$shards")"
    echo "  $dataset shard $shard/$shards -> job $id" >&2
    IDS+=("$id")
  done
done

dependency="$(printf 'done(%s)&&' "${IDS[@]}")"
submit_stage score "${HISTOPLUS_SCORE_QUEUE:-auto}" \
  "${HISTOPLUS_SCORE_MEM_GB:-128}" "${HISTOPLUS_SCORE_NCORES:-16}" \
  "${dependency%&&}" "${HISTOPLUS_SCORE_GMEM:-1000}" score --config "$CONFIG" --dataset all
