#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
CONFIG="${1:-$ROOT/configs/baselines/past.yaml}"
case "$CONFIG" in /*) ;; *) CONFIG="$ROOT/$CONFIG" ;; esac
[ ! -f "$ROOT/scripts/cluster_env.sh" ] || . "$ROOT/scripts/cluster_env.sh"
SUBMIT="$ROOT/scripts/submit_job.sh"
RUNNER="$SCRIPT_DIR/run_in_env.sh"
mapfile -t HEST_SAMPLES < <("$SUBMIT" cohort "$CONFIG" \
  hest.label_test_fast_path hest.splits_path label_test)

submit() {
  local suffix="$1" queue="$2" memory="$3" cores="$4" dependency="$5" gpu="$6"
  shift 6
  local dependencies=()
  [ -z "$dependency" ] || dependencies=(--dependency "$dependency")
  "$SUBMIT" submit --job "past_${suffix}" --log-prefix "$suffix" \
    --queue "$queue" --gpu "$gpu" --memory "$memory" --cores "$cores" \
    "${dependencies[@]}" "$CONFIG" -- "$RUNNER" "$@"
}

GPU_QUEUE="${PAST_GPU_QUEUE:-auto}"
GPU_SPEC="${PAST_GPU_SPEC:-num=1:gmem=12000}"

JOBS=()
for sample in "${HEST_SAMPLES[@]}"; do
  job="$(submit "hest_${sample}" "$GPU_QUEUE" 48 8 "" \
    "$GPU_SPEC" extract-hest --config "$CONFIG" --device cuda --samples "$sample")"
  echo "  HEST $sample -> job $job" >&2
  JOBS+=("$job")
done
for dataset in phenocell pannuke; do
  job="$(submit "$dataset" "$GPU_QUEUE" 48 1 "" "$GPU_SPEC" \
    "extract-$dataset" --config "$CONFIG" --device cuda)"
  echo "  ${dataset^^} -> job $job" >&2
  JOBS+=("$job")
done

dependency="$(printf 'done(%s)&&' "${JOBS[@]}")"
"$SUBMIT" submit --job past_probe --log-prefix probe \
  --queue "${PAST_PROBE_QUEUE:-inference}" --gpu "${PAST_PROBE_GPU_SPEC:-num=1:gmem=1000}" \
  --memory "${PAST_PROBE_MEM_GB:-128}" --cores "${PAST_PROBE_NCORES:-16}" \
  --dependency "${dependency%&&}" "$CONFIG" -- "$RUNNER" probe --config "$CONFIG" \
  --dataset all
