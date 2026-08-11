#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
CONFIG="${1:-$ROOT/configs/baselines/spatialwhisperer.yaml}"
case "$CONFIG" in /*) ;; *) CONFIG="$ROOT/$CONFIG" ;; esac
SUBMIT="$ROOT/scripts/submit_job.sh"
mapfile -t SAMPLES < <("$SUBMIT" cohort "$CONFIG" \
  hest.label_test_fast_path hest.splits_path label_test)

submit_stage() {
  local suffix="$1" queue="$2" memory="$3" cores="$4" dependency="$5" gmem="$6"
  shift 6
  local dep_args=(); [ -z "$dependency" ] || dep_args=(--dependency "$dependency")
  local gpu_args=(); [ -z "$gmem" ] || gpu_args=(--gmem "$gmem")
  "$SUBMIT" submit "$CONFIG" --job "sw_${suffix}" --log-prefix "$suffix" \
    --queue "$queue" "${gpu_args[@]}" --memory "$memory" --cores "$cores" \
    --environment none "${dep_args[@]}" -- "$SCRIPT_DIR/run_in_env.sh" "$@"
}

GPU_QUEUE="${SPATIALWHISPERER_GPU_QUEUE:-auto}"
GPU_GMEM="${SPATIALWHISPERER_GPU_GMEM:-30000}"
prep="$(submit_stage prepare "${SPATIALWHISPERER_PREP_QUEUE:-auto}" \
  "${SPATIALWHISPERER_PREP_MEM_GB:-32}" "${SPATIALWHISPERER_PREP_NCORES:-4}" \
  "" "" prepare --config "$CONFIG")"
echo "  PREPARE -> job $prep" >&2

IDS=()
for sample in "${SAMPLES[@]}"; do
  id="$(submit_stage "hest_${sample}" "$GPU_QUEUE" \
    "${SPATIALWHISPERER_GPU_MEM_GB:-96}" "${SPATIALWHISPERER_GPU_NCORES:-12}" \
    "done($prep)" "$GPU_GMEM" extract-hest --config "$CONFIG" --device cuda \
    --samples "$sample")"
  echo "  HEST $sample -> job $id" >&2; IDS+=("$id")
done
id="$(submit_stage phenocell "$GPU_QUEUE" "${SPATIALWHISPERER_GPU_MEM_GB:-96}" \
  "${SPATIALWHISPERER_GPU_NCORES:-12}" "done($prep)" "$GPU_GMEM" \
  extract-phenocell --config "$CONFIG" --device cuda)"
echo "  PHENOCELL -> job $id" >&2; IDS+=("$id")

dependency="$(printf 'done(%s)&&' "${IDS[@]}")"
"$SUBMIT" submit "$CONFIG" --job sw_probe --log-prefix probe \
  --queue "${SPATIALWHISPERER_PROBE_QUEUE:-auto}" \
  --gmem "${SPATIALWHISPERER_PROBE_GMEM:-1000}" \
  --memory "${SPATIALWHISPERER_PROBE_MEM_GB:-128}" \
  --cores "${SPATIALWHISPERER_PROBE_NCORES:-16}" \
  --dependency "${dependency%&&}" -- python -m \
  cellbridge.baselines.spatialwhisperer probe --config "$CONFIG" --dataset all
