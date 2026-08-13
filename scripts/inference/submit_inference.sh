#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG="${1:-$ROOT/configs/inference/phenocell.yaml}"
case "$CONFIG" in /*) ;; *) CONFIG="$ROOT/$CONFIG" ;; esac
if [ "$#" -gt 0 ]; then shift; fi

SUBMIT="$ROOT/scripts/submit_job.sh"
RUNNER="$SCRIPT_DIR/run_in_env.sh"
TERRA_ENV="${TERRA_ENV:-$ROOT/../../envs/terra-dev}"

if [ "$#" -gt 0 ]; then
  methods=("$@")
else
  mapfile -t methods < <(
    PYTHONPATH="$ROOT/src" "$TERRA_ENV/bin/python" - "$CONFIG" <<'PY'
import sys
from cellbridge.config import load_config

print(*load_config(sys.argv[1])["methods"], sep="\n")
PY
  )
fi

check_arguments=()
for method in "${methods[@]}"; do
  check_arguments+=(--method "$method")
done
"$RUNNER" core check "$CONFIG" "${check_arguments[@]}"

dataset="$("$SUBMIT" value "$CONFIG" dataset)"
case "$dataset" in
  immunocto) evaluation_memory="${INFERENCE_CPU_MEM_GB:-256}" ;;
  *) evaluation_memory="${INFERENCE_CPU_MEM_GB:-64}" ;;
esac
jobs=()
for method in "${methods[@]}"; do
  case "$method" in
    uni) gmem=16000 ;;
    cellbridge) gmem=24000 ;;
    past) gmem=16000 ;;
    volta) gmem=4000 ;;
    spatialwhisperer) gmem=30000 ;;
  esac

  job="$("$SUBMIT" submit "$CONFIG" \
    --job "${dataset}_${method}" --log-prefix "extract_${method}" \
    --queue "${INFERENCE_GPU_QUEUE:-auto}" \
    --gmem "$gmem" --memory "${INFERENCE_GPU_MEM_GB:-64}" \
    --cores "${INFERENCE_GPU_NCORES:-8}" --environment none -- \
    "$RUNNER" "$method" extract "$CONFIG" --device cuda)"
  echo "EXTRACT $method -> job $job" >&2
  jobs+=("$job")
done

dependency="$(printf 'done(%s)&&' "${jobs[@]}")"
evaluate_arguments=()
for method in "${methods[@]}"; do
  evaluate_arguments+=(--method "$method")
done
job="$("$SUBMIT" submit "$CONFIG" \
  --job "${dataset}_evaluate" --log-prefix evaluate \
  --queue "${INFERENCE_CPU_QUEUE:-training-normal}" \
  --gmem "${INFERENCE_EVAL_GMEM_MB:-1000}" \
  --memory "$evaluation_memory" \
  --cores "${INFERENCE_CPU_NCORES:-16}" --environment none \
  --dependency "${dependency%&&}" -- \
  "$RUNNER" core evaluate "$CONFIG" "${evaluate_arguments[@]}")"
echo "EVALUATE -> job $job" >&2
