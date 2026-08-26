#!/bin/bash
# Submit extraction and dependent evaluation for one method/version/dataset.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
WORKSPACE="$(cd "$ROOT/../.." && pwd)"
SUBMIT="$ROOT/scripts/submit_job.sh"
RUNNER="$SCRIPT_DIR/run_in_env.sh"

if [ "$#" -lt 3 ]; then
  echo "usage: $(basename "$0") <config> <method> <version> [options]" >&2
  exit 2
fi

config="$1"
method="$2"
version="$3"
shift 3
checkpoint=""
conformal_config=""

while [ "$#" -gt 0 ]; do
  case "$1" in
    --checkpoint)
      checkpoint="${2:?--checkpoint requires a path}"
      shift 2
      ;;
    --conformal-config)
      conformal_config="${2:?--conformal-config requires a path}"
      shift 2
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

case "$config" in /*) ;; *) config="$ROOT/$config" ;; esac
if [ -n "$conformal_config" ]; then
  case "$conformal_config" in
    /*) ;;
    *) conformal_config="$ROOT/$conformal_config" ;;
  esac
fi

if [ "$method" = cellbridge ] && [ -z "$checkpoint" ]; then
  echo "cellbridge requires --checkpoint" >&2
  exit 2
fi

dataset="$($SUBMIT value "$config" dataset)"
outputs_root="${CELLBRIDGE_OUTPUTS_ROOT:-$WORKSPACE/outputs}"
output_dir="$outputs_root/reproducibility/inference/$method/$version/$dataset"
log_dir="$output_dir/logs"

arguments=(--output-dir "$output_dir")
if [ -n "$checkpoint" ]; then
  arguments+=(--checkpoint "$checkpoint")
fi

check_config="${conformal_config:-$config}"
"$RUNNER" core check "$check_config" --method "$method" "${arguments[@]}"

case "$method" in
  uni) gmem=16000 ;;
  cellbridge) gmem=24000 ;;
  past) gmem=16000 ;;
  volta) gmem=4000 ;;
  spatialwhisperer) gmem=30000 ;;
  *)
    echo "unknown inference method: $method" >&2
    exit 2
    ;;
esac

job_name="${dataset}_${method}_${version}"
extract_job="$($SUBMIT submit "$config" \
  --job "${job_name}_extract" --log-prefix extract --log-dir "$log_dir" \
  --queue "${INFERENCE_GPU_QUEUE:-training-normal}" \
  --gmem "$gmem" --memory "${INFERENCE_GPU_MEM_GB:-64}" \
  --cores "${INFERENCE_GPU_NCORES:-8}" --environment none -- \
  "$RUNNER" "$method" extract "$config" --device cuda "${arguments[@]}")"

case "$dataset" in
  hest) evaluation_memory="${INFERENCE_CPU_MEM_GB:-256}" ;;
  *) evaluation_memory="${INFERENCE_CPU_MEM_GB:-64}" ;;
esac

evaluate_job="$($SUBMIT submit "$config" \
  --job "${job_name}_evaluate" --log-prefix evaluate --log-dir "$log_dir" \
  --queue "${INFERENCE_CPU_QUEUE:-training-normal}" \
  --memory "$evaluation_memory" --cores "${INFERENCE_CPU_NCORES:-16}" \
  --environment none --dependency "done($extract_job)" -- \
  "$RUNNER" core evaluate "$config" --method "$method" \
  --output-dir "$output_dir")"

conformal_job="-"
if [ -n "$conformal_config" ]; then
  conformal_job="$($SUBMIT submit "$conformal_config" \
    --job "${job_name}_conformal" --log-prefix conformal --log-dir "$log_dir" \
    --queue "${INFERENCE_CPU_QUEUE:-training-normal}" \
    --memory "$evaluation_memory" --cores "${INFERENCE_CPU_NCORES:-16}" \
    --environment none --dependency "done($extract_job)" -- \
    "$RUNNER" core conformal "$conformal_config" --method "$method" \
    --output-dir "$output_dir")"
fi

echo "$dataset $method $version extract=$extract_job evaluate=$evaluate_job conformal=$conformal_job"
