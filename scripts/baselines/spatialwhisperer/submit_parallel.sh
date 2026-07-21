#!/bin/bash
set -euo pipefail

# Submit the complete SpatialWhisperer baseline as an LSF dependency graph:
#
#   prepare -> 5 x HEST GPU extraction --+
#           -> 1 x PhenoCell GPU export --+-> CPU probes
#
# Usage: submit_parallel.sh [config]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
CONFIG="${1:-$ROOT/configs/baselines/spatialwhisperer.yaml}"
case "$CONFIG" in /*) ;; *) CONFIG="$ROOT/$CONFIG" ;; esac
[ -f "$CONFIG" ] || { echo "config not found: $CONFIG" >&2; exit 1; }

CLUSTER_ENV="$ROOT/scripts/cluster_env.sh"
[ ! -f "$CLUSTER_ENV" ] || source "$CLUSTER_ENV"

OUTPUT_DIR="$(sed -nE 's/^output_dir:[[:space:]]*([^#[:space:]]+).*/\1/p' "$CONFIG" | head -n1)"
FAST_PATH="$(sed -nE 's/^[[:space:]]*label_test_fast_path:[[:space:]]*([^#[:space:]]+).*/\1/p' "$CONFIG" | head -n1)"
[ -n "$OUTPUT_DIR" ] && [ -n "$FAST_PATH" ] || {
  echo "config must define output_dir and hest.label_test_fast_path" >&2
  exit 1
}
case "$OUTPUT_DIR" in /*) ;; *) OUTPUT_DIR="$ROOT/$OUTPUT_DIR" ;; esac
[ -f "$FAST_PATH" ] || { echo "fast split not found: $FAST_PATH" >&2; exit 1; }

mapfile -t SAMPLES < <(tail -n +2 "$FAST_PATH" | cut -d, -f1 | grep -v '^$')
[ "${#SAMPLES[@]}" -eq 5 ] || {
  echo "expected 5 label_test_fast samples, found ${#SAMPLES[@]} in $FAST_PATH" >&2
  exit 1
}

LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$LOG_DIR"

BGROUP="${BGROUP:-s10396}"
GPU_QUEUE="${SPATIALWHISPERER_GPU_QUEUE:-training-normal}"
CPU_QUEUE="${SPATIALWHISPERER_CPU_QUEUE:-training-normal}"
# prepare only loads the model (~3.3G of weights) and smoke-tests it, so it does
# not need to sit behind training-normal's backlog. `inference` is an idle GPU
# queue; the oversubscribed CPU hosts are unusable here because they leave only
# ~8G of their 42G free.
PREP_QUEUE="${SPATIALWHISPERER_PREP_QUEUE:-inference}"
GPU_SPEC="${SPATIALWHISPERER_GPU_SPEC:-num=1:gmem=30000}"
GPU_MEM_GB="${SPATIALWHISPERER_GPU_MEM_GB:-96}"
GPU_NCORES="${SPATIALWHISPERER_GPU_NCORES:-12}"
PREP_MEM_GB="${SPATIALWHISPERER_PREP_MEM_GB:-32}"
PREP_NCORES="${SPATIALWHISPERER_PREP_NCORES:-4}"
PROBE_MEM_GB="${SPATIALWHISPERER_PROBE_MEM_GB:-128}"
PROBE_NCORES="${SPATIALWHISPERER_PROBE_NCORES:-16}"
TERRA_ENV="${TERRA_ENV:-${ENV_PATH:-/lustre/scratch126/cellgen/lotfollahi/so25/envs/terra-dev}}"

submit_pixi() {
  # submit_pixi <suffix> <queue> <mem-gb> <cores> <dependency> <gpu: 0|1> <cli args...>
  local suffix="$1" queue="$2" mem="$3" cores="$4" dependency="$5" gpu="$6"
  shift 6
  local dependency_directive="" gpu_directive="" run_command=""
  [ -z "$dependency" ] || dependency_directive="#BSUB -w \"$dependency\""
  if [ "$gpu" = 0 ]; then
    # `-gpu "-"` still reserves the default GPU (num=1), but the GPU queues'
    # esub rejects jobs that omit -gpu entirely, so it stays. Keep CPU-ish
    # stages on an uncongested queue instead (see PREP_QUEUE).
    gpu_directive='#BSUB -gpu "-"'
  else
    gpu_directive="#BSUB -gpu \"$GPU_SPEC\""
  fi
  printf -v run_command '%q ' "$SCRIPT_DIR/run_in_env.sh" "$@"
  run_command="${run_command% }"

  bsub <<EOF | sed -nE 's/.*Job <([0-9]+)>.*/\1/p'
#BSUB -J sw_${suffix}
#BSUB -o ${LOG_DIR}/${suffix}_%J.out
#BSUB -e ${LOG_DIR}/${suffix}_%J.err
#BSUB -n ${cores}
#BSUB -q ${queue}
${gpu_directive}
#BSUB -G ${BGROUP}
#BSUB -M ${mem}G
#BSUB -R "select[mem>${mem}G] rusage[mem=${mem}G] span[hosts=1]"
${dependency_directive}

set -euo pipefail
export OMP_NUM_THREADS=${cores} OPENBLAS_NUM_THREADS=${cores} MKL_NUM_THREADS=${cores}
${run_command}
EOF
}

prep_id="$(submit_pixi prepare "$PREP_QUEUE" "$PREP_MEM_GB" "$PREP_NCORES" "" 0 \
  prepare --config "$CONFIG")"
[ -n "$prep_id" ] || { echo "failed to submit prepare job" >&2; exit 1; }
echo "  PREPARE       -> job $prep_id"

export_ids=()
for sample in "${SAMPLES[@]}"; do
  id="$(submit_pixi "hest_${sample}" "$GPU_QUEUE" "$GPU_MEM_GB" "$GPU_NCORES" \
    "done($prep_id)" 1 extract-hest --config "$CONFIG" --device cuda --samples "$sample")"
  [ -n "$id" ] || { echo "failed to submit HEST extraction for $sample" >&2; exit 1; }
  echo "  HEST  $sample -> job $id"
  export_ids+=("$id")
done

phenocell_id="$(submit_pixi phenocell "$GPU_QUEUE" "$GPU_MEM_GB" "$GPU_NCORES" \
  "done($prep_id)" 1 extract-phenocell --config "$CONFIG" --device cuda)"
[ -n "$phenocell_id" ] || { echo "failed to submit PhenoCell extraction" >&2; exit 1; }
echo "  PHENOCELL     -> job $phenocell_id"
export_ids+=("$phenocell_id")

dependency="$(printf 'done(%s)&&' "${export_ids[@]}")"
dependency="${dependency%&&}"

probe_id="$(bsub <<EOF | sed -nE 's/.*Job <([0-9]+)>.*/\1/p'
#BSUB -J sw_probe
#BSUB -o ${LOG_DIR}/probe_%J.out
#BSUB -e ${LOG_DIR}/probe_%J.err
#BSUB -n ${PROBE_NCORES}
#BSUB -q ${CPU_QUEUE}
#BSUB -gpu "-"
#BSUB -G ${BGROUP}
#BSUB -M ${PROBE_MEM_GB}G
#BSUB -R "select[mem>${PROBE_MEM_GB}G] rusage[mem=${PROBE_MEM_GB}G] span[hosts=1]"
#BSUB -w "${dependency}"

set -euo pipefail
module load cellgen/uv
source "${TERRA_ENV}/bin/activate"
cd "${ROOT}"
export OMP_NUM_THREADS=${PROBE_NCORES} OPENBLAS_NUM_THREADS=${PROBE_NCORES} MKL_NUM_THREADS=${PROBE_NCORES}
PYTHONPATH=src python -m terra_multi.baselines.spatialwhisperer probe --config "${CONFIG}" --dataset all
EOF
)"
[ -n "$probe_id" ] || { echo "failed to submit probe job" >&2; exit 1; }
echo "  PROBE         -> job $probe_id (waits on ${#export_ids[@]} extraction jobs)"
