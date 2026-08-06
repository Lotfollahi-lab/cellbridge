#!/bin/bash
set -euo pipefail
CONFIG="${1:?usage: submit_probe.sh <config> <sample-id>}"
SAMPLE="${2:?usage: submit_probe.sh <config> <sample-id>}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
case "$CONFIG" in /*) ;; *) CONFIG="$ROOT/$CONFIG" ;; esac
SUBMIT="$ROOT/scripts/submit_job.sh"
RUN_NAME="$($SUBMIT value "$CONFIG" run_name)"
DEP_ARGS=(); [ -z "${DEPENDENCY:-}" ] || DEP_ARGS=(--dependency "$DEPENDENCY")
"$SUBMIT" submit "$CONFIG" --job "${RUN_NAME}_probe_${SAMPLE}" \
  --log-prefix "probe_${SAMPLE}" --queue "${JEPA_PROBE_QUEUE:-training-normal}" \
  --gpu "${JEPA_PROBE_GPU_SPEC:--}" --memory "${JEPA_PROBE_MEM_GB:-64}" \
  --cores "${JEPA_PROBE_NCORES:-12}" "${DEP_ARGS[@]}" -- \
  python3 -m cellbridge.jepa.evaluate --config "$CONFIG" --samples "$SAMPLE"
