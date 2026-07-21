#!/bin/bash
set -euo pipefail
CONFIG="${1:?usage: submit_probe.sh <config> <arm|aggregate>}"
TARGET="${2:?usage: submit_probe.sh <config> <arm|aggregate>}"
case "$TARGET" in
  uni_baseline|terra_teacher|distilled|concat_oracle|aggregate|prototype_penultimate_*|prototype_probs_*) ;;
  *) echo "unknown target: $TARGET" >&2; exit 1 ;;
esac
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
[ ! -f "$ROOT/scripts/cluster_env.sh" ] || . "$ROOT/scripts/cluster_env.sh"
case "$CONFIG" in /*) ;; *) CONFIG="$ROOT/$CONFIG" ;; esac
SUBMIT="$ROOT/scripts/submit_job.sh"
RUN_NAME="$($SUBMIT value "$CONFIG" run_name)"
MEMORY="${DISTILL_EVAL_MEM_GB:-32}"
PY_ARGS=(--arm "$TARGET")
if [ "$TARGET" = aggregate ]; then
  MEMORY="${DISTILL_AGG_MEM_GB:-8}"
  PY_ARGS=(--aggregate-only)
fi
DEP_ARGS=()
[ -z "${DEPENDENCY:-}" ] || DEP_ARGS=(--dependency "$DEPENDENCY")
"$SUBMIT" submit "$CONFIG" --job "${RUN_NAME}_probe_${TARGET}" \
  --log-prefix "${RUN_NAME}_probe_${TARGET}" \
  --queue "${DISTILL_EVAL_QUEUE:-training-normal}" \
  --gpu "${DISTILL_EVAL_GPU_SPEC:--}" --memory "$MEMORY" \
  --cores "${DISTILL_EVAL_NCORES:-4}" "${DEP_ARGS[@]}" -- \
  python3 -m terra_multi.distill.evaluate_probes --config "$CONFIG" "${PY_ARGS[@]}"
