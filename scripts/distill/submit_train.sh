#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
[ ! -f "$ROOT/scripts/cluster_env.sh" ] || . "$ROOT/scripts/cluster_env.sh"
SUBMIT="$ROOT/scripts/submit_job.sh"
CONFIG="${1:-$ROOT/configs/distill/distill_v0_vicreg_mild.yaml}"
case "$CONFIG" in /*) ;; *) CONFIG="$ROOT/$CONFIG" ;; esac

VERSION="$($SUBMIT value "$CONFIG" distill_version)"
case "$VERSION" in
  v0) MODULE="terra_multi.distill.train" ;;
  v1) MODULE="terra_multi.distill.train_prototypes" ;;
  *) echo "unknown distill_version: $VERSION" >&2; exit 1 ;;
esac
RUN_NAME="$($SUBMIT value "$CONFIG" run_name)"
GPU_SPEC="${DISTILL_GPU_SPEC:--}"
"$SUBMIT" submit "$CONFIG" --job "$RUN_NAME" --log-prefix "$RUN_NAME" \
  --queue "${DISTILL_QUEUE:-training-normal}" --gpu "$GPU_SPEC" \
  --memory "${DISTILL_MEM_GB:-64}" --cores "${DISTILL_NCORES:-12}" -- \
  python3 -m "$MODULE" --config "$CONFIG"
