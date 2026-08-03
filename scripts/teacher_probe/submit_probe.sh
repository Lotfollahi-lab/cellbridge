#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
[ ! -f "$ROOT/scripts/cluster_env.sh" ] || . "$ROOT/scripts/cluster_env.sh"
CONFIG="${1:-$ROOT/configs/teacher_probe/tenx116_gene_tokens.yaml}"
MODE="${2:-full}"
EXTRA=()
[[ "$MODE" == "full" ]] || EXTRA+=(--smoke)
DEPENDENCY="${DEPENDENCY:-}"
DEP_ARGS=()
[[ -z "$DEPENDENCY" ]] || DEP_ARGS=(--dependency "$DEPENDENCY")

"$ROOT/scripts/submit_job.sh" submit \
  --job "teacher_probe_${MODE}" --log-prefix "teacher_probe_${MODE}" \
  --output-key output_dir --log-subdir logs \
  --queue "${QUEUE:-auto}" --memory "${TEACHER_PROBE_MEM_GB:-6}" \
  --cores "${TEACHER_PROBE_CORES:-4}" \
  "${DEP_ARGS[@]}" "$CONFIG" -- python3 -m terra_multi.teacher_probe.run --config "$CONFIG" "${EXTRA[@]}"

