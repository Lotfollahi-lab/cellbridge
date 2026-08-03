#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
[ ! -f "$ROOT/scripts/cluster_env.sh" ] || . "$ROOT/scripts/cluster_env.sh"
CONFIG="${1:-$ROOT/configs/teacher_probe/tenx116_gene_tokens.yaml}"
MODE="${2:-full}"
EXTRA=()
[[ "$MODE" == "full" ]] || EXTRA+=(--smoke)

"$ROOT/scripts/submit_job.sh" submit \
  --job "teacher_tokens_${MODE}" --log-prefix "teacher_tokens_${MODE}" \
  --output-key output_dir --log-subdir logs \
  --queue "${TERRA_QUEUE:-${GPU_QUEUE:-auto}}" \
  --gpu "${TERRA_GPU_SPEC:-${GPU_SPEC:-num=1:gmem=12000}}" \
  --memory "${TERRA_TOKEN_MEM_GB:-16}" --cores "${TERRA_TOKEN_CORES:-8}" \
  "$CONFIG" -- python3 -m terra_multi.teacher_probe.extract_gene_tokens \
  --config "$CONFIG" "${EXTRA[@]}"

