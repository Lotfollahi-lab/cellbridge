#!/bin/bash
set -euo pipefail
CONFIG="${1:?usage: submit_extract.sh <config> <sample-id>}"
SAMPLE="${2:?usage: submit_extract.sh <config> <sample-id>}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
case "$CONFIG" in /*) ;; *) CONFIG="$ROOT/$CONFIG" ;; esac
[ ! -f "$ROOT/scripts/cluster_env.sh" ] || . "$ROOT/scripts/cluster_env.sh"
DEPENDENCIES=()
[ -z "${DEPENDENCY:-}" ] || DEPENDENCIES=(--dependency "$DEPENDENCY")
"$ROOT/scripts/submit_job.sh" submit \
  --job "novae_${SAMPLE}" --log-prefix "extract_${SAMPLE}" \
  --queue "${NOVAE_QUEUE:-gpu-interactive}" \
  --gpu "${NOVAE_GPU_SPEC:-num=1:gmem=12000}" \
  --memory "${NOVAE_MEM_GB:-48}" --cores "${NOVAE_NCORES:-1}" \
  --environment none "${DEPENDENCIES[@]}" "$CONFIG" -- \
  "$SCRIPT_DIR/run_in_env.sh" python -m \
  cellbridge.exploration.extract_novae_embeddings \
  --config "$CONFIG" --sample-id "$SAMPLE" --accelerator gpu
