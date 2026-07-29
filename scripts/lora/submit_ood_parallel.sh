#!/bin/bash
# Shard the LoRA OOD feature export across GPUs, then probe once all shards land.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG="${1:-$ROOT/configs/lora/contrastive_k0.yaml}"
OOD_CONFIG="${2:-$ROOT/configs/evaluation/pannuke_lora_k0.yaml}"
DATASET="${3:-pannuke}"
case "$CONFIG" in /*) ;; *) CONFIG="$ROOT/$CONFIG" ;; esac
case "$OOD_CONFIG" in /*) ;; *) OOD_CONFIG="$ROOT/$OOD_CONFIG" ;; esac
SUBMIT="$ROOT/scripts/submit_job.sh"
SHARDS="${OOD_SHARDS:-16}"

COMMON=(python -m terra_multi.lora.evaluate_ood --config "$CONFIG"
        --ood-config "$OOD_CONFIG" --dataset "$DATASET" --num-shards "$SHARDS")

IDS=()
for ((shard = 0; shard < SHARDS; shard++)); do
  id="$("$SUBMIT" submit --job "${DATASET}_export_${shard}" \
    --log-prefix "export_${shard}" --queue "${OOD_QUEUE:-training-normal}" \
    --gpu "${OOD_GPU_SPEC:-num=1:gmem=30000}" --memory "${OOD_MEM_GB:-48}" \
    --cores "${OOD_NCORES:-8}" --environment hest "$OOD_CONFIG" -- \
    "${COMMON[@]}" --shard "$shard")"
  echo "  EXPORT shard $shard -> job $id" >&2
  IDS+=("$id")
done

dependency="$(printf 'done(%s)&&' "${IDS[@]}")"
"$SUBMIT" submit --job "${DATASET}_probe" --log-prefix probe \
  --queue "${OOD_PROBE_QUEUE:-inference}" --gpu "-" \
  --memory "${OOD_PROBE_MEM_GB:-64}" --cores "${OOD_PROBE_NCORES:-8}" \
  --environment hest --dependency "${dependency%&&}" "$OOD_CONFIG" -- \
  "${COMMON[@]}" --probe-only
