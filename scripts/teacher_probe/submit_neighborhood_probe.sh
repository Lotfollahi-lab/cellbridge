#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BASE="$ROOT/configs/teacher_probe/tenx116_gene_tokens.yaml"
OVERLAY="$ROOT/configs/teacher_probe/tenx116_gene_tokens_neighborhood.yaml"
export TERRA_MULTI_CONFIG_OVERLAY="$OVERLAY"
exec "$SCRIPT_DIR/submit_probe.sh" "$BASE" "${1:-full}"
