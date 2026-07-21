#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
[ ! -f "$ROOT/scripts/cluster_env.sh" ] || . "$ROOT/scripts/cluster_env.sh"
CONFIG="${1:-$ROOT/configs/exploration/terra_phenotype_structure.yaml}"
"$ROOT/scripts/submit_job.sh" submit "$CONFIG" \
  --job "terra_phenotype_structure" --log-prefix "terra_phenotype_structure" \
  --queue "${PHENOTYPE_QUEUE:-${QUEUE:-oversubscribed}}" \
  --memory "${PHENOTYPE_MEM_GB:-${MEM_GB:-24}}" \
  --cores "${PHENOTYPE_NCORES:-${NCORES:-2}}" -- \
  python3 -m terra_multi.exploration.analyze_terra_phenotypes --config "$CONFIG"
