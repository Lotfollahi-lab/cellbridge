#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
[ ! -f "$ROOT/scripts/cluster_env.sh" ] || . "$ROOT/scripts/cluster_env.sh"
CONFIG="${1:-$ROOT/configs/hierarchical/hest_label_test.yaml}"
"$ROOT/scripts/submit_job.sh" submit \
  --job hest_hierarchical_labels --log-prefix hest_hierarchical_labels \
  --queue "${HIERARCHY_QUEUE:-${QUEUE:-oversubscribed}}" \
  --memory "${HIERARCHY_MEM_GB:-${MEM_GB:-32}}" \
  --cores "${HIERARCHY_NCORES:-${NCORES:-2}}" "$CONFIG" -- \
  python3 -m cellbridge.hierarchical.build_labels --config "$CONFIG"
