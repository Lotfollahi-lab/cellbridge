#!/bin/bash
set -euo pipefail
MODEL="${1:?usage: submit_probe.sh <terra|uni|concat>}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
[ ! -f "$ROOT/scripts/cluster_env.sh" ] || . "$ROOT/scripts/cluster_env.sh"
CONFIG="${CONFIG:-$ROOT/configs/exploration/probe.yaml}"
"$ROOT/scripts/submit_job.sh" submit "$CONFIG" --job "probe_${MODEL}" \
  --log-prefix "probe_${MODEL}" --queue "${QUEUE:-oversubscribed}" \
  --memory "${MEM_GB:-4}" --cores "${NCORES:-2}" -- \
  python3 -m terra_multi.exploration.run_linear_probe --config "$CONFIG" --model "$MODEL"
