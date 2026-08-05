#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG="${1:-$ROOT/configs/exploration/novae_phenotype_structure.yaml}"
case "$CONFIG" in /*) ;; *) CONFIG="$ROOT/$CONFIG" ;; esac
[ ! -f "$ROOT/scripts/cluster_env.sh" ] || . "$ROOT/scripts/cluster_env.sh"
DEPENDENCIES=()
[ -z "${DEPENDENCY:-}" ] || DEPENDENCIES=(--dependency "$DEPENDENCY")
"$ROOT/scripts/submit_job.sh" submit \
  --job "novae_phenotype_structure" --log-prefix "analysis" \
  --queue "${NOVAE_ANALYSIS_QUEUE:-oversubscribed}" \
  --memory "${NOVAE_ANALYSIS_MEM_GB:-6}" \
  --cores "${NOVAE_ANALYSIS_NCORES:-1}" \
  "${DEPENDENCIES[@]}" "$CONFIG" -- python3 -m \
  terra_multi.exploration.analyze_novae_phenotypes --config "$CONFIG"
