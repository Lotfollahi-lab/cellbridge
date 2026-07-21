#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${1:-configs/evaluation/phenocell.yaml}"
embed="$($SCRIPT_DIR/embed.sh "$CONFIG")"
project="$(DEPENDENCY="$embed" "$SCRIPT_DIR/project.sh" "$CONFIG")"
echo "embed=$embed project=$project" >&2
DEPENDENCY="$project" "$SCRIPT_DIR/probe.sh" "$CONFIG"
