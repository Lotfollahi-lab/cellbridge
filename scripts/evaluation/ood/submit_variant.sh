#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${1:?usage: submit_variant.sh <config> [arm]}"
ARM="${2:-}"
project="$($SCRIPT_DIR/project.sh "$CONFIG")"
echo "project=$project" >&2
DEPENDENCY="$project" "$SCRIPT_DIR/probe.sh" "$CONFIG" "$ARM"
