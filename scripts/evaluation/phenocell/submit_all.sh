#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${1:-configs/evaluation/phenocell.yaml}"

EMBED_OUTPUT="$($SCRIPT_DIR/embed.sh "$CONFIG" 2>&1)"
printf '%s\n' "$EMBED_OUTPUT"
EMBED_JOB="$(printf '%s\n' "$EMBED_OUTPUT" | sed -nE 's/Job <([0-9]+)>.*/\1/p' | tail -n1)"
[ -n "$EMBED_JOB" ] || { echo "could not parse embedding job ID" >&2; exit 1; }

PROJECT_OUTPUT="$(DEPENDENCY="$EMBED_JOB" "$SCRIPT_DIR/project.sh" "$CONFIG" 2>&1)"
printf '%s\n' "$PROJECT_OUTPUT"
PROJECT_JOB="$(printf '%s\n' "$PROJECT_OUTPUT" | sed -nE 's/Job <([0-9]+)>.*/\1/p' | tail -n1)"
[ -n "$PROJECT_JOB" ] || { echo "could not parse projection job ID" >&2; exit 1; }

DEPENDENCY="$PROJECT_JOB" "$SCRIPT_DIR/probe.sh" "$CONFIG"
