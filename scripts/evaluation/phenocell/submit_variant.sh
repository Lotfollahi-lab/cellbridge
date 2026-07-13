#!/bin/bash
# Reuse an existing UNI cache; project one checkpoint and run its probes.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG="${1:?usage: submit_variant.sh <config> [arm]}"
# Optional arm (uni|distilled). The uni arm does not depend on the checkpoint, so
# when comparing checkpoints only the distilled arm needs re-running.
ARM="${2:-}"

PROJECT_OUTPUT="$($SCRIPT_DIR/project.sh "$CONFIG" 2>&1)"
printf '%s\n' "$PROJECT_OUTPUT"
PROJECT_JOB="$(printf '%s\n' "$PROJECT_OUTPUT" | sed -nE 's/Job <([0-9]+)>.*/\1/p' | tail -n1)"
[ -n "$PROJECT_JOB" ] || { echo "could not parse projection job ID" >&2; exit 1; }

DEPENDENCY="$PROJECT_JOB" "$SCRIPT_DIR/probe.sh" "$CONFIG" "$ARM"
