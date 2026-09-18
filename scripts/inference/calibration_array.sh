#!/bin/bash
# Submit with an LSF array indexed from 1 to the number of cases.json entries.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKSPACE="$(cd "$ROOT/../.." && pwd)"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
export OPENBLAS_NUM_THREADS="${LSB_DJOB_NUMPROC:-4}"
export OMP_NUM_THREADS="$OPENBLAS_NUM_THREADS"
export MKL_NUM_THREADS="$OPENBLAS_NUM_THREADS"
exec "$WORKSPACE/envs/terra-dev/bin/python" -u -m cellbridge.inference.calibration.run evaluate \
    --output-dir "$1" --index "$((LSB_JOBINDEX - 1))"
