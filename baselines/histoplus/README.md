# HistoPLUS baseline

This adapter evaluates the released, frozen HistoPLUS detector/segmentor/classifier.
It is deliberately different from Cellbridge's representation probes: HistoPLUS
sees complete images and detects its own cells; reference masks are used only for
matching and scoring.

## Protocol

- HEST-test-fast: native WSI inference with the 40x/0.25 MPP checkpoint.
- PhenoCell: reconstruct the 1440x1920 field, explicitly resize from 0.377 to
  0.5 MPP, and use the 20x checkpoint.
- PanNuke: reconstruct the 256x256 field at native 0.25 MPP and use the 40x
  checkpoint. Empty official test fields are retained.
- Overlapping field tiles have a non-overlapping ownership core, so each detected
  centroid is emitted once.
- Evaluation uses maximum-cardinality, minimum-distance one-to-one matching within
  5 microns and an explicit dataset-specific common taxonomy.
- Reports include coverage, detection F1, matched macro-F1,
  coverage-adjusted macro-F1, end-to-end macro-F1, and PanNuke DQ/SQ/PQ.

## Setup and checks

Accept the gated checkpoint terms at `Owkin-Bioptimus/histoplus` and export
`HF_TOKEN`. The runner creates/uses an isolated uv environment at
`$TERRA_MULTI_ENV_ROOT/histoplus` (override with `HISTOPLUS_ENV`).

```bash
scripts/baselines/histoplus/run_in_env.sh check \
  --config configs/baselines/histoplus.yaml
scripts/baselines/histoplus/run_in_env.sh prepare \
  --config configs/baselines/histoplus.yaml
```

Submit the complete resumable pipeline with:

```bash
scripts/baselines/histoplus/submit_parallel.sh
```

Predictions are one atomic `csv.gz` per slide/field under
`outputs/baselines/histoplus/<dataset>/predictions`. Final dataset-level metrics
are written to `<dataset>/scores/aggregate.json` and `per_class.csv`.
