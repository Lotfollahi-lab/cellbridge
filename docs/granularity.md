# NuCLS annotation granularity

The experiment compares native NuCLS `raw`, `main` and `super` labels using a fixed held-out population. It supports two distinct analyses: aggregating existing fine-label probabilities, and fitting probes directly at the coarser levels. The refit workflow reuses the raw-level predictions as its reference.

`configs/analysis/granularity.yaml` defines the dataset config, raw annotation CSV directory, inference directories, method/representation selection and output directories. All workspace paths use the README environment variables. The experiment has no dependency on the calibration module or retired scaling runs.

## Required inputs

- NuCLS metadata, its official split CSV and raw annotation CSVs under `${CELLBRIDGE_DATA_ROOT}/eval/nucls`.
- Native-label linear and kNN evaluations for each listed method: `predictions.npz` and `metrics.csv` under `<inference_root>/<method>/<version>/nucls/evaluation/<arm>/all/<probe>/`.
- Corresponding `embeddings/<arm>/manifest.csv` and the files it references for refitting.

Generate these with `cellbridge.inference.run extract` and `evaluate` using `configs/inference/nucls.yaml`; this experiment uses the native-label evaluations, not the harmonized conformal-prediction files. The configured CellBridge version is the main 4L TERRA model. Use the baseline versions in the analysis config; LEMON is `vits8`.

## Reproduce

Use the paper's CPU evaluation environment (`terra-dev` locally), from the repository root with `PYTHONPATH=src`:

```bash
python -m cellbridge.inference.granularity.run --config configs/analysis/granularity.yaml
python -m cellbridge.inference.granularity.refit --config configs/analysis/granularity.yaml
python -m cellbridge.inference.granularity.compare \
  --reuse-dir "$CELLBRIDGE_OUTPUTS_ROOT/reproducibility/analysis/nucls_granularity/aggregated" \
  --refit-dir "$CELLBRIDGE_OUTPUTS_ROOT/reproducibility/analysis/nucls_granularity/refit"
```

The first command regenerates `cases.json`, hierarchy, class-support and population tables from the released configuration and underlying evaluations. The second consumes those freshly generated files; no historical dated analysis folder is required. Override the config or use `--output-dir`/`--reuse-dir` for separate runs. `--workers` and `--n-resamples` override the configured defaults (7 and 1,000). Refit supports `--resume` for interrupted runs with unchanged inputs and settings.

Both analyses write per-case predictions/metrics, `summary.csv`, paired slide-bootstrap results, figures and provenance. IDs, labels and class ordering must match across methods; the code checks official splits and slide isolation. kNN aggregation reconstructs integer votes to avoid floating-point tie changes. Native ambiguous labels and rare-class support remain explicit.

The preserved paper outputs remain in their dated analysis folders. Updating these entry points does not imply the full training and probe-fitting experiments were rerun.
