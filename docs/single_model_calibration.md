# Single-model probability calibration

This audit compares CellBridge **four-layer / all cells / adapted** with UNI2-h,
Virchow2, VOLTA, LEMON (vits8), PAST and SpatialWhisperer. It uses the linear
probes, harmonized labels, supported classes and exact test populations from
the conformal comparison. HEST excludes TENX143 and averages 14 sections;
NuCLS, PanNuke, Lizard and both PhenoCell tasks use their fixed external splits.
These external tasks use their own training and validation labels: this is not
source-only calibration under distribution shift.

## Protocol

1. Load existing embeddings. Fit PCA only on the training rows where the existing
   evaluation protocol requires it.
2. Reuse the saved conformal run's regularization `C`, originally selected using
   training-only cross-validation. Refit the standardized logistic probe on
   **train only**, recording convergence warnings without changing the original
   iteration budget. Never use the
   ordinary evaluator's train-plus-validation refit for this audit.
3. Require identical saved test IDs, true labels and class order. Require the
   refitted probabilities to agree with the saved probabilities within absolute
   tolerance 0.001 (0.1 percentage point); record maximum and mean differences and changed argmax counts.
   Small numerical differences can arise from PCA/BLAS implementations or threads.
   Cases outside tolerance retry with the original 16-thread setting and must then
   pass the same check.
4. Fit a single positive temperature on held-out validation logits by minimizing
   cell-weighted negative log likelihood. Optimize log-temperature in [-6, 6]
   and flag solutions at the numerical bounds. Keep temperature 1 if the optimizer
   fails to improve validation NLL. An optimization failure stops the case.
5. Evaluate raw and scaled probabilities on untouched test cells. Save validation
   and test logits, labels, IDs, classes and test bootstrap groups so temperature
   fitting and metrics can be independently reproduced.

The inherited class-support filter uses support counts in all three splits;
we retain the existing predefined evaluation population rather than select a
new cohort. Temperature fitting does not use test labels or optimize test ECE.
Graph closure and conformal prediction sets do not enter these probability metrics.

## Metrics and uncertainty

- Primary: top-label ECE, 15 fixed equal-width bins, also reported with 10 and 20
  bins. Bins are left-closed/right-open except that confidence 1 is in the last bin.
- NLL uses natural logarithms and stable log-softmax.
- Multiclass Brier score sums squared error over all classes, without dividing by K.
- Accuracy and macro-F1 use all retained task classes. Positive scalar temperature
  scaling preserves the predicted class, so these should not change.
- All CSV metric values are fractions, except NLL/Brier. The Markdown table reports
  ECE in percentage points and accuracy/F1 in percent.
- 1,000 paired bootstrap draws resample the existing configured test groups:
  NuCLS slides, PanNuke and Lizard sample IDs, and PhenoCell region IDs.
  HEST resamples its 14 sections and preserves equal section weighting.
- Confidence intervals condition on fitted models and temperatures. They do not
  include variation from retraining or estimating temperature on a new validation set.
- Cross-method comparisons require identical train/validation IDs and identical
  test IDs, labels, groups and class ordering. Shared sorted groups and random seeds
  ensure paired resampling. Differences are CellBridge minus the baseline, or
  temperature-scaled minus raw; negative is favorable for ECE/NLL/Brier.
- HEST reliability plots pool cells for visualization; the headline metrics remain
  equal-section averages.

## Running

From the repository root, activate the terra-dev environment and set `PYTHONPATH`:

```bash
module load cellgen/uv
source ../../envs/terra-dev/bin/activate
export PYTHONPATH="$PWD/src"
AUDIT_OUTPUT="../../outputs/reproducibility/analysis/single_model_calibration_4l"
python -m cellbridge.inference.calibration.run prepare --output-dir "$AUDIT_OUTPUT"
python -m cellbridge.inference.calibration.run evaluate-all --output-dir "$AUDIT_OUTPUT" --workers 4
python -m cellbridge.inference.calibration.run report --output-dir "$AUDIT_OUTPUT"
```

`prepare` freezes 133 cases and expanded dataset configurations and refuses to
replace an existing manifest. `evaluate-all` uses at most four workers, initially two
BLAS threads each, and skips completed cases. A failed reproduction check retries
that worker with 16 threads. These local retries are serialized to avoid multiple
16-thread fits competing for the same CPU allocation. Use a new output directory when
changing the experiment. Individual cases can run with `evaluate --index N`
(zero-based). `scripts/inference/calibration_array.sh` supports one-based LSF
arrays; request CPU resources appropriate for the embedding sizes (e.g. 16 cores,
32 GB per case). No GPU or representation retraining is required.

The report refuses incomplete or misaligned comparisons. It writes:

- `results.md`: raw → calibrated comparisons for all five headline metrics.
- `summary.csv`: point estimates, bootstrap SEs and 95% percentile intervals.
- `paired_differences.csv`: paired CellBridge–baseline and scaling differences.
- `per_section_metrics.csv`, `bootstrap_metrics.csv`, `audit.csv`.
- `figures/`: reliability plots as PDF and PNG.
- `runs/`: per-case metrics, bins, bootstrap draws, saved logits and audit metadata.

Run validation with `PYTHONPATH=src python -m pytest tests/test_calibration -q`.

`provenance.json` records package versions, the source revision/branch and source
hashes. `audit.csv` flags iteration-limit warnings inherited from the existing
probe settings and records any argmax changes relative to saved predictions.

## Completed results

All 133 cases completed and passed independent metric checks. See the
[results table](results/single_model_calibration_4l.md) and
[summary with confidence intervals](results/single_model_calibration_4l.csv).
CellBridge has the lowest raw ECE point estimate on four of six tasks and the
lowest temperature-scaled ECE on two; these ranks alone do not establish
statistical significance. Eighteen VOLTA fits retained convergence warnings
under the original iteration budget. This audit evaluates single models; it
does not measure deep ensembles.
