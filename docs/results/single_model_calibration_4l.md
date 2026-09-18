# Single-model calibration audit

CellBridge 4L / all / adapted and six baselines; linear probes on the existing conformal label tasks.

Each cell is raw → temperature-scaled. ECE is top-label ECE with 15 equal-width bins, in percentage points; lower is better.
Temperatures minimize held-out validation NLL. No test labels enter temperature fitting. Probe C is reused from the saved train-only selection.
HEST averages 14 section metrics equally; external datasets use cell-weighted fixed-test metrics.

## ECE (percentage points)

| Task | CellBridge | UNI2-h | Virchow2 | VOLTA | LEMON | PAST | SpatialWhisperer |
|---|---:|---:|---:|---:|---:|---:|---:|
| hest all | 1.975 → 1.516 | 2.122 → 1.817 | 2.488 → 1.942 | 3.462 → 2.550 | 2.728 → 2.438 | 3.755 → 2.087 | 7.678 → 2.438 |
| nucls all | 4.260 → 3.543 | 5.522 → 4.306 | 9.767 → 2.841 | 9.553 → 4.615 | 17.347 → 10.121 | 17.739 → 5.037 | 16.965 → 2.079 |
| pannuke all | 1.242 → 0.425 | 1.532 → 0.627 | 2.074 → 0.685 | 1.435 → 0.632 | 1.803 → 0.535 | 3.444 → 0.581 | 6.277 → 0.855 |
| lizard all | 1.186 → 0.551 | 3.110 → 1.536 | 2.167 → 1.102 | 2.321 → 0.868 | 3.235 → 1.155 | 5.122 → 0.499 | 8.670 → 3.551 |
| phenocell all | 4.515 → 2.427 | 6.380 → 3.248 | 5.404 → 1.906 | 4.134 → 1.072 | 3.811 → 2.519 | 7.210 → 1.100 | 13.636 → 2.663 |
| phenocell immune | 3.986 → 2.061 | 5.333 → 1.779 | 5.408 → 1.116 | 1.905 → 3.606 | 2.013 → 4.699 | 8.130 → 2.057 | 14.008 → 2.533 |

## Negative log likelihood

| Task | CellBridge | UNI2-h | Virchow2 | VOLTA | LEMON | PAST | SpatialWhisperer |
|---|---:|---:|---:|---:|---:|---:|---:|
| hest all | 0.801 → 0.799 | 1.206 → 1.204 | 1.247 → 1.244 | 1.313 → 1.306 | 1.175 → 1.171 | 0.978 → 0.967 | 1.288 → 1.241 |
| nucls all | 0.678 → 0.648 | 0.842 → 0.793 | 1.017 → 0.893 | 0.965 → 0.893 | 2.046 → 1.541 | 1.716 → 1.159 | 1.286 → 0.936 |
| pannuke all | 0.450 → 0.448 | 0.577 → 0.576 | 0.581 → 0.579 | 0.864 → 0.862 | 0.720 → 0.718 | 0.600 → 0.590 | 0.626 → 0.590 |
| lizard all | 0.384 → 0.384 | 0.860 → 0.852 | 0.907 → 0.903 | 0.589 → 0.586 | 0.508 → 0.499 | 0.694 → 0.667 | 0.993 → 0.931 |
| phenocell all | 1.423 → 1.413 | 1.635 → 1.625 | 1.658 → 1.645 | 1.843 → 1.839 | 1.718 → 1.709 | 1.679 → 1.651 | 1.776 → 1.688 |
| phenocell immune | 1.056 → 1.052 | 1.158 → 1.156 | 1.164 → 1.161 | 1.292 → 1.296 | 1.214 → 1.217 | 1.238 → 1.211 | 1.301 → 1.233 |

## Brier score (sum over classes)

| Task | CellBridge | UNI2-h | Virchow2 | VOLTA | LEMON | PAST | SpatialWhisperer |
|---|---:|---:|---:|---:|---:|---:|---:|
| hest all | 0.383 → 0.382 | 0.559 → 0.559 | 0.576 → 0.576 | 0.583 → 0.582 | 0.529 → 0.528 | 0.451 → 0.450 | 0.579 → 0.570 |
| nucls all | 0.318 → 0.316 | 0.380 → 0.377 | 0.418 → 0.403 | 0.426 → 0.415 | 0.514 → 0.491 | 0.530 → 0.497 | 0.507 → 0.459 |
| pannuke all | 0.243 → 0.243 | 0.317 → 0.317 | 0.320 → 0.320 | 0.468 → 0.468 | 0.387 → 0.387 | 0.321 → 0.319 | 0.333 → 0.326 |
| lizard all | 0.205 → 0.205 | 0.442 → 0.441 | 0.472 → 0.472 | 0.319 → 0.318 | 0.271 → 0.269 | 0.347 → 0.342 | 0.482 → 0.473 |
| phenocell all | 0.633 → 0.630 | 0.718 → 0.713 | 0.720 → 0.716 | 0.788 → 0.786 | 0.739 → 0.737 | 0.724 → 0.717 | 0.756 → 0.731 |
| phenocell immune | 0.566 → 0.564 | 0.628 → 0.624 | 0.625 → 0.621 | 0.673 → 0.674 | 0.631 → 0.634 | 0.650 → 0.639 | 0.687 → 0.657 |

## Accuracy (%)

| Task | CellBridge | UNI2-h | Virchow2 | VOLTA | LEMON | PAST | SpatialWhisperer |
|---|---:|---:|---:|---:|---:|---:|---:|
| hest all | 72.436 → 72.436 | 57.407 → 57.407 | 55.922 → 55.922 | 55.412 → 55.412 | 60.254 → 60.254 | 67.038 → 67.038 | 56.523 → 56.523 |
| nucls all | 78.114 → 78.114 | 73.898 → 73.898 | 72.326 → 72.326 | 70.883 → 70.883 | 67.933 → 67.933 | 66.763 → 66.763 | 66.843 → 66.843 |
| pannuke all | 83.241 → 83.241 | 77.669 → 77.669 | 77.459 → 77.459 | 65.325 → 65.325 | 72.122 → 72.122 | 77.505 → 77.505 | 76.994 → 76.994 |
| lizard all | 85.302 → 85.302 | 67.674 → 67.674 | 64.728 → 64.728 | 77.150 → 77.150 | 80.999 → 80.999 | 75.010 → 75.010 | 64.844 → 64.844 |
| phenocell all | 50.876 → 50.876 | 40.399 → 40.399 | 40.423 → 40.423 | 33.456 → 33.456 | 39.782 → 39.782 | 40.920 → 40.920 | 38.619 → 38.619 |
| phenocell immune | 56.350 → 56.350 | 47.947 → 47.947 | 49.163 → 49.163 | 44.921 → 44.921 | 49.966 → 49.966 | 48.731 → 48.731 | 44.713 → 44.713 |

## Macro-F1 (%)

| Task | CellBridge | UNI2-h | Virchow2 | VOLTA | LEMON | PAST | SpatialWhisperer |
|---|---:|---:|---:|---:|---:|---:|---:|
| hest all | 52.886 → 52.886 | 30.946 → 30.946 | 29.456 → 29.456 | 29.324 → 29.324 | 36.311 → 36.311 | 44.824 → 44.824 | 32.095 → 32.095 |
| nucls all | 58.408 → 58.408 | 48.894 → 48.894 | 46.790 → 46.790 | 46.708 → 46.708 | 43.786 → 43.786 | 45.192 → 45.192 | 45.834 → 45.834 |
| pannuke all | 82.444 → 82.444 | 75.923 → 75.923 | 75.665 → 75.665 | 62.427 → 62.427 | 69.833 → 69.833 | 75.975 → 75.975 | 75.359 → 75.359 |
| lizard all | 63.706 → 63.706 | 35.294 → 35.294 | 34.886 → 34.886 | 58.021 → 58.021 | 67.384 → 67.384 | 44.948 → 44.948 | 35.494 → 35.494 |
| phenocell all | 42.303 → 42.303 | 29.536 → 29.536 | 29.185 → 29.185 | 19.439 → 19.439 | 27.510 → 27.510 | 27.386 → 27.386 | 28.955 → 28.955 |
| phenocell immune | 46.698 → 46.698 | 37.437 → 37.437 | 38.189 → 38.189 | 27.392 → 27.392 | 36.350 → 36.350 | 32.455 → 32.455 | 34.530 → 34.530 |

## Validation and interpretation

All 133 refitted probes matched the original test IDs, labels and class ordering. Maximum absolute probability difference: 0.00057.
Train/validation/test identities match across all seven methods. Training and temperature-fitting rows are disjoint.
Refitted versus saved argmax differences: 25 cells across all cases. Probabilities are checked within absolute tolerance 0.001.
Cases with logistic-regression convergence warnings: 18. These retain the existing iteration budget; see audit.csv.
Temperature fits at numerical bounds: 0.
`summary.csv` includes 95% percentile intervals from 1,000 grouped bootstrap draws; `paired_differences.csv` contains paired model and scaling differences.
OOD groups follow each existing dataset config; HEST resamples sections. Intervals condition on the fitted probes and temperatures and do not include training or temperature-fitting uncertainty.
All external tasks fit their probe and temperature on that dataset's own designated train/validation splits; this is not source-only calibration under distribution shift.
The inherited conformal protocol selects supported classes using all three splits. This audit preserves that predefined evaluation population; it is not a newly selected test cohort.
ECE bin sensitivity (10, 15, 20 bins) is included in the CSVs. Reliability plots for HEST pool cells for visualization, while headline HEST metrics weight sections equally.
No representation training, ensembling or graph-set calibration is performed.

## Saved artifacts

The complete output, including paired confidence intervals, audit records, saved logits and reliability figures, is in
`outputs/reproducibility/analysis/single_model_calibration_4l` relative to the workspace root.
All 133 cases passed independent metric recomputation from saved logits.
See [the protocol](../single_model_calibration.md) and [summary with confidence intervals](single_model_calibration_4l.csv).
