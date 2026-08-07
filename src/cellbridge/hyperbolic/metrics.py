"""Small geometry diagnostics shared by the existing probe harnesses."""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


def norm_correctness_correlation(norms, y_true, y_pred) -> float:
    """Spearman correlation between embedding radius and probe correctness."""

    norms = np.asarray(norms, dtype=np.float64)
    truth = np.asarray(y_true)
    predicted = np.asarray(y_pred)
    if norms.ndim != 1 or truth.shape != norms.shape or predicted.shape != norms.shape:
        raise ValueError("norms, y_true, and y_pred must be equal-length vectors")
    if len(norms) < 2 or not np.isfinite(norms).all():
        return float("nan")
    correct = (truth == predicted).astype(np.float64)
    if np.ptp(norms) == 0 or np.ptp(correct) == 0:
        return float("nan")
    return float(spearmanr(norms, correct).statistic)
