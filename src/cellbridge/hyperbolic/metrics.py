"""Small geometry diagnostics shared by the existing probe harnesses."""

from __future__ import annotations

import numpy as np
import torch
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


@torch.no_grad()
def ontology_radius_metrics(ontology, K: float) -> dict[str, torch.Tensor]:
    """Summarize ontology radii and expose any cone-domain regression."""

    radius = ontology.ops.radius(ontology.points)
    poincare = ontology.ops.poincare_radius(ontology.points)
    non_root = ontology.depths > 0
    if not bool(non_root.any()):
        raise ValueError("ontology diagnostics require a non-root node")
    selected = poincare[non_root]
    argument = float(K) * (1.0 - selected.square()) / selected.clamp_min(1e-15)
    metrics = {
        "ontology_poincare_radius_min": selected.min(),
        "ontology_saturation_rate": (argument >= 1.0).float().mean(),
    }
    for depth in ontology.depths.unique(sorted=True).tolist():
        if depth == 0:
            continue
        values = radius[ontology.depths == depth]
        metrics[f"ontology_radius_min_d{depth}"] = values.min()
        metrics[f"ontology_radius_median_d{depth}"] = values.median()
    return metrics
