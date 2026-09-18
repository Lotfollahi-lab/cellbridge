"""Temperature scaling and grouped classification-calibration statistics."""

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import log_softmax

METRICS = ("ece_15", "ece_10", "ece_20", "nll", "brier", "accuracy", "macro_f1")


def log_probabilities(logits, temperature=1.0):
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 2 or logits.shape[1] < 2 or not np.isfinite(logits).all():
        raise ValueError("finite [cells, classes] logits are required")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    return log_softmax(logits / temperature, axis=1)


def fit_temperature(logits, targets):
    """Fit one scalar on held-out labels by minimizing unweighted cell NLL."""
    targets = np.asarray(targets, dtype=int)
    if targets.shape != (len(logits),) or not len(targets):
        raise ValueError("nonempty, aligned temperature-fitting labels are required")
    if targets.min() < 0 or targets.max() >= np.shape(logits)[1]:
        raise ValueError("temperature-fitting labels are outside the class range")

    def objective(log_temperature):
        values = log_probabilities(logits, np.exp(log_temperature))
        return float(-values[np.arange(len(targets)), targets].mean())

    result = minimize_scalar(objective, bounds=(-6, 6), method="bounded")
    if not result.success:
        raise RuntimeError("temperature optimization failed")
    log_temperature = float(result.x) if result.fun <= objective(0) else 0.0
    return {
        "temperature": float(np.exp(log_temperature)),
        "temperature_at_bound": abs(log_temperature) > 5.99,
        "validation_nll_raw": objective(0),
        "validation_nll_scaled": objective(log_temperature),
    }


def statistics(logits, targets, groups, temperature=1.0, n_resamples=1000, seed=0):
    """Compute point metrics, reliability bins and paired group-bootstrap draws.

    ECE is top-label, equal-width; Brier sums over all classes (no /K).
    Bootstrap sufficient statistics avoid repeating individual cell arrays.
    """
    logs = log_probabilities(logits, temperature)
    probabilities = np.exp(logs)
    targets = np.asarray(targets, dtype=int)
    groups = np.asarray(groups).astype(str)
    n, k = probabilities.shape
    if targets.shape != (n,) or groups.shape != (n,) or not n:
        raise ValueError("nonempty targets and groups must align with logits")
    if targets.min() < 0 or targets.max() >= k:
        raise ValueError("targets are outside the class range")
    unique, indices = np.unique(groups, return_inverse=True)
    g = len(unique)
    counts = np.bincount(indices, minlength=g)
    predicted = probabilities.argmax(axis=1)
    correct = predicted == targets
    confidence = probabilities.max(axis=1)
    losses = -logs[np.arange(n), targets]
    brier = (probabilities ** 2).sum(axis=1) - 2 * probabilities[np.arange(n), targets] + 1
    weights = np.ones((1, g))
    if g > 1 and n_resamples:
        draws = np.random.default_rng(seed).multinomial(g, np.full(g, 1 / g), size=n_resamples)
        weights = np.concatenate((weights, draws), axis=0)
    totals = weights @ counts
    values = {}
    for metric, cell_values in (("nll", losses), ("brier", brier), ("accuracy", correct)):
        sums = np.bincount(indices, weights=cell_values, minlength=g)
        values[metric] = weights @ sums / totals

    confusions = np.bincount(indices * k * k + targets * k + predicted, minlength=g * k * k).reshape(g, k, k)
    sampled = np.einsum("bg,gij->bij", weights, confusions)
    denominator = sampled.sum(axis=1) + sampled.sum(axis=2)
    f1 = np.divide(2 * np.diagonal(sampled, axis1=1, axis2=2), denominator,
                   out=np.zeros_like(denominator), where=denominator > 0)
    values["macro_f1"] = f1.mean(axis=1)

    reliability = []
    for bins in (10, 15, 20):
        bin_index = np.minimum((confidence * bins).astype(int), bins - 1)
        flat = indices * bins + bin_index
        bin_counts = np.bincount(flat, minlength=g * bins).reshape(g, bins)
        bin_correct = np.bincount(flat, weights=correct, minlength=g * bins).reshape(g, bins)
        bin_confidence = np.bincount(flat, weights=confidence, minlength=g * bins).reshape(g, bins)
        values[f"ece_{bins}"] = np.abs(weights @ (bin_correct - bin_confidence)).sum(axis=1) / totals
        for b in range(bins):
            count = int(bin_counts[:, b].sum())
            reliability.append({"n_bins": bins, "bin": b, "lower": b / bins, "upper": (b + 1) / bins,
                                "n": count, "accuracy": bin_correct[:, b].sum() / count if count else np.nan,
                                "confidence": bin_confidence[:, b].sum() / count if count else np.nan})
    point = {name: float(values[name][0]) for name in METRICS}
    replicates = pd.DataFrame({name: values[name][1:] for name in METRICS})
    return point, replicates, pd.DataFrame(reliability)
