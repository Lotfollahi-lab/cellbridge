"""Shared scoring and paired slide bootstrap for both granularity protocols."""

import subprocess

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

from cellbridge.config import config_root
from cellbridge.evaluation import save_result
from cellbridge.inference.evaluate.bootstrap import (
    _group_confusions,
    _metrics_from_confusions,
)

DEFAULT_CONFIG = config_root() / "configs/analysis/granularity.yaml"
LEVELS = ("super", "main", "raw")
METRICS = ("macro_f1", "balanced_accuracy", "accuracy", "weighted_f1")


def source_revision():
    """Return checkout provenance when Git is available; wheels have no revision."""
    root = config_root()
    if not (root / ".git").exists():
        return None
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def bootstrap_weights(groups, n_resamples):
    """Same sorted slide order and seed pair every method, probe and level."""
    count = len(np.unique(groups))
    return np.random.default_rng(0).multinomial(
        count,
        np.full(count, 1 / count),
        size=n_resamples,
    )


def score_result(result, groups, weights, case, level, destination):
    save_result(destination, result)
    pd.DataFrame(
        classification_report(
            result.y_true, result.y_pred, output_dict=True, zero_division=0
        )
    ).T.to_csv(destination / "per_class.csv")
    confusions, true_mask, _ = _group_confusions(result, groups)
    values = _metrics_from_confusions(
        np.einsum("bg,gij->bij", weights, confusions), true_mask
    )
    rows = []
    for metric in METRICS:
        low, high = np.percentile(values[metric], [2.5, 97.5])
        rows.append(
            dict(
                method=case["method"],
                probe=case["probe"],
                level=level,
                n_classes=len(result.classes),
                n_test=len(groups),
                metric=metric,
                estimate=result.metrics[metric],
                bootstrap_se=values[metric].std(ddof=1),
                ci_low=low,
                ci_high=high,
            )
        )
    draws = pd.DataFrame(values).assign(
        method=case["method"],
        probe=case["probe"],
        level=level,
        bootstrap=np.arange(len(weights)),
    )
    return pd.DataFrame(rows), draws
