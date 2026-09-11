"""Grouped bootstrap uncertainty for fixed-split OOD conformal evaluation."""

import numpy as np
import pandas as pd

from cellbridge.inference.conformal.conformal_ood import ConformalResult
from cellbridge.inference.evaluate.bootstrap import _parameters

METRICS = (
    "marginal_coverage",
    "coverage_gap",
    "mean_prediction_set_size",
    "singleton_rate",
)


def prediction_sets(result: ConformalResult) -> dict[str, np.ndarray]:
    """Prediction sets of every construction, in metrics.csv order."""
    return {
        "split": result.split_sets,
        "graph": result.graph_sets,
        **result.conditional_sets,
    }


def _metrics(weights, totals, target_coverage):
    """Conformal metrics for each row of group weights."""
    cells = weights @ totals["cells"]
    class_cells = weights @ totals["class_cells"]
    class_covered = weights @ totals["class_covered"]
    present = class_cells > 0
    class_gaps = np.abs(
        np.divide(
            class_covered,
            class_cells,
            out=np.zeros_like(class_covered),
            where=present,
        )
        - target_coverage
    )
    return {
        "marginal_coverage": weights @ totals["covered"] / cells,
        "coverage_gap": (
            100.0 * np.where(present, class_gaps, 0.0).sum(axis=1) / present.sum(axis=1)
        ),
        "mean_prediction_set_size": weights @ totals["sizes"] / cells,
        "singleton_rate": weights @ totals["singletons"] / cells,
    }


def summarize(sets_by_method, labels, classes, groups, settings, *, target_coverage):
    """Compute paired group-bootstrap replicates of the conformal metrics.

    Group weights are drawn exactly as for the classification metrics, so
    replicate ``b`` resamples the same test groups for every construction and
    for macro F1. Estimates match ``conformal_ood._summary``.
    """
    n_resamples, seed = _parameters(settings)
    labels = np.asarray(labels).astype(str)
    classes = np.asarray(classes).astype(str)
    groups = np.asarray(groups).astype(str)
    if len(groups) != len(labels):
        raise ValueError("bootstrap groups and predictions must be row-aligned")

    lookup = {label: column for column, label in enumerate(classes)}
    try:
        columns = np.asarray([lookup[label] for label in labels], dtype=int)
    except KeyError as error:
        raise ValueError(
            f"test label is absent from classifier classes: {error.args[0]!r}"
        ) from error
    unique_groups, group_index = np.unique(groups, return_inverse=True)
    n_cells, n_groups, n_classes = len(labels), len(unique_groups), len(classes)
    if n_groups < 2:
        raise ValueError("grouped bootstrap requires at least two test groups")

    weights = np.random.default_rng(seed).multinomial(
        n_groups,
        np.full(n_groups, 1.0 / n_groups),
        size=n_resamples,
    )
    group_class = group_index * n_classes + columns
    class_cells = np.bincount(group_class, minlength=n_groups * n_classes).reshape(
        n_groups, n_classes
    )
    unit = np.ones((1, n_groups), dtype=np.int64)

    rows, replicates = [], []
    for method, sets in sets_by_method.items():
        sets = np.asarray(sets, dtype=bool)
        if sets.shape != (n_cells, n_classes):
            raise ValueError(f"{method} sets must have shape {(n_cells, n_classes)}")
        covered = sets[np.arange(n_cells), columns].astype(np.float64)
        sizes = sets.sum(axis=1).astype(np.float64)
        totals = {
            "cells": class_cells.sum(axis=1),
            "covered": np.bincount(group_index, weights=covered, minlength=n_groups),
            "sizes": np.bincount(group_index, weights=sizes, minlength=n_groups),
            "singletons": np.bincount(
                group_index, weights=(sizes == 1).astype(np.float64), minlength=n_groups
            ),
            "class_cells": class_cells,
            "class_covered": np.bincount(
                group_class, weights=covered, minlength=n_groups * n_classes
            ).reshape(n_groups, n_classes),
        }
        estimates = _metrics(unit, totals, target_coverage)
        values = _metrics(weights, totals, target_coverage)
        replicates.append(
            pd.DataFrame(
                {
                    "bootstrap": np.arange(n_resamples, dtype=np.int64),
                    "conformal_method": method,
                    **values,
                }
            )
        )
        for metric in METRICS:
            finite = values[metric][np.isfinite(values[metric])]
            if len(finite) < 2:
                raise ValueError(
                    f"fewer than two finite bootstrap replicates for {method} {metric}"
                )
            rows.append(
                {
                    "conformal_method": method,
                    "metric": metric,
                    "estimate": float(estimates[metric][0]),
                    "bootstrap_se": float(finite.std(ddof=1)),
                    "ci_2_5": float(np.percentile(finite, 2.5)),
                    "ci_97_5": float(np.percentile(finite, 97.5)),
                    "n_bootstrap_finite": len(finite),
                    "n_test_groups": n_groups,
                    "n_bootstrap": n_resamples,
                    "seed": seed,
                    "group_column": str(settings.get("group_column", "")),
                    "group_regex": str(settings.get("group_regex", "")),
                }
            )
    return pd.DataFrame(rows), pd.concat(replicates, ignore_index=True)
