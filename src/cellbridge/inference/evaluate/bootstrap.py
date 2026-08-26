"""Grouped bootstrap uncertainty for fixed-split OOD evaluation."""

from pathlib import Path

import numpy as np
import pandas as pd

from cellbridge.evaluation import ClassificationResult

METRICS = (
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "weighted_f1",
)


def _parameters(settings):
    if not isinstance(settings, dict):
        raise ValueError("probe.bootstrap must be a mapping")
    try:
        n_resamples = int(settings.get("n_resamples", 1000))
        seed = int(settings.get("seed", 0))
    except (TypeError, ValueError) as error:
        raise ValueError("bootstrap n_resamples and seed must be integers") from error
    if n_resamples < 2:
        raise ValueError("probe.bootstrap.n_resamples must be at least two")
    return n_resamples, seed


def resolve_groups(metadata, ids, settings) -> np.ndarray:
    """Return bootstrap groups aligned to evaluated row IDs."""
    _parameters(settings)
    column = settings.get("group_column")
    if not column:
        raise ValueError("probe.bootstrap.group_column is required")
    if "row_id" not in metadata or column not in metadata:
        missing = {"row_id", str(column)} - set(metadata)
        raise ValueError(f"bootstrap metadata is missing {sorted(missing)}")
    if metadata["row_id"].duplicated().any():
        raise ValueError("bootstrap metadata contains duplicate row IDs")

    ids = np.asarray(ids)
    positions = pd.Index(metadata["row_id"]).get_indexer(ids)
    if np.any(positions < 0):
        missing = ids[positions < 0]
        raise ValueError(f"bootstrap row IDs are missing: {missing[:5].tolist()}")

    groups = metadata.iloc[positions][column].astype("string")
    pattern = settings.get("group_regex")
    if pattern:
        groups = groups.str.extract(str(pattern), expand=False)
        if isinstance(groups, pd.DataFrame):
            raise ValueError("bootstrap group_regex must have one capture group")
    if groups.isna().any():
        raise ValueError("bootstrap groups contain null or unmatched values")

    groups = groups.str.strip().to_numpy(dtype=str)
    if np.any(groups == ""):
        raise ValueError("bootstrap groups contain empty values")
    if len(np.unique(groups)) < 2:
        raise ValueError("grouped bootstrap requires at least two test groups")
    return groups


def _group_confusions(result, groups):
    y_true = np.asarray(result.y_true).astype(str)
    y_pred = np.asarray(result.y_pred).astype(str)
    groups = np.asarray(groups).astype(str)
    if len(groups) != len(y_true):
        raise ValueError("bootstrap groups and predictions must be row-aligned")

    labels = np.unique(np.concatenate((y_true, y_pred)))
    label_index = {label: index for index, label in enumerate(labels)}
    true_index = np.fromiter(
        (label_index[label] for label in y_true), dtype=np.int64
    )
    pred_index = np.fromiter(
        (label_index[label] for label in y_pred), dtype=np.int64
    )
    unique_groups, group_index = np.unique(groups, return_inverse=True)

    n_labels = len(labels)
    flat_index = (
        group_index * n_labels * n_labels + true_index * n_labels + pred_index
    )
    confusions = np.bincount(
        flat_index,
        minlength=len(unique_groups) * n_labels * n_labels,
    ).reshape(len(unique_groups), n_labels, n_labels)
    true_label_mask = confusions.sum(axis=(0, 2)) > 0
    class_group_counts = (confusions.sum(axis=2) > 0).sum(axis=0)
    return confusions, true_label_mask, int(class_group_counts[true_label_mask].min())


def _metrics_from_confusions(confusions, true_label_mask):
    confusions = np.asarray(confusions, dtype=np.float64)
    true_support = confusions.sum(axis=2)
    pred_support = confusions.sum(axis=1)
    true_positive = np.diagonal(confusions, axis1=1, axis2=2)
    totals = true_support.sum(axis=1)

    recall = np.divide(
        true_positive,
        true_support,
        out=np.zeros_like(true_positive),
        where=true_support > 0,
    )
    f1 = np.divide(
        2 * true_positive,
        true_support + pred_support,
        out=np.zeros_like(true_positive),
        where=(true_support + pred_support) > 0,
    )
    return {
        "accuracy": true_positive.sum(axis=1) / totals,
        "balanced_accuracy": recall[:, true_label_mask].mean(axis=1),
        "macro_f1": f1.mean(axis=1),
        "weighted_f1": (f1 * true_support).sum(axis=1) / totals,
    }


def summarize(result: ClassificationResult, groups, settings):
    """Compute paired group-bootstrap replicates and their uncertainty summary."""
    n_resamples, seed = _parameters(settings)

    confusions, true_label_mask, min_class_groups = _group_confusions(result, groups)
    n_groups = len(confusions)
    if n_groups < 2:
        raise ValueError("grouped bootstrap requires at least two test groups")
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(
        n_groups,
        np.full(n_groups, 1.0 / n_groups),
        size=n_resamples,
    )
    sampled_confusions = np.einsum("bg,gij->bij", weights, confusions)
    values = _metrics_from_confusions(sampled_confusions, true_label_mask)

    replicates = pd.DataFrame(
        {"bootstrap": np.arange(n_resamples, dtype=np.int64), **values}
    )
    rows = []
    for metric in METRICS:
        metric_values = np.asarray(values[metric], dtype=np.float64)
        rows.append(
            {
                "metric": metric,
                "estimate": float(result.metrics[metric]),
                "bootstrap_se": float(metric_values.std(ddof=1)),
                "ci_2_5": float(np.percentile(metric_values, 2.5)),
                "ci_97_5": float(np.percentile(metric_values, 97.5)),
                "n_test_groups": n_groups,
                "min_class_group_count": min_class_groups,
                "n_bootstrap": n_resamples,
                "seed": seed,
                "group_column": str(settings.get("group_column", "")),
                "group_regex": str(settings.get("group_regex", "")),
            }
        )
    return pd.DataFrame(rows), replicates


def save_result(output_dir, summary, replicates) -> tuple[Path, Path]:
    """Write bootstrap summary and paired replicate metrics."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "uncertainty.csv"
    replicates_path = output_dir / "bootstrap_metrics.csv"
    summary.to_csv(summary_path, index=False)
    replicates.to_csv(replicates_path, index=False)
    return summary_path, replicates_path
