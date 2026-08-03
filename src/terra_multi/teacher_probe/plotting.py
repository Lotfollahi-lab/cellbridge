"""Compact plots for teacher-only gene-token probe results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from terra_multi.teacher_probe.features import gene_feature_slice


def plot_confusion_matrices(confusion, classes, path):
    """Plot fold-summed, true-class-normalized confusion matrices by arm."""
    arms = confusion["arm"].drop_duplicates().tolist()
    figure, axes = plt.subplots(
        1, len(arms), figsize=(6 * len(arms), 5.5), squeeze=False
    )
    for axis, arm in zip(axes[0], arms):
        table = confusion[confusion["arm"] == arm]
        matrix = (
            table.groupby(["true", "predicted"])["count"]
            .sum()
            .unstack(fill_value=0)
            .reindex(index=classes, columns=classes, fill_value=0)
            .to_numpy(dtype=float)
        )
        denominator = matrix.sum(axis=1, keepdims=True)
        normalized = np.divide(
            matrix, denominator, out=np.zeros_like(matrix), where=denominator > 0
        )
        image = axis.imshow(normalized, vmin=0, vmax=1, cmap="Blues", aspect="auto")
        axis.set_title(arm.replace("_", " "))
        axis.set_xlabel("Predicted")
        axis.set_ylabel("True")
        axis.set_xticks(range(len(classes)), classes, rotation=90, fontsize=6)
        axis.set_yticks(range(len(classes)), classes, fontsize=6)
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _sample_rows(rows, maximum, seed):
    if len(rows) <= maximum:
        return rows
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(rows, size=maximum, replace=False))


def plot_gene_comparisons(
    data,
    fold_features,
    importance,
    n_components,
    output_dir,
    *,
    top_n=6,
    max_points=5000,
    seed=0,
):
    """Compare normalized expression with contextual PC1/PC2 for top genes."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ranked = (
        importance.groupby("gene")["pc_coefficient_norm"]
        .mean()
        .sort_values(ascending=False)
        .head(int(top_n))
    )
    classes = np.unique(data.labels)
    palette = plt.get_cmap("tab20", len(classes))
    class_index = {label: index for index, label in enumerate(classes)}
    for rank, gene in enumerate(ranked.index):
        gene_index = int(np.flatnonzero(data.genes == gene)[0])
        pc_slice, _ = gene_feature_slice(gene_index, n_components)
        present = np.flatnonzero(data.presence[:, gene_index])
        rows = _sample_rows(present, int(max_points), seed + rank)
        expression = data.expression[rows, gene_index]
        pcs = fold_features[rows, pc_slice]
        colors = [palette(class_index[label]) for label in data.labels[rows]]
        figure, axes = plt.subplots(1, 3, figsize=(12, 3.5))
        panels = (
            (expression, pcs[:, 0], "log-normalized expression", "contextual PC1"),
            (expression, pcs[:, 1], "log-normalized expression", "contextual PC2"),
            (pcs[:, 0], pcs[:, 1], "contextual PC1", "contextual PC2"),
        )
        for axis, (x, y, x_label, y_label) in zip(axes, panels):
            axis.scatter(x, y, c=colors, s=3, alpha=0.25, linewidths=0)
            axis.set_xlabel(x_label)
            axis.set_ylabel(y_label)
        figure.suptitle(
            f"{gene}: contextual token PCs vs expression "
            f"(fold-0 train PCA; coefficient rank {rank + 1})"
        )
        figure.tight_layout()
        figure.savefig(output_dir / f"{rank + 1:02d}_{gene}.png", dpi=180)
        plt.close(figure)
