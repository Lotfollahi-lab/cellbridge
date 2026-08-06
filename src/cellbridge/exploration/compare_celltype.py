"""Aggregate cell-type classifiers: metrics table, per-class F1, confusion
matrices.

    PYTHONPATH=src python -m cellbridge.exploration.compare_celltype \\
        --results_dir outputs/celltype
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MODELS = ("terra", "terra_neighbor", "uni", "concat", "neighbor_concat")
LEGACY_MODELS = ("terra", "uni", "concat")


def _is_complete(results_dir, model):
    model_dir = results_dir / model
    return (model_dir / "metrics.csv").is_file() and (
        model_dir / "predictions.npz"
    ).is_file()


def _select_models(results_dir):
    if all(_is_complete(results_dir, model) for model in MODELS):
        return MODELS
    if all(_is_complete(results_dir, model) for model in LEGACY_MODELS):
        return LEGACY_MODELS
    raise FileNotFoundError(
        f"'{results_dir}' does not contain a complete five- or three-model result set"
    )


def _load(results_dir, models):
    metrics, preds = {}, {}
    for m in models:
        d = results_dir / m
        metrics[m] = pd.read_csv(d / "metrics.csv").iloc[0]
        with np.load(d / "predictions.npz", allow_pickle=True) as z:
            preds[m] = {k: z[k] for k in z.files}
    return metrics, preds


def _validate_test_sets(preds, models):
    reference = models[0]
    for key, description in (
        ("y_true", "test labels"),
        ("cell_ids_test", "test cells"),
    ):
        expected = preds[reference][key]
        for model in models[1:]:
            if not np.array_equal(preds[model][key], expected):
                raise ValueError(
                    f"{description} differ for '{model}' -- splits not aligned"
                )


def _print_table(metrics, baseline, models):
    model_width = max(10, *(len(model) for model in models))
    header = (
        f"{'Model':{model_width}s} | {'Acc':>6s} | {'BalAcc':>7s} | {'MacroF1':>7s} | "
        f"{'WgtF1':>6s} | {'MacroAUROC':>10s}"
    )
    print(header)
    print("-" * len(header))
    for m in models:
        r = metrics[m]
        print(
            f"{m:{model_width}s} | {r['accuracy']:6.3f} | "
            f"{r['balanced_accuracy']:7.3f} | "
            f"{r['macro_f1']:7.3f} | {r['weighted_f1']:6.3f} | {r['macro_auroc']:10.3f}"
        )
    print(
        f"{'majority':{model_width}s} | {baseline['accuracy']:6.3f} | "
        f"{baseline['balanced_accuracy']:7.3f} | {baseline['macro_f1']:7.3f} | "
        f"{'-':>6s} | {'-':>10s}"
    )


def _confusion_figure(preds, metrics, classes, models, out_path):
    ncols = min(3, len(models))
    nrows = int(np.ceil(len(models) / ncols))
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4.5 * ncols, 4.4 * nrows), squeeze=False
    )
    axes = axes.ravel()
    y_true = preds[models[0]]["y_true"]
    for ax, m in zip(axes, models):
        cm = confusion_matrix(y_true, preds[m]["y_pred"], labels=classes)
        cmn = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
        ax.imshow(cmn, vmin=0, vmax=1, cmap="Blues")
        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(classes)))
        ax.set_yticklabels(classes, fontsize=8)
        ax.set_xlabel("predicted")
        if ax.get_subplotspec().is_first_col():
            ax.set_ylabel("true")
        ax.set_title(f"{m} (bal.acc {metrics[m]['balanced_accuracy']:.2f})")
        if len(classes) <= 10:
            for i in range(len(classes)):
                for j in range(len(classes)):
                    ax.text(
                        j,
                        i,
                        f"{cmn[i, j]:.2f}",
                        ha="center",
                        va="center",
                        color="white" if cmn[i, j] > 0.5 else "black",
                        fontsize=7,
                    )
    for ax in axes[len(models) :]:
        ax.set_visible(False)
    fig.suptitle("Row-normalized confusion matrices (test)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _f1_figure(perclass_f1, classes, models, out_path):
    x = np.arange(len(classes))
    width = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(max(8, len(classes) * 1.2), 4.2))
    for i, m in enumerate(models):
        offset = (i - (len(models) - 1) / 2) * width
        ax.bar(x + offset, perclass_f1[m], width=width, label=m)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_ylabel("F1 (test)")
    ax.set_ylim(0, 1)
    ax.set_title("Per-class F1 by model")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Compare cell-type classifiers.")
    parser.add_argument("--results_dir", required=True)
    args = parser.parse_args()
    results_dir = Path(args.results_dir)

    models = _select_models(results_dir)
    metrics, preds = _load(results_dir, models)
    _validate_test_sets(preds, models)

    y_true = preds[models[0]]["y_true"]
    classes = list(preds[models[0]]["classes"])

    vals, cnts = np.unique(y_true, return_counts=True)
    majority = vals[cnts.argmax()]
    y_major = np.full_like(y_true, majority)
    baseline = {
        "accuracy": float((y_major == y_true).mean()),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_major)),
        "macro_f1": float(f1_score(y_true, y_major, average="macro")),
    }

    print(f"test cells: {len(y_true)} | classes: {classes}")
    print("test-set support:", dict(zip(vals.tolist(), cnts.tolist())))
    print()
    _print_table(metrics, baseline, models)

    perclass_f1 = {}
    for m in models:
        _, _, f1, _ = precision_recall_fscore_support(
            y_true, preds[m]["y_pred"], labels=classes, zero_division=0
        )
        perclass_f1[m] = f1
    class_width = max(14, *(len(str(cls)) for cls in classes))
    model_width = max(8, *(len(model) for model in models))
    print("\nPer-class F1:")
    print(
        f"  {'class':{class_width}s} "
        + " ".join(f"{model:>{model_width}s}" for model in models)
    )
    for i, c in enumerate(classes):
        print(
            f"  {c:{class_width}s} "
            + " ".join(f"{perclass_f1[m][i]:{model_width}.3f}" for m in models)
        )

    cm_path = results_dir / "confusion_matrices.png"
    f1_path = results_dir / "per_class_f1.png"
    _confusion_figure(preds, metrics, classes, models, cm_path)
    _f1_figure(perclass_f1, classes, models, f1_path)

    scores = {m: float(metrics[m]["balanced_accuracy"]) for m in models}
    print(
        "\nBalanced accuracy -- "
        + " ".join(f"{model}={scores[model]:.3f}" for model in models)
    )
    comparisons = [("concat", "terra", "uni")]
    if "neighbor_concat" in models:
        comparisons.append(("neighbor_concat", "terra_neighbor", "uni"))
    for combined, first, second in comparisons:
        beats_both = (
            scores[combined] > scores[first] and scores[combined] > scores[second]
        )
        print(
            f"{combined} beats BOTH {first} and {second} (balanced acc): "
            f"{'YES' if beats_both else 'NO'}"
        )
    print(f"\nSaved figures:\n  {cm_path}\n  {f1_path}")


if __name__ == "__main__":
    main()
