"""Aggregate the three cell-type classifiers: metrics table, per-class F1,
confusion matrices.

    PYTHONPATH=src python -m terra_multi.experiments.compare_celltype \\
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

MODELS = ("terra", "uni", "concat")


def _load(results_dir):
    metrics, preds = {}, {}
    for m in MODELS:
        d = results_dir / m
        metrics[m] = pd.read_csv(d / "metrics.csv").iloc[0]
        z = np.load(d / "predictions.npz", allow_pickle=True)
        preds[m] = {k: z[k] for k in z.files}
    return metrics, preds


def _print_table(metrics, baseline):
    header = (
        f"{'Model':10s} | {'Acc':>6s} | {'BalAcc':>7s} | {'MacroF1':>7s} | "
        f"{'WgtF1':>6s} | {'MacroAUROC':>10s}"
    )
    print(header)
    print("-" * len(header))
    for m in MODELS:
        r = metrics[m]
        print(
            f"{m:10s} | {r['accuracy']:6.3f} | {r['balanced_accuracy']:7.3f} | "
            f"{r['macro_f1']:7.3f} | {r['weighted_f1']:6.3f} | {r['macro_auroc']:10.3f}"
        )
    print(
        f"{'majority':10s} | {baseline['accuracy']:6.3f} | "
        f"{baseline['balanced_accuracy']:7.3f} | {baseline['macro_f1']:7.3f} | "
        f"{'-':>6s} | {'-':>10s}"
    )


def _confusion_figure(preds, metrics, classes, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
    for ax, m in zip(axes, MODELS):
        cm = confusion_matrix(
            preds["terra"]["y_true"], preds[m]["y_pred"], labels=classes
        )
        cmn = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
        ax.imshow(cmn, vmin=0, vmax=1, cmap="Blues")
        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels(classes, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(classes)))
        ax.set_yticklabels(classes, fontsize=8)
        ax.set_xlabel("predicted")
        if m == "terra":
            ax.set_ylabel("true")
        ax.set_title(f"{m} (bal.acc {metrics[m]['balanced_accuracy']:.2f})")
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
    fig.suptitle("Row-normalized confusion matrices (test)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _f1_figure(perclass_f1, classes, out_path):
    x = np.arange(len(classes))
    width = 0.27
    fig, ax = plt.subplots(figsize=(max(6, len(classes) * 1.8), 4.2))
    for i, m in enumerate(MODELS):
        ax.bar(x + (i - 1) * width, perclass_f1[m], width=width, label=m)
    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_ylabel("F1 (test)")
    ax.set_ylim(0, 1)
    ax.set_title("Per-class F1 by model")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Compare cell-type classifiers.")
    parser.add_argument("--results_dir", required=True)
    args = parser.parse_args()
    results_dir = Path(args.results_dir)

    metrics, preds = _load(results_dir)

    # The three models share an identical test set (same split on the same cells).
    y_true = preds["terra"]["y_true"]
    for m in MODELS:
        if not np.array_equal(preds[m]["y_true"], y_true):
            raise ValueError(f"test labels differ for '{m}' -- splits not aligned")
    classes = list(preds["terra"]["classes"])

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
    _print_table(metrics, baseline)

    perclass_f1 = {}
    for m in MODELS:
        _, _, f1, _ = precision_recall_fscore_support(
            y_true, preds[m]["y_pred"], labels=classes, zero_division=0
        )
        perclass_f1[m] = f1
    print("\nPer-class F1:")
    print(f"  {'class':14s} " + " ".join(f"{m:>8s}" for m in MODELS))
    for i, c in enumerate(classes):
        print(f"  {c:14s} " + " ".join(f"{perclass_f1[m][i]:8.3f}" for m in MODELS))

    cm_path = results_dir / "confusion_matrices.png"
    f1_path = results_dir / "per_class_f1.png"
    _confusion_figure(preds, metrics, classes, cm_path)
    _f1_figure(perclass_f1, classes, f1_path)

    tb = float(metrics["terra"]["balanced_accuracy"])
    ub = float(metrics["uni"]["balanced_accuracy"])
    cb = float(metrics["concat"]["balanced_accuracy"])
    print(f"\nBalanced accuracy -- terra={tb:.3f} uni={ub:.3f} concat={cb:.3f}")
    print(
        f"Concat beats BOTH single modalities (balanced acc): {'YES' if cb > tb and cb > ub else 'NO'}"
    )
    print(f"\nSaved figures:\n  {cm_path}\n  {f1_path}")


if __name__ == "__main__":
    main()
