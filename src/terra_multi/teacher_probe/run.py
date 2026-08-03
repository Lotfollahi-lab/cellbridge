"""Run spatial-block CV probes for pooled, expression, and gene-token arms."""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)

from terra_multi.evaluation.linear_probe import fit_linear_classifier
from terra_multi.teacher_probe.config import (
    ensure_output_directories,
    load_teacher_probe_config,
)
from terra_multi.teacher_probe.data import load_probe_data
from terra_multi.teacher_probe.features import (
    build_fold_token_features,
    gene_feature_slice,
)
from terra_multi.teacher_probe.plotting import (
    plot_confusion_matrices,
    plot_gene_comparisons,
)
from terra_multi.teacher_probe.reporting import summarize_results, write_markdown
from terra_multi.teacher_probe.splits import spatial_block_folds

ARMS = ("cell_emb", "cell_emb_expression", "cell_emb_expression_gene_tokens")


def _evaluate_arm(features, data, fold, cfg, classes, arm):
    model = fit_linear_classifier(
        features[fold.train],
        data.labels[fold.train],
        cfg.get("Cs", [0.1, 1, 10]),
        cfg.get("class_weight", "balanced"),
        cfg.get("max_iter", 2000),
    )
    predicted = model.predict(features[fold.test])
    truth = data.labels[fold.test]
    metrics = {
        "fold": fold.number,
        "arm": arm,
        "n_train": len(fold.train),
        "n_test": len(fold.test),
        "n_features": features.shape[1],
        "best_C": float(np.asarray(model.named_steps["logreg"].C_).ravel()[0]),
        "macro_f1": f1_score(
            truth, predicted, labels=classes, average="macro", zero_division=0
        ),
        "accuracy": accuracy_score(truth, predicted),
        "balanced_accuracy": balanced_accuracy_score(truth, predicted),
        "weighted_f1": f1_score(
            truth, predicted, labels=classes, average="weighted", zero_division=0
        ),
    }
    class_scores = f1_score(
        truth, predicted, labels=classes, average=None, zero_division=0
    )
    supports = np.asarray([(truth == label).sum() for label in classes])
    per_class = [
        {
            "fold": fold.number,
            "arm": arm,
            "class": label,
            "f1": score,
            "support": support,
        }
        for label, score, support in zip(classes, class_scores, supports)
    ]
    matrix = confusion_matrix(truth, predicted, labels=classes)
    confusion = [
        {
            "fold": fold.number,
            "arm": arm,
            "true": true,
            "predicted": predicted_label,
            "count": int(matrix[i, j]),
        }
        for i, true in enumerate(classes)
        for j, predicted_label in enumerate(classes)
    ]
    predictions = pd.DataFrame(
        {
            "fold": fold.number,
            "arm": arm,
            "cell_id": data.cell_ids[fold.test],
            "y_true": truth,
            "y_pred": predicted,
        }
    )
    return model, metrics, per_class, confusion, predictions


def _gene_importance(model, data, fold, n_components, base_width):
    coefficients = np.asarray(model.named_steps["logreg"].coef_)
    rows = []
    for gene_index, gene in enumerate(data.genes):
        pc_slice, mask_column = gene_feature_slice(gene_index, n_components)
        pc_slice = slice(base_width + pc_slice.start, base_width + pc_slice.stop)
        rows.append(
            {
                "fold": fold,
                "gene": gene,
                "pc_coefficient_norm": float(
                    np.linalg.norm(coefficients[:, pc_slice], axis=1).mean()
                ),
                "mask_coefficient_norm": float(
                    np.linalg.norm(coefficients[:, base_width + mask_column], axis=0)
                ),
            }
        )
    return rows


def run(cfg, *, overwrite_features=False):
    output_dir, cache_dir = ensure_output_directories(cfg)
    data = load_probe_data(cfg)
    folds, block_ids = spatial_block_folds(
        data.coords_um,
        cfg["eval"]["n_splits"],
        cfg["eval"]["spatial_block_size_um"],
        data.labels,
    )
    classes = np.unique(data.labels)
    pd.DataFrame(
        {
            "cell_id": data.cell_ids,
            "spatial_block": block_ids,
            "test_fold": np.concatenate(
                [np.full(len(fold.test), fold.number) for fold in folds]
            )[np.argsort(np.concatenate([fold.test for fold in folds]))],
        }
    ).to_csv(output_dir / "spatial_folds.csv", index=False)
    with open(output_dir / "alignment_report.json", "w") as handle:
        json.dump(data.alignment_report, handle, indent=2, sort_keys=True)
    with open(output_dir / "resolved_config.yaml", "w") as handle:
        yaml.safe_dump(cfg, handle, sort_keys=False)

    metrics_rows, class_rows, confusion_rows, predictions = [], [], [], []
    pca_rows, importance_rows, fold_zero_features = [], [], None
    base_expression = np.concatenate([data.cell_emb, data.expression], axis=1)
    for fold in folds:
        token_features, stats = build_fold_token_features(
            data,
            fold.train,
            fold.number,
            cfg["eval"],
            cache_dir,
            overwrite=overwrite_features,
        )
        pca_rows.extend(stats)
        if fold.number == 0:
            fold_zero_features = token_features
        arms = {
            "cell_emb": data.cell_emb,
            "cell_emb_expression": base_expression,
            "cell_emb_expression_gene_tokens": np.concatenate(
                [base_expression, token_features], axis=1
            ),
        }
        for arm in ARMS:
            model, metrics, per_class, confusion, prediction = _evaluate_arm(
                arms[arm], data, fold, cfg["eval"], classes, arm
            )
            metrics_rows.append(metrics)
            class_rows.extend(per_class)
            confusion_rows.extend(confusion)
            predictions.append(prediction)
            if arm == "cell_emb_expression_gene_tokens":
                importance_rows.extend(
                    _gene_importance(
                        model,
                        data,
                        fold.number,
                        cfg["eval"]["gene_pca_components"],
                        base_expression.shape[1],
                    )
                )

    metrics = pd.DataFrame(metrics_rows)
    per_class = pd.DataFrame(class_rows)
    confusion = pd.DataFrame(confusion_rows)
    importance = pd.DataFrame(importance_rows)
    summary, class_summary, delta = summarize_results(metrics, per_class)
    metrics.to_csv(output_dir / "fold_metrics.csv", index=False)
    summary.to_csv(output_dir / "summary.csv", index=False)
    delta.to_csv(output_dir / "key_delta.csv", index=False)
    per_class.to_csv(output_dir / "per_class_f1.csv", index=False)
    class_summary.to_csv(output_dir / "per_class_f1_summary.csv", index=False)
    confusion.to_csv(output_dir / "confusion_matrices.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(
        output_dir / "predictions.csv.gz", index=False, compression="gzip"
    )
    pd.DataFrame(pca_rows).to_json(
        output_dir / "gene_pca.json", orient="records", indent=2
    )
    importance.to_csv(output_dir / "gene_importance.csv", index=False)
    write_markdown(output_dir / "REPORT.md", summary, delta)
    plot_confusion_matrices(confusion, classes, output_dir / "confusion_matrices.png")
    plot_gene_comparisons(
        data,
        fold_zero_features,
        importance,
        cfg["eval"]["gene_pca_components"],
        output_dir / "gene_plots",
        top_n=cfg["plots"].get("top_genes", 6),
        max_points=cfg["plots"].get("max_points", 5000),
        seed=cfg["eval"].get("seed", 0),
    )
    print((output_dir / "REPORT.md").read_text())
    return summary, delta


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--overwrite-features", action="store_true")
    args = parser.parse_args(argv)
    cfg = load_teacher_probe_config(args.config, smoke=args.smoke)
    run(cfg, overwrite_features=args.overwrite_features)


if __name__ == "__main__":
    main()
