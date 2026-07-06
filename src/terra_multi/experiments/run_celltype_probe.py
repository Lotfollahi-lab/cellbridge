"""Cell-type classification probe: one modality per job.

    PYTHONPATH=src python -m terra_multi.experiments.run_celltype_probe \\
        --config configs/celltype.yaml --model {terra|uni|concat}

Mirrors run_linear_probe (same cells, same spatial split, same UNI PCA), but the
target is the CellViT class and the estimator is a logistic-regression classifier.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)

from terra_multi.experiments.run_linear_probe import apply_uni_pca
from terra_multi.helpers.io import align_by_cell_id, load_coords, load_embeddings
from terra_multi.helpers.probe import fit_classifier
from terra_multi.helpers.splits import spatial_split

MODELS = ("terra", "uni", "concat")


def load_labels(path, keep_classes):
    """Return a ``{cell_id: cell_type}`` dict restricted to ``keep_classes``."""
    df = pd.read_parquet(path)
    df = df[df["cell_type"].isin(set(keep_classes))]
    return dict(zip(df["cell_id"].astype(str), df["cell_type"].astype(str)))


def main():
    parser = argparse.ArgumentParser(description="Run one cell-type classifier.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True, choices=MODELS)
    args = parser.parse_args()
    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    model = args.model

    # Same common cell set + coords as the expression probe.
    terra_emb, terra_ids = load_embeddings(
        cfg["terra_embeddings"], key=cfg["terra_key"]
    )
    uni_emb, uni_ids = load_embeddings(cfg["uni_embeddings"], key=cfg["uni_key"])
    coords_all, coord_ids = load_coords(cfg["adata_path"])
    (terra_al, uni_al, coords), common_ids = align_by_cell_id(
        [(terra_emb, terra_ids), (uni_emb, uni_ids), (coords_all, coord_ids)]
    )

    # Restrict to cells carrying a kept CellViT label.
    label_map = load_labels(cfg["labels_path"], cfg["keep_classes"])
    keep = np.array([cid in label_map for cid in common_ids])
    ids = common_ids[keep]
    y = np.array([label_map[cid] for cid in ids])
    terra_al, uni_al, coords = terra_al[keep], uni_al[keep], coords[keep]
    print(f"[{model}] labelled cells: {len(ids):,} / {len(common_ids):,}")
    for cls, n in zip(*np.unique(y, return_counts=True)):
        print(f"    {cls:14s} {n}")

    # Same spatial split as before.
    train_idx, test_idx = spatial_split(
        ids, coords, cfg["test_frac"], cfg["split_axis"]
    )

    # PCA-reduce UNI (fit on train only), for the uni and concat models.
    if bool(cfg.get("pca", False)) and model in {"uni", "concat"}:
        uni_al, ev = apply_uni_pca(
            uni_al, train_idx, n_components=cfg.get("uni_pca_components", 256)
        )
        print(f"[{model}] UNI PCA -> {uni_al.shape[1]} dims (train EV={ev:.3f})")

    if model == "terra":
        X = terra_al
    elif model == "uni":
        X = uni_al
    else:  # concat
        X = np.concatenate([terra_al, uni_al], axis=1)
    print(
        f"[{model}] features {X.shape} | train={len(train_idx)} test={len(test_idx)} "
        f"(held out top {cfg['test_frac']:.0%} along {cfg['split_axis']})"
    )

    clf = fit_classifier(
        X[train_idx], y[train_idx], cfg["Cs"], cfg.get("class_weight", "balanced")
    )
    classes = clf.classes_
    y_test = y[test_idx]
    y_pred = clf.predict(X[test_idx])
    y_proba = clf.predict_proba(X[test_idx])

    try:
        auroc = float(
            roc_auc_score(
                y_test, y_proba, multi_class="ovr", average="macro", labels=classes
            )
        )
    except ValueError:
        auroc = float("nan")  # a class absent from the test fold

    summary = {
        "model": model,
        "n_features": int(X.shape[1]),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        "n_classes": int(len(classes)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "macro_f1": float(f1_score(y_test, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_test, y_pred, average="weighted")),
        "macro_auroc": auroc,
    }

    out_dir = Path(cfg["output_dir"]) / model
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([summary]).to_csv(out_dir / "metrics.csv", index=False)
    np.savez(
        out_dir / "predictions.npz",
        y_true=y_test,
        y_pred=y_pred,
        y_proba=y_proba,
        classes=np.asarray(classes),
        cell_ids_test=ids[test_idx],
    )

    print(f"\n[{model}] summary")
    for k, v in summary.items():
        print(f"  {k:20s} {v}")
    print(
        f"[{model}] wrote {out_dir / 'metrics.csv'} and {out_dir / 'predictions.npz'}"
    )


if __name__ == "__main__":
    main()
