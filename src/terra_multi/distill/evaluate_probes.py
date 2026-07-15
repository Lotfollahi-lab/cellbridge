"""Evaluate UNI, TERRA, and distilled embeddings with linear probes."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import f1_score

from terra_multi.distill.data import LABEL_SUFFIX
from terra_multi.distill.make_splits import (
    label_test_split_name,
    select_label_test_samples,
)
from terra_multi.evaluation.classification import method_directory, resolve_methods

ARMS = {
    "uni_baseline": ("uni", False),
    "terra_teacher": ("terra", False),
    "distilled": ("terra", True),
    "concat_oracle": ("concat", False),
}

PROTOTYPE_REPRESENTATIONS = {
    "prototype_penultimate": "features_penultimate",
    "prototype_probs": "prototype_probs",
}


def prototype_arm_names(cfg):
    return (
        f"prototype_penultimate_{cfg.get('feat_dim', 512)}",
        f"prototype_probs_{cfg.get('num_prototypes', 256)}",
    )


def _prototype_key(arm):
    """Representation key for a prototype arm, or None for a v0 arm."""
    for prefix, key in PROTOTYPE_REPRESENTATIONS.items():
        if arm.startswith(f"{prefix}_"):
            return key
    return None


def arms_for_config(cfg):
    if cfg.get("distill_version") == "v1":
        return prototype_arm_names(cfg)
    return tuple(ARMS)


def _model_for_arm(arm):
    return "terra" if _prototype_key(arm) else ARMS[arm][0]


def _sample_path(data_root, sample_id, suffix):
    return str(Path(data_root) / sample_id / f"{sample_id}{suffix}")


def _paths(cfg):
    output = Path(cfg["output_dir"])
    split_name = label_test_split_name(cfg)
    return (
        output,
        output / f"{split_name}_distilled",
        output / f"{split_name}_probes",
    )


def _label_test_samples(cfg, limit=None):
    splits = pd.read_csv(cfg["splits_path"])
    samples = splits.loc[splits["split"] == "label_test", "sample_id"].astype(str)
    selected = select_label_test_samples(cfg, samples.tolist())
    return selected[:limit] if limit is not None else selected


def _keep_classes(labels_path, label_column, min_count):
    counts = pd.read_csv(labels_path)[label_column].value_counts()
    return counts.index[counts >= min_count].astype(str).tolist()


def _arm_config(cfg, sample_id, arm, distilled_dir, output_dir):
    settings = cfg["eval"]
    model = _model_for_arm(arm)
    data_root = cfg["data_root"]
    labels_path = _sample_path(data_root, sample_id, LABEL_SUFFIX)
    prototype_key = _prototype_key(arm)
    if prototype_key:
        split_name = label_test_split_name(cfg)
        terra_path = str(
            Path(cfg["output_dir"])
            / f"{split_name}_features"
            / f"{sample_id}_prototype.npz"
        )
        terra_key = prototype_key
    else:
        use_distilled = ARMS[arm][1]
        terra_path = (
            str(distilled_dir / f"{sample_id}_terra.npz")
            if use_distilled
            else _sample_path(data_root, sample_id, "_terra.npz")
        )
        terra_key = cfg["terra_key"]
    config = {
        "sample_id": sample_id,
        "terra_embeddings": terra_path,
        "uni_embeddings": _sample_path(data_root, sample_id, "_uni2.npz"),
        "terra_key": terra_key,
        "terra_neighbor_key": "neighborhood_emb",
        "uni_key": cfg["uni_key"],
        "adata_path": _sample_path(data_root, sample_id, ".h5ad"),
        "labels_path": labels_path,
        "label_id_column": settings["label_id_column"],
        "label_column": settings["label_column"],
        "keep_classes": _keep_classes(
            labels_path, settings["label_column"], settings["min_class_count"]
        ),
        "test_frac": settings["test_frac"],
        "split_axis": settings["split_axis"],
        "pca": settings["pca"],
        "uni_pca_components": settings["uni_pca_components"],
        "Cs": settings["Cs"],
        "class_weight": settings["class_weight"],
        "output_dir": str(output_dir),
        "model": model,
    }
    for name in ("methods", "knn_neighbors", "knn_metric", "knn_weights"):
        if name in settings:
            config[name] = settings[name]
    return config


def _run_probe(config_path, model, root):
    env = dict(os.environ)
    src = str(root / "src")
    env["PYTHONPATH"] = (
        src + os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else src
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "terra_multi.evaluation.run",
            "--config",
            str(config_path),
            "--model",
            model,
        ],
        cwd=root,
        env=env,
        check=True,
    )


def run_arm(cfg, arm, samples, root):
    _, distilled_dir, probe_root = _paths(cfg)
    model = _model_for_arm(arm)
    for sample_id in samples:
        output_dir = probe_root / sample_id / arm
        output_dir.mkdir(parents=True, exist_ok=True)
        config_path = output_dir / "config.yaml"
        with open(config_path, "w") as fh:
            yaml.safe_dump(
                _arm_config(cfg, sample_id, arm, distilled_dir, output_dir), fh
            )
        _run_probe(config_path, model, root)
        print(f"[{arm}] {sample_id} done")


def _read_result(model_dir):
    metrics = pd.read_csv(model_dir / "metrics.csv").iloc[0].to_dict()
    with np.load(model_dir / "predictions.npz", allow_pickle=True) as data:
        classes = data["classes"].astype(str)
        scores = f1_score(
            data["y_true"],
            data["y_pred"],
            labels=classes,
            average=None,
            zero_division=0,
        )
    return metrics, dict(zip(classes, scores.astype(float)))


def _aggregate_method(cfg, samples, method):
    output_dir, _, probe_root = _paths(cfg)
    split_name = label_test_split_name(cfg)
    arms = arms_for_config(cfg)
    headline_rows, class_rows, missing = [], [], []
    for sample_id in samples:
        for arm in arms:
            model = _model_for_arm(arm)
            base_dir = probe_root / sample_id / arm
            model_dir = method_directory(base_dir, method) / model
            required = (model_dir / "metrics.csv", model_dir / "predictions.npz")
            if not all(path.exists() for path in required):
                missing.append(f"{sample_id}/{arm}")
                continue
            metrics, per_class = _read_result(model_dir)
            headline_rows.append(
                {
                    "sample_id": sample_id,
                    "arm": arm,
                    **{
                        name: metrics[name]
                        for name in (
                            "accuracy",
                            "balanced_accuracy",
                            "macro_f1",
                            "macro_auroc",
                            "n_classes",
                        )
                    },
                }
            )
            class_rows.extend(
                {
                    "sample_id": sample_id,
                    "arm": arm,
                    "class": label,
                    "f1": score,
                }
                for label, score in per_class.items()
            )

    if missing:
        raise SystemExit(
            f"{len(missing)} incomplete {method} runs, e.g. {missing[:5]}"
        )
    if not headline_rows:
        raise SystemExit(f"no complete {method} probe runs found")

    headline = pd.DataFrame(headline_rows)
    method_name = "" if method == "linear" else f"_{method}"
    headline.to_csv(
        output_dir / f"{split_name}{method_name}_probe_metrics.csv", index=False
    )
    pd.DataFrame(class_rows).to_csv(
        output_dir / f"{split_name}{method_name}_per_class_f1.csv", index=False
    )
    metrics = ["accuracy", "balanced_accuracy", "macro_f1", "macro_auroc"]
    summary = headline.groupby("arm")[metrics].agg(["mean", "std"])
    summary.columns = [f"{name}_{stat}" for name, stat in summary.columns]
    summary = summary.reindex([arm for arm in arms if arm in summary.index])
    summary.to_csv(output_dir / f"{split_name}{method_name}_summary.csv")
    print(summary.to_string())


def aggregate(cfg, samples):
    for method in resolve_methods(cfg["eval"]):
        _aggregate_method(cfg, samples, method)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    # Valid arms depend on the config (v1 names embed feat_dim / num_prototypes),
    # so the value is checked against arms_for_config once the config is loaded.
    parser.add_argument("--arm")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--limit-samples", type=int)
    args = parser.parse_args()
    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    samples = _label_test_samples(cfg, args.limit_samples)
    arms = arms_for_config(cfg)
    if args.arm and args.arm not in arms:
        parser.error(f"arm {args.arm!r} is not configured for this experiment")
    root = Path.cwd()
    if args.aggregate_only:
        aggregate(cfg, samples)
    elif args.arm:
        run_arm(cfg, args.arm, samples, root)
    else:
        for arm in arms:
            run_arm(cfg, arm, samples, root)
        aggregate(cfg, samples)


if __name__ == "__main__":
    main()
