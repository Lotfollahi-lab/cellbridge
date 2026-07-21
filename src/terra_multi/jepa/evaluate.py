"""Evaluate JEPA embeddings on the held-out HEST fast split."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from terra_multi.cohorts import cohort_in_split
from terra_multi.config import load_config
from terra_multi.evaluation.classification import resolve_methods
from terra_multi.evaluation.hest import result_directory, run_hest_evaluations

ARMS = {
    "jepa": "terra",
    "uni_baseline": "uni",
    "terra_teacher": "terra",
}


def fast_samples(cfg: dict) -> list[str]:
    """Return the configured held-out sample IDs in their declared order."""

    return cohort_in_split(
        cfg["eval"]["label_test_fast_path"],
        cfg["splits_path"],
        "label_test",
    )


def _sample_path(cfg: dict, sample_id: str, suffix: str) -> Path:
    return Path(cfg["data_root"]) / sample_id / f"{sample_id}{suffix}"


def _kept_classes(cfg: dict, sample_id: str) -> list[str]:
    settings = cfg["eval"]
    labels = pd.read_csv(_sample_path(cfg, sample_id, "_granular_cell_labels.csv"))
    column = settings["label_column"]
    if column not in labels:
        raise ValueError(f"label file for {sample_id} is missing {column!r}")
    counts = labels[column].astype(str).value_counts()
    return counts.index[counts >= int(settings["min_class_count"])].tolist()


def arm_config(cfg: dict, sample_id: str, arm: str) -> dict:
    """Build the shared HEST evaluator config for one JEPA comparison arm."""

    if arm not in ARMS:
        raise ValueError(f"unknown JEPA evaluation arm {arm!r}")
    settings = cfg["eval"]
    raw_terra = _sample_path(cfg, sample_id, "_terra.npz")
    raw_uni = _sample_path(cfg, sample_id, "_uni2.npz")
    jepa_path = Path(cfg["output_dir"]) / "features" / f"{sample_id}.npz"
    if arm == "terra_teacher":
        terra_path, terra_key = raw_terra, cfg.get("terra_key", "cell_emb")
        # The shared loader aligns all supplied artifacts. Using JEPA IDs as the
        # otherwise-unused image input keeps every comparison on identical cells.
        uni_path, uni_key = jepa_path, "jepa_emb"
    else:
        terra_path, terra_key = jepa_path, "jepa_emb"
        uni_path, uni_key = raw_uni, cfg.get("uni_key", "uni_emb")
    output = (
        Path(cfg["output_dir"])
        / "evaluation"
        / "hest_label_test_fast"
        / "probes"
        / sample_id
        / arm
    )
    result = {
        "sample_id": sample_id,
        "terra_embeddings": str(terra_path),
        "uni_embeddings": str(uni_path),
        "terra_key": terra_key,
        "terra_neighbor_key": "neighborhood_emb",
        "uni_key": uni_key,
        "adata_path": str(_sample_path(cfg, sample_id, ".h5ad")),
        "labels_path": str(_sample_path(cfg, sample_id, "_granular_cell_labels.csv")),
        "label_id_column": settings["label_id_column"],
        "label_column": settings["label_column"],
        "keep_classes": _kept_classes(cfg, sample_id),
        "test_frac": settings["test_frac"],
        "split_axis": settings["split_axis"],
        "pca": settings["pca"],
        "uni_pca_components": settings["uni_pca_components"],
        "Cs": settings["Cs"],
        "class_weight": settings["class_weight"],
        "output_dir": str(output),
    }
    for name in ("methods", "knn_neighbors", "knn_metric", "knn_weights"):
        if name in settings:
            result[name] = settings[name]
    return result


def evaluate_samples(cfg: dict, samples: list[str]) -> None:
    """Run every configured arm and classifier for the requested samples."""

    methods = resolve_methods(cfg["eval"])
    for sample_id in samples:
        for arm, model in ARMS.items():
            settings = arm_config(cfg, sample_id, arm)
            run_hest_evaluations(settings, model, methods)
            print(f"[{sample_id}/{arm}] done")


def aggregate(cfg: dict, samples: list[str]) -> None:
    """Write per-sample metrics and cross-sample summaries."""

    root = Path(cfg["output_dir"]) / "evaluation" / "hest_label_test_fast"
    for method in resolve_methods(cfg["eval"]):
        rows = []
        for sample_id in samples:
            for arm, model in ARMS.items():
                base = root / "probes" / sample_id / arm
                path = result_directory(base, model, method) / "metrics.csv"
                if not path.exists():
                    raise FileNotFoundError(f"missing {method} result: {path}")
                metrics = pd.read_csv(path).iloc[0]
                rows.append(
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

        suffix = "" if method == "linear" else f"_{method}"
        table = pd.DataFrame(rows)
        root.mkdir(parents=True, exist_ok=True)
        table.to_csv(root / f"probe_metrics{suffix}.csv", index=False)
        metrics = ["accuracy", "balanced_accuracy", "macro_f1", "macro_auroc"]
        summary = table.groupby("arm", sort=False)[metrics].agg(["mean", "std"])
        summary.columns = [f"{name}_{stat}" for name, stat in summary.columns]
        summary.to_csv(root / f"probe_summary{suffix}.csv")
        print(f"\n{method}\n{summary.to_string()}")


def _parse_samples(value: str | None, configured: list[str]) -> list[str]:
    if value is None:
        return configured
    requested = [sample.strip() for sample in value.split(",") if sample.strip()]
    invalid = sorted(set(requested) - set(configured))
    if not requested or invalid:
        raise ValueError(f"invalid fast-split samples: {invalid or requested}")
    return requested


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--samples", help="comma-separated fast-split sample IDs")
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    configured = fast_samples(cfg)
    samples = _parse_samples(args.samples, configured)
    if args.aggregate_only:
        aggregate(cfg, samples)
    else:
        evaluate_samples(cfg, samples)


if __name__ == "__main__":
    main()
