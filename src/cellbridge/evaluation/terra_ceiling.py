"""Probe the frozen TERRA teacher at a chosen label granularity.

TERRA sees the molecular data the image model never does, so its score bounds
what any H&E model could reach. The existing arms are compared at raw
source-cluster granularity; this reports the same ceiling at whatever
``eval.label_level`` is configured, using the identical split, probe, and
metrics so the numbers line up with every other arm.

TERRA is used at full 384-D and is deliberately not PCA-reduced, matching the
established teacher protocol.

    PYTHONPATH=src python -m cellbridge.evaluation.terra_ceiling \
      --config configs/lora/contrastive_k0.yaml \
      --samples TENX200 --output-dir outputs/evaluation/terra_ceiling
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from cellbridge.evaluation.classification import (
    evaluate_classification,
    method_directory,
    resolve_methods,
    write_probe_result,
)
from cellbridge.evaluation.hest import prepare_labeled_hest_features
from cellbridge.evaluation.provenance import stable_signature
from cellbridge.lora.hest import manifest_row, sample_dir
from cellbridge.lora.train import load_config


def _terra_features(cfg, sample_id):
    """Load the frozen teacher embedding and its cell IDs for one section."""

    directory = sample_dir(cfg, sample_id, manifest_row(cfg, sample_id)[1])
    path = directory / f"{sample_id}_terra.npz"
    key = cfg.get("terra_key", "cell_emb")
    with np.load(path, allow_pickle=True) as data:
        features = np.asarray(data[key], dtype=np.float32)
        cell_ids = np.asarray([str(value) for value in data["cell_ids"]])
    return features, cell_ids, path


def probe_sample(cfg, sample_id, output_dir):
    """Run every configured method for one section and write the artifacts."""

    settings = cfg["eval"]
    features, cell_ids, source = _terra_features(cfg, sample_id)
    directory = sample_dir(cfg, sample_id)
    data = prepare_labeled_hest_features(
        features,
        cell_ids,
        directory / f"{sample_id}_granular_cell_labels.csv",
        directory / f"{sample_id}.h5ad",
        settings,
        pca=False,
    )
    rows = []
    for method in resolve_methods(settings):
        signature = stable_signature(
            {
                "arm": "terra_teacher",
                "source": str(source),
                "sample_id": sample_id,
                "label_level": settings.get("label_level", "source"),
                "settings": settings,
                "method": method,
            }
        )
        result = evaluate_classification(
            data.features,
            data.labels,
            data.train_indices,
            data.test_indices,
            method,
            settings,
            row_ids=data.cell_ids,
        )
        metrics = {
            "sample_id": sample_id,
            "arm": "terra_teacher",
            "method": method,
            "label_level": settings.get("label_level", "source"),
            "n_cells_scored": len(data.cell_ids),
            **{
                key: result.metrics[key]
                for key in (
                    "n_features",
                    "n_train",
                    "n_test",
                    "n_classes",
                    "accuracy",
                    "balanced_accuracy",
                    "macro_f1",
                    "weighted_f1",
                )
            },
        }
        predictions = {
            "cell_ids_test": result.predictions["row_ids_test"],
            "y_true": result.predictions["y_true"],
            "y_pred": result.predictions["y_pred"],
            "classes": result.predictions["classes"].astype(str),
        }
        result_dir = method_directory(
            Path(output_dir) / "probes" / sample_id / "terra_teacher", method
        )
        rows.append(write_probe_result(result_dir, metrics, predictions, signature))
        print(
            f"[TERRA/{sample_id}/{method}] macro-F1={metrics['macro_f1']:.4f} "
            f"balanced-acc={metrics['balanced_accuracy']:.4f} "
            f"({metrics['n_classes']} classes, {metrics['n_test']:,} test cells)"
        )
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--samples", required=True, help="comma-separated section IDs")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    for sample_id in args.samples.split(","):
        probe_sample(cfg, sample_id.strip(), args.output_dir)


if __name__ == "__main__":
    main()
