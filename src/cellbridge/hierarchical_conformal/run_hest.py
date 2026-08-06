"""Run patient-disjoint conformal evaluation on annotated HEST breast sections."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from cellbridge.config import load_config
from cellbridge.evaluation.linear_probe import fit_fixed_linear_classifier, reduce_features
from cellbridge.evaluation.storage import write_table
from cellbridge.hierarchical.splits import load_phenotype_split

from .conformal import calibrate_graph, calibrate_split, graph_sets, split_sets
from .hest_data import load_frozen_hest_representation, load_pruned_ontology_hierarchy
from .metrics import per_class_coverage, summarize


def _samples(config):
    split = load_phenotype_split(
        config["data"]["split_manifest"],
        config["data"]["base_split"],
        expected_counts=config["data"].get("expected_counts"),
    )
    return {
        "train": list(split.breast_train),
        "calibration": list(split.breast_val),
        "test": list(split.breast_test),
    }


def _fit_and_predict(data, settings, pca_components, seed):
    features, explained = reduce_features(
        data.features,
        data.train_indices,
        pca_components,
        seed,
    )
    model = fit_fixed_linear_classifier(
        features[data.train_indices],
        data.labels[data.train_indices],
        C=settings.get("C", 1.0),
        class_weight=settings.get("class_weight", "balanced"),
        max_iter=settings.get("max_iter", 2000),
    )
    classes = np.asarray(model.classes_).astype(str)
    calibration = model.predict_proba(features[data.calibration_indices])
    test = model.predict_proba(features[data.test_indices])
    return classes, calibration, test, explained


def _sample_rows(representation, method, sets, labels, sample_ids, classes):
    lookup = {label: column for column, label in enumerate(classes)}
    columns = np.asarray([lookup[str(label)] for label in labels], dtype=int)
    covered = sets[np.arange(len(labels)), columns]
    sizes = sets.sum(axis=1)
    rows = []
    for sample_id in sorted(set(sample_ids)):
        selected = sample_ids == sample_id
        rows.append(
            {
                "representation": representation,
                "method": method,
                "sample_id": sample_id,
                "n_test": int(selected.sum()),
                "coverage": float(covered[selected].mean()),
                "mean_prediction_set_size": float(sizes[selected].mean()),
                "singleton_rate": float(np.mean(sizes[selected] == 1)),
            }
        )
    return rows


def run(config):
    alpha = float(config.get("alpha", 0.1))
    seed = int(config.get("seed", 0))
    pca_components = int(config.get("pca_components", 256))
    samples = _samples(config)
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    manifests = config["data"]["feature_manifests"]
    classifier = config.get("classifier", {})

    summaries, class_rows, sample_rows, calibration_rows = [], [], [], []
    reference_identity = None
    for name, arm in config["representations"].items():
        print(f"[{name}] loading frozen {arm} features")
        data = load_frozen_hest_representation(
            manifests,
            samples,
            config["data"]["labels_dir"],
            arm,
        )
        identity = (data.row_ids, data.labels, data.split)
        if reference_identity is None:
            reference_identity = identity
        elif not all(
            np.array_equal(reference, current)
            for reference, current in zip(reference_identity, identity)
        ):
            raise ValueError("representations are not aligned to identical HEST cells")

        split_counts = {
            "train": len(data.train_indices),
            "calibration": len(data.calibration_indices),
            "test": len(data.test_indices),
        }
        print(f"[{name}] cells {split_counts}; fitting train-only PCA and classifier")
        classes, p_calibration, p_test, explained = _fit_and_predict(
            data,
            classifier,
            pca_components,
            seed,
        )
        hierarchy = load_pruned_ontology_hierarchy(
            config["data"]["hierarchy_nodes"], classes
        )
        hierarchy.validate_classes(classes)
        y_calibration = data.labels[data.calibration_indices]
        y_test = data.labels[data.test_indices]
        test_samples = data.sample_ids[data.test_indices]
        top1 = classes[np.argmax(p_test, axis=1)]

        q_hat = calibrate_split(p_calibration, y_calibration, classes, alpha)
        standard_sets = split_sets(p_test, q_hat)
        graph_calibration = calibrate_graph(
            p_calibration, y_calibration, classes, hierarchy, alpha
        )
        constrained_sets, returned_nodes = graph_sets(
            p_test, classes, hierarchy, graph_calibration.lambda_hat
        )
        for method, sets, nodes in (
            ("split", standard_sets, None),
            ("hierarchy", constrained_sets, returned_nodes),
        ):
            summaries.append(
                summarize(
                    name,
                    method,
                    sets,
                    y_test,
                    top1,
                    classes,
                    hierarchy,
                    1.0 - alpha,
                    nodes,
                )
            )
            class_rows.extend(per_class_coverage(name, method, sets, y_test, classes))
            sample_rows.extend(
                _sample_rows(name, method, sets, y_test, test_samples, classes)
            )

        calibration_rows.extend(
            [
                {
                    "representation": name,
                    "method": "split",
                    "parameter": "q_hat",
                    "value": q_hat,
                    "empirical_risk": float("nan"),
                    "corrected_risk": float("nan"),
                    "pca_explained_variance": explained,
                    **split_counts,
                },
                {
                    "representation": name,
                    "method": "hierarchy",
                    "parameter": "lambda_hat",
                    "value": graph_calibration.lambda_hat,
                    "empirical_risk": graph_calibration.empirical_risk,
                    "corrected_risk": graph_calibration.corrected_risk,
                    "pca_explained_variance": explained,
                    **split_counts,
                },
            ]
        )
        del data, p_calibration, p_test, standard_sets, constrained_sets
        gc.collect()

    summary = pd.DataFrame(summaries)
    write_table(output / "summary.csv", summary.to_dict("records"))
    write_table(output / "per_class_coverage.csv", class_rows)
    write_table(output / "per_sample_coverage.csv", sample_rows)
    write_table(output / "calibration.csv", calibration_rows)
    (output / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    print(summary.to_string(index=False))
    print(f"wrote results to {output}")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    run(load_config(args.config))


if __name__ == "__main__":
    main()
