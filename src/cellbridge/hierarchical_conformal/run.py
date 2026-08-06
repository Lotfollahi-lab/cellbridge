"""Run frozen UNI versus CellBridge hierarchical conformal evaluation."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from cellbridge.config import load_config
from cellbridge.evaluation.linear_probe import fit_fixed_linear_classifier
from cellbridge.evaluation.ood import prepare_ood_task
from cellbridge.evaluation.storage import write_table

from .conformal import calibrate_graph, calibrate_split, graph_sets, split_sets
from .data import load_frozen_representation
from .hierarchy import Hierarchy
from .metrics import per_class_coverage, summarize


def _hierarchy(settings):
    hierarchy = Hierarchy.from_edges(settings["root"], settings["edges"])
    if not hierarchy.is_tree:
        raise ValueError("returned-ancestor metrics require a task hierarchy tree")
    leaf_depths = {hierarchy.depths[leaf] for leaf in hierarchy.leaves}
    if len(leaf_depths) != 1:
        raise ValueError("all task leaves must have the same depth")
    return hierarchy


def _fit_and_predict(data, settings):
    model = fit_fixed_linear_classifier(
        data.features[data.train_indices],
        data.labels[data.train_indices],
        C=settings.get("C", 1.0),
        class_weight=settings.get("class_weight", "balanced"),
        max_iter=settings.get("max_iter", 2000),
    )
    classes = np.asarray(model.classes_).astype(str)
    calibration = model.predict_proba(data.features[data.val_indices])
    test = model.predict_proba(data.features[data.test_indices])
    return classes, calibration, test


def _identity(table):
    return table.loc[:, ["row_id", "label", "split"]].astype(str).to_numpy()


def run(config):
    alpha = float(config.get("alpha", 0.1))
    hierarchy = _hierarchy(config["hierarchy"])
    classifier = config.get("classifier", {})
    pca_components = int(config.get("pca_components", 256))
    prep_settings = {
        "uni_pca_components": pca_components,
        "seed": int(config.get("seed", 0)),
    }
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)

    summaries, class_rows, calibration_rows = [], [], []
    reference_identity = None
    for name, arm in config["representations"].items():
        print(f"[{name}] loading frozen {arm} features")
        features, table = load_frozen_representation(
            config["data"]["feature_manifest"],
            arm,
            config["data"]["official_split"],
            config["data"].get("exclude_labels", ()),
        )
        current_identity = _identity(table)
        if reference_identity is None:
            reference_identity = current_identity
        elif not np.array_equal(reference_identity, current_identity):
            raise ValueError("representations are not aligned to identical cells")

        data = prepare_ood_task(
            features,
            table,
            "all",
            prep_settings,
            pca=bool(pca_components),
        )
        split_counts = {
            "train": len(data.train_indices),
            "calibration": len(data.val_indices),
            "test": len(data.test_indices),
        }
        print(f"[{name}] cells {split_counts}; fitting fixed classifier")
        classes, p_calibration, p_test = _fit_and_predict(data, classifier)
        hierarchy.validate_classes(classes)
        y_calibration = data.labels[data.val_indices].astype(str)
        y_test = data.labels[data.test_indices].astype(str)
        top1 = classes[np.argmax(p_test, axis=1)]

        q_hat = calibrate_split(
            p_calibration,
            y_calibration,
            classes,
            alpha,
        )
        standard_sets = split_sets(p_test, q_hat)
        graph_calibration = calibrate_graph(
            p_calibration,
            y_calibration,
            classes,
            hierarchy,
            alpha,
        )
        constrained_sets, returned_nodes = graph_sets(
            p_test,
            classes,
            hierarchy,
            graph_calibration.lambda_hat,
        )

        methods = {
            "split": (standard_sets, None),
            "hierarchy": (constrained_sets, returned_nodes),
        }
        for method, (sets, nodes) in methods.items():
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

        calibration_rows.extend(
            [
                {
                    "representation": name,
                    "method": "split",
                    "parameter": "q_hat",
                    "value": q_hat,
                    "empirical_risk": float("nan"),
                    "corrected_risk": float("nan"),
                    **split_counts,
                },
                {
                    "representation": name,
                    "method": "hierarchy",
                    "parameter": "lambda_hat",
                    "value": graph_calibration.lambda_hat,
                    "empirical_risk": graph_calibration.empirical_risk,
                    "corrected_risk": graph_calibration.corrected_risk,
                    **split_counts,
                },
            ]
        )
        del features, data, p_calibration, p_test
        gc.collect()

    summary = pd.DataFrame(summaries)
    write_table(output / "summary.csv", summary.to_dict("records"))
    write_table(output / "per_class_coverage.csv", class_rows)
    write_table(output / "calibration.csv", calibration_rows)
    (output / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False)
    )
    shown = [
        "representation",
        "method",
        "marginal_coverage",
        "mean_prediction_set_size",
        "leaf_prediction_rate",
        "mean_returned_ancestor_depth",
        "root_inconclusive_rate",
        "leaf_accuracy",
        "leaf_macro_f1",
    ]
    print(summary[shown].to_string(index=False))
    print(f"wrote results to {output}")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    run(load_config(args.config))


if __name__ == "__main__":
    main()
