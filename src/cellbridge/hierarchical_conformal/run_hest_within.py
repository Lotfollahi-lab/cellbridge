"""Run matched within-slide conformal evaluation on annotated HEST sections."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from cellbridge.config import load_config
from cellbridge.evaluation.linear_probe import fit_linear_classifier, reduce_features
from cellbridge.evaluation.storage import write_table

from .conformal import calibrate_graph, calibrate_split, graph_sets, split_sets
from .hest_data import load_pruned_ontology_hierarchy
from .metrics import per_class_coverage, summarize
from .within_slide_data import load_within_slide_representation


def _fit_and_predict(data, settings, pca_components, seed):
    features, explained = reduce_features(
        data.features,
        data.train_indices,
        pca_components,
        seed,
    )
    model = fit_linear_classifier(
        features[data.train_indices],
        data.labels[data.train_indices],
        settings.get("Cs", [0.1, 1.0, 10.0]),
        class_weight=settings.get("class_weight", "balanced"),
        max_iter=settings.get("max_iter", 2000),
    )
    classes = np.asarray(model.classes_).astype(str)
    calibration = model.predict_proba(features[data.calibration_indices])
    test = model.predict_proba(features[data.test_indices])
    return classes, calibration, test, explained


def _aggregate(summary):
    rows = []
    metrics = [
        "marginal_coverage",
        "mean_prediction_set_size",
        "singleton_rate",
        "leaf_prediction_rate",
        "mean_returned_ancestor_depth",
        "root_inconclusive_rate",
        "leaf_accuracy",
        "leaf_macro_f1",
    ]
    for (representation, method), group in summary.groupby(
        ["representation", "method"], sort=False
    ):
        row = {
            "representation": representation,
            "method": method,
            "target_coverage": float(group["target_coverage"].iloc[0]),
            "n_samples": len(group),
            "n_test": int(group["n_test"].sum()),
        }
        for metric in metrics:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std())
        rows.append(row)
    return pd.DataFrame(rows)


def run(config):
    alpha = float(config.get("alpha", 0.1))
    seed = int(config.get("seed", 0))
    pca_components = int(config.get("pca_components", 256))
    data_settings = config["data"]
    split_settings = config.get("split", {})
    classifier = config.get("classifier", {})
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)

    summaries, class_rows, calibration_rows = [], [], []
    for sample_id in data_settings["samples"]:
        reference_identity = None
        for representation, arm in config["representations"].items():
            print(f"[{sample_id}/{representation}] loading frozen {arm} features")
            data = load_within_slide_representation(
                data_settings["feature_manifest"],
                data_settings["labels_dir"],
                data_settings["data_root"],
                sample_id,
                arm,
                min_class_count=data_settings.get("min_class_count", 25),
                test_fraction=split_settings.get("test_fraction", 0.2),
                calibration_fraction_within_fit=split_settings.get(
                    "calibration_fraction_within_fit", 0.2
                ),
                split_axis=split_settings.get("axis", "y"),
            )
            identity = (data.cell_ids, data.labels)
            if reference_identity is None:
                reference_identity = identity
            elif not all(
                np.array_equal(reference, current)
                for reference, current in zip(reference_identity, identity)
            ):
                raise ValueError(
                    f"{sample_id}: representations are not aligned to identical cells"
                )

            classes, p_calibration, p_test, explained = _fit_and_predict(
                data, classifier, pca_components, seed
            )
            hierarchy = load_pruned_ontology_hierarchy(
                data_settings["hierarchy_nodes"], classes
            )
            hierarchy.validate_classes(classes)
            y_calibration = data.labels[data.calibration_indices]
            y_test = data.labels[data.test_indices]
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
                row = summarize(
                    representation,
                    method,
                    sets,
                    y_test,
                    top1,
                    classes,
                    hierarchy,
                    1.0 - alpha,
                    nodes,
                )
                row["sample_id"] = sample_id
                summaries.append(row)
                for class_row in per_class_coverage(
                    representation, method, sets, y_test, classes
                ):
                    class_rows.append({"sample_id": sample_id, **class_row})

            counts = {
                "train": len(data.train_indices),
                "calibration": len(data.calibration_indices),
                "test": len(data.test_indices),
            }
            calibration_rows.extend(
                [
                    {
                        "sample_id": sample_id,
                        "representation": representation,
                        "method": "split",
                        "parameter": "q_hat",
                        "value": q_hat,
                        "empirical_risk": float("nan"),
                        "corrected_risk": float("nan"),
                        "pca_explained_variance": explained,
                        **counts,
                    },
                    {
                        "sample_id": sample_id,
                        "representation": representation,
                        "method": "hierarchy",
                        "parameter": "lambda_hat",
                        "value": graph_calibration.lambda_hat,
                        "empirical_risk": graph_calibration.empirical_risk,
                        "corrected_risk": graph_calibration.corrected_risk,
                        "pca_explained_variance": explained,
                        **counts,
                    },
                ]
            )
            del data, p_calibration, p_test, standard_sets, constrained_sets
            gc.collect()

    summary = pd.DataFrame(summaries)
    columns = ["sample_id", *[column for column in summary if column != "sample_id"]]
    summary = summary.loc[:, columns]
    aggregate = _aggregate(summary)
    write_table(output / "summary_per_sample.csv", summary.to_dict("records"))
    write_table(output / "summary.csv", aggregate.to_dict("records"))
    write_table(output / "per_class_coverage.csv", class_rows)
    write_table(output / "calibration.csv", calibration_rows)
    (output / "resolved_config.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False)
    )
    print(aggregate.to_string(index=False))
    print(f"wrote results to {output}")
    return aggregate


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args(argv)
    run(load_config(args.config))


if __name__ == "__main__":
    main()
