"""NuCLS native super/main/raw resolution sweep without fitting probes.

Run with ``python -m cellbridge.inference.granularity.run --config CONFIG``.
Save metric summaries, bootstrap samples and per-level predictions.
"""

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from cellbridge.config import load_config
from cellbridge.evaluation import evaluate_predictions
from cellbridge.inference.granularity.common import (
    DEFAULT_CONFIG,
    LEVELS,
    METRICS,
    bootstrap_weights,
    score_result,
    source_revision,
)


def validate_hierarchy(table):
    table = table[["raw", "main", "super"]].drop_duplicates()
    if table.empty:
        raise ValueError("hierarchy is empty")
    if table.isna().any().any() or (table.astype(str) == "").any().any():
        raise ValueError("empty hierarchy label")
    if table.groupby("raw").size().max() != 1:
        raise ValueError("raw labels have conflicting parents")
    if table.groupby("main")["super"].nunique().max() != 1:
        raise ValueError("main labels have conflicting super parents")
    return table.sort_values("raw").reset_index(drop=True)


def aggregate(probabilities, classes, mapping, *, n_neighbors=None):
    probabilities = np.asarray(probabilities)
    classes = np.asarray(classes).astype(str)
    if (
        probabilities.ndim != 2
        or probabilities.shape[1] != len(classes)
        or len(set(classes)) != len(classes)
    ):
        raise ValueError("invalid probability shape or duplicate classes")
    if (
        not np.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or not np.allclose(probabilities.sum(1), 1, atol=1e-5)
    ):
        raise ValueError("invalid probability values")
    if set(classes) - set(mapping):
        raise ValueError("missing class mapping")
    if n_neighbors is not None:
        if n_neighbors < 1:
            raise ValueError("n_neighbors must be positive")
        votes = np.rint(probabilities.astype(np.float64) * n_neighbors).astype(np.int64)
        if not np.allclose(
            votes / n_neighbors, probabilities, atol=1e-7, rtol=0
        ) or not np.all(votes.sum(1) == n_neighbors):
            raise ValueError("probabilities are not uniform kNN votes")
        probabilities = votes
    groups = np.array([mapping[label] for label in classes])
    grouped_classes = np.unique(groups)
    grouped = np.column_stack(
        [probabilities[:, groups == label].sum(1) for label in grouped_classes]
    )
    if n_neighbors is not None:
        grouped = grouped / n_neighbors
    return grouped, grouped_classes


def load_predictions(path):
    with np.load(path, allow_pickle=False) as archive:
        data = {
            key: archive[key] for key in ("ids", "y_true", "classes", "probabilities")
        }
    for key in ("ids", "y_true", "classes"):
        data[key] = data[key].astype(str)
    if len(set(data["ids"])) != len(data["ids"]):
        raise ValueError(f"duplicate prediction IDs: {path}")
    order = np.argsort(data["ids"])
    for key in ("ids", "y_true", "probabilities"):
        data[key] = data[key][order]
    return data


def prepare(output, settings=None):
    settings = load_config(DEFAULT_CONFIG) if settings is None else settings
    config = load_config(settings["dataset_config"])
    metadata = pd.read_csv(config["data"]["metadata"], dtype={"row_id": str})
    raw_dir = Path(settings["hierarchy_dir"])
    tables = []
    for path in sorted(raw_dir.glob("*.csv")):
        tables.append(
            pd.read_csv(path, usecols=[f"{level}_classification" for level in LEVELS])
            .rename(columns={f"{level}_classification": level for level in LEVELS})
            .drop_duplicates()
        )
    if not tables:
        raise FileNotFoundError(f"No NuCLS annotation CSVs in {raw_dir}")
    hierarchy = validate_hierarchy(pd.concat(tables, ignore_index=True))
    cases, reference = [], None
    for name, (method, version, arm) in settings["methods"].items():
        root = Path(settings["inference_root"]) / method / version
        for probe in ("linear", "knn"):
            source = root / "nucls/evaluation" / arm / "all" / probe
            saved = load_predictions(source / "predictions.npz")
            if reference is None:
                reference = saved
            for key in ("ids", "y_true", "classes"):
                if not np.array_equal(saved[key], reference[key]):
                    raise ValueError(f"unpaired {key}: {source}")
            cases.append(
                {
                    "method": name,
                    "probe": probe,
                    "source": str(source),
                    "key": f"{method}/{probe}",
                }
            )
    metadata = metadata.set_index("row_id").loc[reference["ids"]]
    if not np.array_equal(metadata["label"].to_numpy(), reference["y_true"]):
        raise ValueError("saved labels disagree with raw metadata")
    if set(metadata["split"]) != {"test"}:
        raise ValueError("predictions include non-test cells")
    # Match official split and verify slide isolation across the complete metadata.
    complete = pd.read_csv(config["data"]["metadata"])
    from cellbridge.inference.evaluate.ood import validate_split

    validate_split(complete, config["data"]["split"])
    if (complete.groupby("slide_id")["split"].nunique() > 1).any():
        raise ValueError("slides overlap between splits")
    groups = metadata["slide_id"].astype(str).to_numpy()
    if len(set(groups)) < 2:
        raise ValueError("need multiple independent test slides")
    hierarchy = hierarchy[hierarchy.raw.isin(reference["classes"])]
    if set(hierarchy.raw) != set(reference["classes"]):
        raise ValueError("native hierarchy missing prediction classes")
    hierarchy.to_csv(output / "hierarchy.csv", index=False)
    metadata[["label", "slide_id", "split"]].to_csv(output / "test_population.csv")
    support = []
    for level in LEVELS:
        labels = metadata.label.map(dict(zip(hierarchy.raw, hierarchy[level])))
        for label in sorted(set(hierarchy[level])):
            rows = metadata.loc[labels.to_numpy() == label]
            support.append(
                {
                    "level": level,
                    "class": label,
                    "n_test": len(rows),
                    "n_slides": rows.slide_id.nunique(),
                }
            )
    pd.DataFrame(support).to_csv(output / "class_support.csv", index=False)
    (output / "cases.json").write_text(json.dumps(cases, indent=2))
    provenance = {
        "config": config,
        "hierarchy_source": str(raw_dir),
        "annotation_files": len(tables),
        "n_test": len(groups),
        "n_test_slides": len(set(groups)),
        "source_revision": source_revision(),
        "analysis_config": settings,
        "analysis_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "prediction_sha256": {
            c["key"]: hashlib.sha256(
                (Path(c["source"]) / "predictions.npz").read_bytes()
            ).hexdigest()
            for c in cases
        },
        "aggregation": "sum raw-class probabilities, then argmax; no refitting",
        "knn_aggregation": "recover integer uniform-neighbor votes before grouping to avoid float32 ties",
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2))
    return cases, hierarchy, groups


def evaluate_case(arguments):
    case, hierarchy, groups, output, n_resamples = arguments
    saved = load_predictions(Path(case["source"]) / "predictions.npz")
    original_metrics = pd.read_csv(Path(case["source"]) / "metrics.csv").iloc[0]
    n_neighbors = (
        int(original_metrics["n_neighbors"]) if case["probe"] == "knn" else None
    )
    weights = bootstrap_weights(groups, n_resamples)
    rows, draws = [], []
    for level in LEVELS:
        mapping = dict(zip(hierarchy.raw, hierarchy[level]))
        probabilities, classes = aggregate(
            saved["probabilities"], saved["classes"], mapping, n_neighbors=n_neighbors
        )
        result = evaluate_predictions(
            ids=saved["ids"],
            y_true=np.array([mapping[label] for label in saved["y_true"]]),
            probabilities=probabilities,
            classes=classes,
            min_test_support=20,
            metrics={
                "method": case["method"],
                "probe": case["probe"],
                "level": level,
                "n_classes": len(classes),
                "n_test": len(groups),
            },
        )
        if level == "raw":
            original = pd.read_csv(Path(case["source"]) / "metrics.csv").iloc[0]
            for metric in (*METRICS, "macro_auroc"):
                if not np.isclose(original[metric], result.metrics[metric], atol=1e-7):
                    raise ValueError(
                        f"raw {metric} does not reproduce table: {case['key']}"
                    )
        destination = output / "runs" / case["key"] / level
        summary, bootstrap = score_result(
            result, groups, weights, case, level, destination
        )
        rows.append(summary)
        draws.append(bootstrap)
    print(f"Completed {case['key']}", flush=True)
    return pd.concat(rows, ignore_index=True), pd.concat(draws, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--n-resamples", type=int)
    args = parser.parse_args()
    settings = load_config(args.config)
    args.output_dir = args.output_dir or Path(settings["output_dir"])
    args.workers = settings["workers"] if args.workers is None else args.workers
    args.n_resamples = (
        settings["n_resamples"] if args.n_resamples is None else args.n_resamples
    )
    if args.workers < 1 or args.n_resamples < 2:
        parser.error("workers must be positive and n-resamples at least two")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir / "cases.json").exists():
        raise FileExistsError("use a new output directory to preserve earlier runs")
    cases, hierarchy, groups = prepare(args.output_dir, settings)
    provenance_path = args.output_dir / "provenance.json"
    provenance = json.loads(provenance_path.read_text())
    provenance.update(
        n_resamples=args.n_resamples,
        seed=0,
        workers=args.workers,
    )
    provenance_path.write_text(json.dumps(provenance, indent=2))
    print(
        f"Prepared {len(cases)} cases, {len(groups)} cells, {len(set(groups))} slides",
        flush=True,
    )
    arguments = [
        (case, hierarchy, groups, args.output_dir, args.n_resamples) for case in cases
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(evaluate_case, arguments))
    summary = pd.concat([result[0] for result in results], ignore_index=True)
    draws = pd.concat([result[1] for result in results], ignore_index=True)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    draws.to_csv(args.output_dir / "bootstrap_metrics.csv", index=False)
    print(f"Results: {args.output_dir / 'summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
