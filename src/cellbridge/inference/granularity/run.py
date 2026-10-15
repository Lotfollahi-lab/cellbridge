"""NuCLS native super/main/raw resolution sweep without fitting probes.

Run with ``python -m cellbridge.inference.granularity.run --config CONFIG``.
Analysis tables and figures have separate configurable output directories.
"""

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

from cellbridge.config import load_config
from cellbridge.evaluation import evaluate_predictions, save_result
from cellbridge.inference.evaluate.bootstrap import (
    _group_confusions, _metrics_from_confusions,
)

DEFAULT_CONFIG = Path(__file__).resolve().parents[4] / "configs/analysis/granularity.yaml"

LEVELS = ("super", "main", "raw")
METRICS = ("macro_f1", "balanced_accuracy", "accuracy", "weighted_f1")


def validate_hierarchy(table):
    table = table[["raw", "main", "super"]].drop_duplicates()
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
    if (probabilities.ndim != 2 or probabilities.shape[1] != len(classes)
            or len(set(classes)) != len(classes)):
        raise ValueError("invalid probability shape or duplicate classes")
    if (not np.isfinite(probabilities).all() or (probabilities < 0).any()
            or not np.allclose(probabilities.sum(1), 1, atol=1e-5)):
        raise ValueError("invalid probability values")
    if set(classes) - set(mapping):
        raise ValueError("missing class mapping")
    if n_neighbors is not None:
        if n_neighbors < 1:
            raise ValueError("n_neighbors must be positive")
        votes = np.rint(probabilities.astype(np.float64) * n_neighbors).astype(np.int64)
        if (not np.allclose(votes / n_neighbors, probabilities, atol=1e-7, rtol=0)
                or not np.all(votes.sum(1) == n_neighbors)):
            raise ValueError("probabilities are not uniform kNN votes")
        probabilities = votes
    groups = np.array([mapping[label] for label in classes])
    grouped_classes = np.unique(groups)
    grouped = np.column_stack([
        probabilities[:, groups == label].sum(1) for label in grouped_classes
    ])
    if n_neighbors is not None:
        grouped = grouped / n_neighbors
    return grouped, grouped_classes


def load_predictions(path):
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in ("ids", "y_true", "classes", "probabilities")}
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
        tables.append(pd.read_csv(path, usecols=[f"{level}_classification" for level in LEVELS])
                      .rename(columns={f"{level}_classification": level for level in LEVELS})
                      .drop_duplicates())
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
            cases.append({"method": name, "probe": probe, "source": str(source),
                          "key": f"{method}/{probe}"})
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
            support.append({"level": level, "class": label, "n_test": len(rows),
                            "n_slides": rows.slide_id.nunique()})
    pd.DataFrame(support).to_csv(output / "class_support.csv", index=False)
    (output / "cases.json").write_text(json.dumps(cases, indent=2))
    provenance = {
        "config": config, "hierarchy_source": str(raw_dir), "annotation_files": len(tables),
        "n_test": len(groups), "n_test_slides": len(set(groups)),
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[4], text=True).strip(),
        "analysis_config": settings,
        "analysis_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "prediction_sha256": {c["key"]: hashlib.sha256((Path(c["source"]) / "predictions.npz").read_bytes()).hexdigest() for c in cases},
        "aggregation": "sum raw-class probabilities, then argmax; no refitting",
        "knn_aggregation": "recover integer uniform-neighbor votes before grouping to avoid float32 ties",
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2))
    return cases, hierarchy, groups


def evaluate_case(arguments):
    case, hierarchy, groups, output, n_resamples = arguments
    saved = load_predictions(Path(case["source"]) / "predictions.npz")
    original_metrics = pd.read_csv(Path(case["source"]) / "metrics.csv").iloc[0]
    n_neighbors = int(original_metrics["n_neighbors"]) if case["probe"] == "knn" else None
    unique_groups = np.unique(groups)
    # Identical sorted slide order and seed pair every method, probe and level.
    weights = np.random.default_rng(0).multinomial(
        len(unique_groups), np.full(len(unique_groups), 1 / len(unique_groups)), size=n_resamples,
    )
    rows, draws = [], []
    for level in LEVELS:
        mapping = dict(zip(hierarchy.raw, hierarchy[level]))
        probabilities, classes = aggregate(saved["probabilities"], saved["classes"], mapping, n_neighbors=n_neighbors)
        result = evaluate_predictions(
            ids=saved["ids"], y_true=np.array([mapping[label] for label in saved["y_true"]]),
            probabilities=probabilities, classes=classes, min_test_support=20,
            metrics={"method": case["method"], "probe": case["probe"], "level": level,
                     "n_classes": len(classes), "n_test": len(groups)},
        )
        if level == "raw":
            original = pd.read_csv(Path(case["source"]) / "metrics.csv").iloc[0]
            for metric in (*METRICS, "macro_auroc"):
                if not np.isclose(original[metric], result.metrics[metric], atol=1e-7):
                    raise ValueError(f"raw {metric} does not reproduce table: {case['key']}")
        destination = output / "runs" / case["key"] / level
        save_result(destination, result)
        pd.DataFrame(classification_report(result.y_true, result.y_pred, output_dict=True,
                                          zero_division=0)).T.to_csv(destination / "per_class.csv")
        confusions, true_mask, _ = _group_confusions(result, groups)
        values = _metrics_from_confusions(np.einsum("bg,gij->bij", weights, confusions), true_mask)
        for metric in METRICS:
            interval = np.percentile(values[metric], [2.5, 97.5])
            rows.append({**{key: result.metrics[key] for key in ("method", "probe", "level", "n_classes", "n_test")},
                         "metric": metric, "estimate": result.metrics[metric],
                         "bootstrap_se": values[metric].std(ddof=1), "ci_low": interval[0], "ci_high": interval[1]})
        draws.append(pd.DataFrame(values).assign(method=case["method"], probe=case["probe"], level=level,
                                                 bootstrap=np.arange(n_resamples)))
    print(f"Completed {case['key']}", flush=True)
    return pd.DataFrame(rows), pd.concat(draws, ignore_index=True)


def report(output, summary, draws, *, figure_dir, refitted=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir = Path(figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    methods = list(summary["method"].drop_duplicates())
    paired = []
    for probe in ("linear", "knn"):
        for metric in METRICS:
            points = summary.query("probe == @probe and metric == @metric").pivot(index="method", columns="level", values="estimate")
            boot = draws.query("probe == @probe").pivot(index="bootstrap", columns=["method", "level"], values=metric)
            cb_change = boot["CellBridge", "raw"] - boot["CellBridge", "super"]
            for method in methods:
                changes = boot[method, "raw"] - boot[method, "super"]
                comparisons = {"raw_minus_super": (points.loc[method, "raw"] - points.loc[method, "super"], changes)}
                if method != "CellBridge":
                    comparisons["CellBridge_minus_baseline_change"] = (
                        points.loc["CellBridge", "raw"] - points.loc["CellBridge", "super"] - comparisons["raw_minus_super"][0], cb_change - changes)
                    for level in LEVELS:
                        comparisons[f"CellBridge_minus_baseline_{level}"] = (
                            points.loc["CellBridge", level] - points.loc[method, level],
                            boot["CellBridge", level] - boot[method, level])
                for contrast, (estimate, values) in comparisons.items():
                    low, high = np.percentile(values, [2.5, 97.5])
                    paired.append(dict(probe=probe, metric=metric, method=method, contrast=contrast,
                                       estimate=estimate, ci_low=low, ci_high=high))
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))
        for ax, metric, title in zip(axes, METRICS[:3], ("Macro F1", "Balanced accuracy", "Accuracy")):
            for method in methods:
                rows = summary.query("probe == @probe and metric == @metric and method == @method").set_index("level").loc[list(LEVELS)]
                colors = dict(zip(methods, ["#bd2435", "#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b", "#7f7f7f"]))
                color = colors[method]
                line, = ax.plot(range(3), rows.estimate * 100, marker="o", label=method, color=color,
                                linewidth=3 if method == "CellBridge" else 1.5)
                ax.fill_between(range(3), rows.ci_low * 100, rows.ci_high * 100, color=line.get_color(), alpha=.08)
            ax.set_xticks(range(3), [f"{level.title()} ({int(k)})" for level, k in zip(LEVELS, rows.n_classes)])
            ax.set_xlabel("Annotation granularity (number of classes)")
            ax.set_ylabel(f"{title} (%)")
            ax.grid(alpha=.2)
        protocol = "probes fitted at each level" if refitted else "grouped saved probabilities"
        fig.suptitle(f"NuCLS · {'Linear' if probe == 'linear' else 'kNN'} probe · {protocol}")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=7, frameon=False)
        fig.tight_layout(rect=(0, .1, 1, .95))
        for extension in ("png", "pdf"):
            fig.savefig(figure_dir / f"granularity_{probe}.{extension}", dpi=200, bbox_inches="tight")
        plt.close(fig)
    pd.DataFrame(paired).to_csv(output / "paired_differences.csv", index=False)
    text = ["# NuCLS label granularity", "", "Saved raw-class probabilities summed within native groups; no probe refitting.", "",
            "Same 6,349 test cells at all levels; native AMBIGUOUS category retained to reproduce the table.", "",
            "Tables show point estimates; 95% paired slide-bootstrap intervals are in the figures and summary.csv. Macro F1 follows the existing evaluator's union of true/predicted classes; class counts on the x-axis count all output columns.", "",
            "Changing class composition and weighting affects macro F1, so curves alone do not establish training robustness. See paired_differences.csv: positive CellBridge_minus_baseline_change means a smaller CellBridge loss from super to raw. Intervals are exploratory, unadjusted for multiple comparisons.", ""]
    if refitted:
        text.extend(["Direct uniform kNN is checked against the regrouped analysis, which reconstructs integer neighbor votes to prevent floating-point rounding from changing exact ties.", ""])
    else:
        text.extend(["For uniform kNN, integer neighbor votes are reconstructed before aggregation to prevent floating-point rounding from changing exact ties.", ""])
    if refitted:
        text[2] = "Main/super probes fitted directly to grouped labels on the existing splits and embeddings; raw level reuses the original fitted probe. Linear C selected on validation at each level; final fit uses train+validation."
    text.extend(["Intervals condition on the fitted models and use paired slide resamples (seed 0). Other_nucleus has only seven test cells on one slide; bootstrap metrics retain the original class universe and score absent classes as zero. Two raw output classes have no true test observations. Macro AUROC is saved as a point estimate only in each run metrics.csv.", ""])
    if refitted and (output / "audit.csv").exists():
        audit = pd.read_csv(output / "audit.csv")
        warned = audit.loc[audit.convergence_warning.eq(True)]
        unknown = audit.loc[(audit.level != "raw") & audit.convergence_warning.isna()]
        text.extend([f"Convergence audit: {len(warned)} coarse tasks recorded an iteration-limit warning; {len(unknown)} recovered coarse tasks have unavailable warning history. The original iteration budget was retained. See audit.csv for method and level details.", ""])
    for probe in ("linear", "knn"):
        text.extend([f"## {probe}", "", "| Method | Super | Main | Raw | Raw − super (pp) |", "|---|---:|---:|---:|---:|"])
        for method in methods:
            rows = summary.query("probe == @probe and method == @method and metric == 'macro_f1'").set_index("level")
            values = [f"{rows.loc[level, 'estimate'] * 100:.2f}" for level in LEVELS]
            change = (rows.loc['raw', 'estimate'] - rows.loc['super', 'estimate']) * 100
            text.append(f"| {method} | {' | '.join(values)} | {change:+.2f} |")
        text.append("")
    (output / "results.md").write_text("\n".join(text))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--figure-dir", type=Path)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--n-resamples", type=int)
    args = parser.parse_args()
    settings = load_config(args.config)
    args.output_dir = args.output_dir or Path(settings["output_dir"])
    args.figure_dir = args.figure_dir or Path(settings["figure_dir"])
    args.workers = settings["workers"] if args.workers is None else args.workers
    args.n_resamples = settings["n_resamples"] if args.n_resamples is None else args.n_resamples
    if args.workers < 1 or args.n_resamples < 2:
        parser.error("workers must be positive and n-resamples at least two")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir / "cases.json").exists():
        raise FileExistsError("use a new output directory to preserve earlier runs")
    cases, hierarchy, groups = prepare(args.output_dir, settings)
    provenance_path = args.output_dir / "provenance.json"
    provenance = json.loads(provenance_path.read_text())
    provenance.update(n_resamples=args.n_resamples, seed=0, workers=args.workers,
                      figure_dir=str(args.figure_dir))
    provenance_path.write_text(json.dumps(provenance, indent=2))
    print(f"Prepared {len(cases)} cases, {len(groups)} cells, {len(set(groups))} slides", flush=True)
    arguments = [(case, hierarchy, groups, args.output_dir, args.n_resamples) for case in cases]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(evaluate_case, arguments))
    summary = pd.concat([result[0] for result in results], ignore_index=True)
    draws = pd.concat([result[1] for result in results], ignore_index=True)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    draws.to_csv(args.output_dir / "bootstrap_metrics.csv", index=False)
    report(args.output_dir, summary, draws, figure_dir=args.figure_dir)
    print(f"Results: {args.output_dir / 'results.md'}", flush=True)


if __name__ == "__main__":
    main()
