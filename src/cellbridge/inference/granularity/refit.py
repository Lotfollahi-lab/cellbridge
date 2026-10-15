"""Fit NuCLS probes directly at native annotation levels using cached embeddings."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import warnings

import numpy as np
import pandas as pd
from sklearn import config_context
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import classification_report

from cellbridge.config import load_config
from cellbridge.evaluation import evaluate_predictions, save_result
from cellbridge.inference.embed.artifacts import load_manifest
from cellbridge.inference.evaluate import ood, probe as probe_api
from cellbridge.inference.evaluate.bootstrap import _group_confusions, _metrics_from_confusions
from cellbridge.inference.granularity.run import DEFAULT_CONFIG, LEVELS, METRICS, load_predictions, report
from cellbridge.inference.run import _align_metadata


def digest(values):
    return hashlib.sha256(json.dumps(np.asarray(values).astype(str).tolist()).encode()).hexdigest()


def score_result(result, groups, weights, case, level, destination):
    save_result(destination, result)
    pd.DataFrame(classification_report(result.y_true, result.y_pred, output_dict=True,
                                      zero_division=0)).T.to_csv(destination / "per_class.csv")
    confusions, true_mask, _ = _group_confusions(result, groups)
    values = _metrics_from_confusions(np.einsum("bg,gij->bij", weights, confusions), true_mask)
    rows = []
    for metric in METRICS:
        low, high = np.percentile(values[metric], [2.5, 97.5])
        rows.append(dict(method=case["method"], probe=case["probe"], level=level,
                         n_classes=len(result.classes), n_test=len(groups), metric=metric,
                         estimate=result.metrics[metric], bootstrap_se=values[metric].std(ddof=1),
                         ci_low=low, ci_high=high))
    draws = pd.DataFrame(values).assign(method=case["method"], probe=case["probe"], level=level,
                                        bootstrap=np.arange(len(weights)))
    return pd.DataFrame(rows), draws


def evaluate_method(arguments):
    cases, config, hierarchy, population, reuse, output, n_resamples = arguments
    case = cases[0]
    source = Path(case["source"])
    arm = source.parents[1].name
    manifest = source.parents[3] / "embeddings" / arm / "manifest.csv"
    loaded = load_manifest(manifest)
    metadata = _align_metadata(config, loaded)
    settings = config["probe"]
    task = ood.prepare_task(loaded.embeddings, metadata, task="all", arm=arm,
                            pca_components=settings.get("pca_components", 256), seed=settings.get("seed", 0))
    test_order = np.argsort(task.ids[task.test].astype(str))
    test = task.test[test_order]
    saved = load_predictions(source / "predictions.npz")
    if not np.array_equal(task.ids[test].astype(str), saved["ids"]):
        raise ValueError(f"reconstructed test IDs differ: {case['method']}")
    if not np.array_equal(task.labels[test].astype(str), saved["y_true"]):
        raise ValueError(f"reconstructed labels differ: {case['method']}")
    if not np.array_equal(population.index.astype(str), saved["ids"]):
        raise ValueError("test population order differs")
    groups = population.slide_id.astype(str).to_numpy()
    n_groups = len(np.unique(groups))
    weights = np.random.default_rng(0).multinomial(n_groups, np.full(n_groups, 1 / n_groups), size=n_resamples)
    identity = {name: digest(np.sort(task.ids[indices].astype(str)))
                for name, indices in (("train", task.train), ("validation", task.validation), ("test", task.test))}
    summaries, draws, audits = [], [], []
    audit_path = output / "runs" / case["key"].split("/")[0] / "audit.csv"
    prior_audit = pd.read_csv(audit_path) if audit_path.exists() else pd.DataFrame()
    for case in cases:
        for level in LEVELS:
            print(f"Evaluating {case['method']}/{case['probe']}/{level}", flush=True)
            mapping = dict(zip(hierarchy.raw, hierarchy[level]))
            original = load_predictions(Path(case["source"]) / "predictions.npz")
            if not np.array_equal(original["ids"], saved["ids"]) or not np.array_equal(original["y_true"], saved["y_true"]):
                raise ValueError("raw probe predictions are not paired")
            convergence, details = False, {}
            destination = output / "runs" / case["key"] / level
            resumed = level != "raw" and (destination / "predictions.npz").exists()
            if level == "raw":
                convergence = None
                classes, probabilities = original["classes"], original["probabilities"]
                raw_metrics = pd.read_csv(Path(case["source"]) / "metrics.csv").iloc[0]
                details = {key: raw_metrics[key] for key in ("C", "selection_macro_f1", "n_neighbors", "n_reference") if key in raw_metrics}
            elif resumed:
                previous = load_predictions(destination / "predictions.npz")
                if not np.array_equal(previous["ids"], saved["ids"]):
                    raise ValueError("completed result IDs differ")
                if not np.array_equal(previous["y_true"], np.array([mapping[label] for label in task.labels[test]])):
                    raise ValueError("completed result labels differ")
                classes, probabilities = previous["classes"], previous["probabilities"]
                previous_metrics = pd.read_csv(destination / "metrics.csv").iloc[0]
                details = {key: previous_metrics[key] for key in ("C", "selection_macro_f1", "n_neighbors", "n_reference") if key in previous_metrics}
                convergence = None
                if not prior_audit.empty:
                    match = prior_audit[(prior_audit.probe == case["probe"]) & (prior_audit.level == level)]
                    if len(match):
                        convergence = match.iloc[0].convergence_warning
            else:
                grouped = replace(task, labels=np.array([mapping[label] for label in task.labels]))
                options = dict(settings[case["probe"]], seed=settings.get("seed", 0))
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always", ConvergenceWarning)
                    model, details = probe_api.fit(grouped, case["probe"], options)
                convergence = any(issubclass(w.category, ConvergenceWarning) for w in caught)
                if case["probe"] == "knn":
                    model.set_params(n_jobs=1)
                classes = model.classes_.astype(str)
                with config_context(working_memory=128):
                    probabilities = np.concatenate([model.predict_proba(grouped.features[test[start:start + 256]])
                                                    for start in range(0, len(test), 256)])
            result = evaluate_predictions(ids=task.ids[test].astype(str),
                y_true=np.array([mapping[label] for label in task.labels[test]]), probabilities=probabilities,
                classes=classes, min_test_support=20,
                metrics=dict(method=case["method"], probe=case["probe"], level=level,
                             n_classes=len(classes), n_test=len(test), n_train=len(task.train),
                             n_validation=len(task.validation), n_features=task.features.shape[1], **details))
            if level == "raw":
                original_metrics = pd.read_csv(Path(case["source"]) / "metrics.csv").iloc[0]
                for metric in (*METRICS, "macro_auroc"):
                    if not np.isclose(original_metrics[metric], result.metrics[metric], atol=1e-7):
                        raise ValueError(f"raw {metric} differs from saved table")
            reused = load_predictions(reuse / "runs" / case["key"] / level / "predictions.npz")
            if not np.array_equal(result.ids.astype(str), reused["ids"]):
                raise ValueError("refitted and reused test IDs differ")
            if not np.array_equal(result.y_true.astype(str), reused["y_true"]):
                raise ValueError("refitted and reused true labels differ")
            if not np.array_equal(result.classes.astype(str), reused["classes"]):
                raise ValueError("refitted and reused class order differs")
            error = float(np.abs(result.probabilities - reused["probabilities"]).max())
            # A uniform kNN vote commutes exactly with grouping class labels.
            if case["probe"] == "knn" and error > 1e-6:
                print(f"Numerical neighbor discrepancy: {case['method']}/{level}: {error}", flush=True)
            destination = output / "runs" / case["key"] / level
            summary, bootstrap = score_result(result, groups, weights, case, level, destination)
            summaries.append(summary)
            draws.append(bootstrap)
            audits.append(dict(method=case["method"], probe=case["probe"], level=level,
                manifest=str(manifest), source=case["source"], reused_raw=level == "raw", resumed_completed_fit=resumed,
                n_train=len(task.train), n_validation=len(task.validation), n_test=len(test),
                convergence_warning=convergence, max_probability_difference_from_reuse=error,
                changed_argmax_from_reuse=int(np.sum(result.probabilities.argmax(1) != reused["probabilities"].argmax(1))),
                **{f"{key}_ids_sha256": value for key, value in identity.items()}, **details))
            pd.DataFrame(audits).to_csv(audit_path, index=False)
    audit = pd.DataFrame(audits)
    method_dir = output / "runs" / case["key"].split("/")[0]
    audit.to_csv(method_dir / "audit.csv", index=False)
    print(f"Completed {case['method']}", flush=True)
    return pd.concat(summaries, ignore_index=True), pd.concat(draws, ignore_index=True), audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--reuse-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--n-resamples", type=int)
    parser.add_argument("--resume", action="store_true", help="reuse completed fits and rebuild the report")
    args = parser.parse_args()
    settings = load_config(args.config)
    args.reuse_dir = args.reuse_dir or Path(settings["output_dir"])
    args.output_dir = args.output_dir or Path(settings["refit_output_dir"])
    args.workers = settings["workers"] if args.workers is None else args.workers
    args.n_resamples = settings["n_resamples"] if args.n_resamples is None else args.n_resamples
    if args.workers < 1 or args.n_resamples < 2:
        parser.error("workers must be positive and n-resamples at least two")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir / "cases.json").exists() and not args.resume:
        raise FileExistsError("use a new output directory or --resume")
    cases = json.loads((args.reuse_dir / "cases.json").read_text())
    provenance = json.loads((args.reuse_dir / "provenance.json").read_text())
    hierarchy = pd.read_csv(args.reuse_dir / "hierarchy.csv")
    population = pd.read_csv(args.reuse_dir / "test_population.csv", dtype={"row_id": str}).set_index("row_id")
    for filename in ("cases.json", "hierarchy.csv", "test_population.csv", "class_support.csv"):
        shutil.copyfile(args.reuse_dir / filename, args.output_dir / filename)
    if args.resume and (args.output_dir / "provenance.json").exists():
        provenance = json.loads((args.output_dir / "provenance.json").read_text())
    if not args.resume:
        provenance["analysis_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    provenance.update(final_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      reuse_dir=str(args.reuse_dir), analysis="Direct super/main probe fitting; raw predictions reused",
                      n_resamples=args.n_resamples, bootstrap_seed=0, workers=args.workers,
                      aggregation="direct grouped-label probe fitting; raw predictions reused")
    (args.output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2))
    methods = list(dict.fromkeys(case["method"] for case in cases))
    arguments = [([case for case in cases if case["method"] == method], provenance["config"], hierarchy,
                  population, args.reuse_dir, args.output_dir, args.n_resamples) for method in methods]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(evaluate_method, arguments))
    summary, draws, audit = [pd.concat([result[i] for result in results], ignore_index=True) for i in range(3)]
    for split in ("train", "validation", "test"):
        if audit[f"{split}_ids_sha256"].nunique() != 1:
            raise ValueError(f"unpaired {split} identities across methods")
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    draws.to_csv(args.output_dir / "bootstrap_metrics.csv", index=False)
    audit.to_csv(args.output_dir / "audit.csv", index=False)
    provenance["convergence_warning_unknown_cases"] = int(audit.convergence_warning.isna().sum())
    provenance["convergence_warning_unknown_refitted_cases"] = int((audit.convergence_warning.isna() & ~audit.reused_raw).sum())
    (args.output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2))
    report(args.output_dir, summary, draws, refitted=True)
    print(f"Completed: {args.output_dir / 'results.md'}", flush=True)


if __name__ == "__main__":
    main()
