"""Reproduce existing conformal probes, then audit their probability calibration.

python -m cellbridge.inference.calibration.run prepare --output-dir PATH
python -m cellbridge.inference.calibration.run evaluate --output-dir PATH --index 0
python -m cellbridge.inference.calibration.run report --output-dir PATH
"""

import argparse
import fcntl
import gc
import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning

from cellbridge.config import _defaults, config_root, load_config
from cellbridge.inference.calibration.metrics import fit_temperature, log_probabilities, statistics
from cellbridge.inference.conformal import conformal_ood
from cellbridge.inference.embed import sources
from cellbridge.inference.embed.artifacts import load_manifest, load_manifest_sample
from cellbridge.inference.evaluate import bootstrap, hest
from cellbridge.inference.evaluate import probe as probe_api
from cellbridge.inference.run import _align_metadata
from cellbridge.ontology import load_mapping

ROOT = config_root()
OUTPUTS_ROOT = Path(os.environ.get("CELLBRIDGE_OUTPUTS_ROOT", _defaults()["CELLBRIDGE_OUTPUTS_ROOT"]))
METHODS = {
    "CellBridge": ("cellbridge", "uni_terra_four_layers_adapted_all_cells", "lora_adapted"),
    "UNI2-h": ("uni", "all", "uni"),
    "Virchow2": ("virchow2", "all", "virchow2"),
    "VOLTA": ("volta", "all", "volta"),
    "LEMON": ("lemon", "vits8", "lemon"),
    "PAST": ("past", "all", "past"),
    "SpatialWhisperer": ("spatialwhisperer", "all", "spatialwhisperer"),
}


def digest(values):
    return hashlib.sha256(json.dumps(np.asarray(values).astype(str).tolist()).encode()).hexdigest()


def prepare(output):
    """Freeze the 133 paper-comparison inputs and expanded dataset configs."""
    if (output / "cases.json").exists():
        raise FileExistsError("cases.json already exists; use a new output directory")
    cases = []
    for dataset in ("nucls", "pannuke", "lizard", "phenocell", "hest"):
        config = load_config(ROOT / "configs/inference" / f"{dataset}.yaml")
        for name, (method, version, arm) in METHODS.items():
            root = OUTPUTS_ROOT / "reproducibility/inference" / method / version
            runs = sorted((root / dataset).iterdir()) if dataset == "hest" else [root / dataset]
            runs = [p for p in runs if p.is_dir() and p.name != "TENX143"]
            if len(runs) != (14 if dataset == "hest" else 1):
                raise ValueError(f"unexpected run count for {name}/{dataset}: {len(runs)}")
            for run in runs:
                for task in (["all", "immune"] if dataset == "phenocell" else ["all"]):
                    sample = run.name if dataset == "hest" else ""
                    source = run / "evaluation" / arm / task / "linear"
                    if sample:
                        source /= sample
                    source /= "conformal_prediction"
                    manifest = run / "embeddings" / arm / "manifest.csv"
                    for path in (manifest, source / "metrics.csv", source / "predictions.npz"):
                        if not path.is_file():
                            raise FileNotFoundError(path)
                    cases.append({"method": name, "dataset": dataset, "task": task,
                                  "sample_id": sample, "arm": arm, "manifest": str(manifest),
                                  "source": str(source), "config": config,
                                  "key": f"{method}/{dataset}/{task}/{sample or 'test'}"})
    output.mkdir(parents=True, exist_ok=True)
    (output / "cases.json").write_text(json.dumps(cases, indent=2))
    print(f"Prepared {len(cases)} cases in {output}")


def prepare_task(case):
    config = case["config"]
    settings, conformal = config["probe"], config["conformal"]
    seed = int(settings.get("seed", 0))
    common = {"arm": case["arm"], "pca_components": int(settings.get("pca_components", 256)), "seed": seed,
              "min_class_count_per_split": int(conformal.get("min_class_count_per_split", 20))}
    if case["dataset"] == "hest":
        source = sources.load(config["data"])
        sample = case["sample_id"]
        loaded = load_manifest_sample(case["manifest"], sample)
        metadata = hest.align_metadata(source, config["data"], loaded, sample)
        labels, coordinates = source.evaluation_paths(config["data"], sample)
        mapping = hest.load_sample_mapping(config["data"]["mapping_dir"], sample, "hest")
        task = hest.prepare_task(
            loaded.embeddings, metadata["cell_id"].to_numpy(), labels, coordinates,
            id_column=settings.get("label_id_column", "cell_id"), label_column=settings.get("label_column", "group"),
            min_class_count=int(settings.get("min_class_count", 25)),
            test_fraction=float(settings.get("test_fraction", .2)), split_axis=settings.get("split_axis", "y"),
            calibration_fraction_within_fit=float(conformal.get("calibration_fraction_within_fit", .2)),
            mapping=mapping, **common,
        )
        groups = np.repeat(sample, len(task.test))
    else:
        loaded = load_manifest(case["manifest"])
        metadata = _align_metadata(config, loaded)
        task = conformal_ood.prepare_task(loaded.embeddings, metadata, load_mapping(conformal["mapping"]),
                                          task=case["task"], **common)
        groups = bootstrap.resolve_groups(metadata, task.ids[task.test], settings["bootstrap"])
    return task, groups


def evaluate(case, output):
    destination = output / "runs" / case["key"]
    if (destination / "audit.json").exists():
        raise FileExistsError(f"completed case exists: {destination}")
    source = Path(case["source"])
    original = pd.read_csv(source / "metrics.csv")
    if original["C"].nunique() != 1:
        raise ValueError("saved conformal methods disagree on probe C")
    C = float(original["C"].iloc[0])
    task, groups = prepare_task(case)
    train, validation, test = conformal_ood._indices(task)
    for column, count in (("n_train", len(train)), ("n_calibration", len(validation)), ("n_test", len(test))):
        if original[column].nunique() != 1 or int(original[column].iloc[0]) != count:
            raise ValueError(f"saved {column} differs from reconstructed split")
    settings = dict(case["config"]["probe"]["linear"], Cs=[C])
    settings["seed"] = int(case["config"]["probe"].get("seed", 0))
    print(f"Fitting {case['key']}: train={len(train)}, validation={len(validation)}, test={len(test)}, C={C}", flush=True)
    with warnings.catch_warnings(record=True) as fit_warnings:
        warnings.simplefilter("always", ConvergenceWarning)
        model, _ = probe_api.fit(replace(task, validation=None), "linear", settings)
    classes = model.classes_.astype(str)
    logits = []
    for indices in (validation, test):
        values = model.decision_function(task.features[indices])
        if values.ndim == 1:
            values = np.column_stack((np.zeros(len(values)), values))
        logits.append(values)
    validation_logits, test_logits = logits
    probabilities = model.predict_proba(task.features[test])
    if not np.allclose(np.exp(log_probabilities(test_logits)), probabilities, atol=1e-6, rtol=1e-6):
        raise ValueError("exported logits do not reproduce the fitted classifier probabilities")
    with np.load(source / "predictions.npz", allow_pickle=False) as saved:
        for name, value in (("ids", task.ids[test]), ("y_true", task.labels[test]), ("classes", classes)):
            if not np.array_equal(saved[name].astype(str), np.asarray(value).astype(str)):
                raise ValueError(f"saved {name} mismatch: {case['key']}")
        difference = np.abs(saved["probabilities"] - probabilities)
        error = float(np.max(difference))
        mean_error = float(difference.mean())
        changed_predictions = int(np.sum(saved["probabilities"].argmax(1) != probabilities.argmax(1)))
        if error > 1e-3:
            if os.environ.get("OPENBLAS_NUM_THREADS") != "16":
                print(f"Reproduction error {error:.3g}; retrying with the original 16-thread setting", flush=True)
                cases = json.loads((output / "cases.json").read_text())
                index = next(i for i, candidate in enumerate(cases) if candidate["key"] == case["key"])
                environment = dict(os.environ, OPENBLAS_NUM_THREADS="16", OMP_NUM_THREADS="16",
                                   MKL_NUM_THREADS="16", NUMEXPR_NUM_THREADS="16")
                # Serialize expensive local retries to avoid competing 16-thread fits.
                # Batch jobs start with 16 threads and never enter this retry path.
                with (output / "reproduction.lock").open("a") as lock:
                    fcntl.flock(lock, fcntl.LOCK_EX)
                    subprocess.run([sys.executable, "-u", "-m", "cellbridge.inference.calibration.run",
                                    "evaluate", "--output-dir", str(output), "--index", str(index)],
                                   env=environment, check=True)
                return
            raise ValueError(f"refitted probabilities differ from saved run by {error}")
    lookup = {label: i for i, label in enumerate(classes)}
    validation_y = np.array([lookup[str(label)] for label in task.labels[validation]])
    test_y = np.array([lookup[str(label)] for label in task.labels[test]])
    temperature = fit_temperature(validation_logits, validation_y)
    destination.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination / "predictions.npz", classes=classes,
                        validation_ids=task.ids[validation].astype(str), validation_y=validation_y,
                        validation_logits=validation_logits, test_ids=task.ids[test].astype(str),
                        test_y=test_y, test_logits=test_logits, groups=groups)
    rows, bins, replicates = [], [], []
    for variant, value in (("raw", 1.0), ("temperature_scaled", temperature["temperature"])):
        point, draws, reliability = statistics(test_logits, test_y, groups, temperature=value)
        rows.append({"variant": variant, **point})
        bins.append(reliability.assign(variant=variant))
        replicates.append(draws.assign(variant=variant, bootstrap=np.arange(len(draws))))
    pd.DataFrame(rows).to_csv(destination / "metrics.csv", index=False)
    pd.concat(bins).to_csv(destination / "reliability.csv", index=False)
    pd.concat(replicates).to_csv(destination / "bootstrap.csv", index=False)
    audit = {**{k: case[k] for k in ("method", "dataset", "task", "sample_id", "source", "manifest")},
             **temperature, "C": C, "n_train": len(train), "n_validation": len(validation), "n_test": len(test),
             "n_groups": len(np.unique(groups)), "max_probability_error": error,
             "mean_probability_error": mean_error, "changed_argmax_predictions": changed_predictions,
             "convergence_warning": any(issubclass(w.category, ConvergenceWarning) for w in fit_warnings),
             "blas_threads": os.environ.get("OPENBLAS_NUM_THREADS", "unspecified"),
             "probability_reproduction_atol": 1e-3,
             "train_ids_sha256": digest(task.ids[train]), "validation_ids_sha256": digest(task.ids[validation]),
             "test_identity_sha256": digest(np.column_stack((task.ids[test].astype(str), task.labels[test].astype(str), groups))),
             "classes_sha256": digest(classes), "probe_selection": "C reused from train-only CV in saved conformal run",
             "numpy_version": np.__version__}
    (destination / "audit.json").write_text(json.dumps(audit, indent=2))
    print(f"Completed {case['key']}: T={temperature['temperature']:.4f}, reproduction error={error:.3g}", flush=True)
    gc.collect()


def evaluate_all(output, workers):
    """Run independent CPU evaluations with bounded concurrency; resume completed cases."""
    if not 1 <= workers <= 4:
        raise ValueError("workers must be between 1 and 4")
    cases = json.loads((output / "cases.json").read_text())
    pending = [i for i, case in enumerate(cases) if not (output / "runs" / case["key"] / "audit.json").exists()]
    logs = output / "logs"
    logs.mkdir(exist_ok=True)
    environment = dict(os.environ, OPENBLAS_NUM_THREADS="2", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2")

    def run_one(index):
        with (logs / f"case_{index:03d}.log").open("w") as handle:
            result = subprocess.run([sys.executable, "-u", "-m", "cellbridge.inference.calibration.run",
                                     "evaluate", "--output-dir", str(output), "--index", str(index)],
                                    stdout=handle, stderr=subprocess.STDOUT, env=environment)
        return index, result.returncode

    failures = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run_one, index) for index in pending]
        for future in as_completed(futures):
            index, code = future.result()
            print(f"{'FAILED' if code else 'Completed'} {index}: {cases[index]['key']}", flush=True)
            if code:
                failures.append(index)
    if failures:
        raise RuntimeError(f"Cases failed: {failures}; see {logs}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "evaluate", "evaluate-all", "report"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--index", type=int, help="zero-based case index")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args.output_dir)
    elif args.command == "evaluate":
        cases = json.loads((args.output_dir / "cases.json").read_text())
        if args.index is None or not 0 <= args.index < len(cases):
            parser.error("evaluate requires a valid --index")
        evaluate(cases[args.index], args.output_dir)
    elif args.command == "evaluate-all":
        evaluate_all(args.output_dir, args.workers)
    else:
        from cellbridge.inference.calibration.report import report
        report(args.output_dir)


if __name__ == "__main__":
    main()
