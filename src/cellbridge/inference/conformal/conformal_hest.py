"""Within-sample conformal evaluation for annotated HEST sections."""

import gc
from dataclasses import replace
from pathlib import Path

import pandas as pd

from cellbridge.inference.embed import methods as embedding_methods
from cellbridge.inference.embed import sources
from cellbridge.inference.embed.artifacts import load_manifest_sample
from cellbridge.inference.conformal import conformal_ood as conformal
from cellbridge.inference.evaluate import hest
from cellbridge.ontology import load_ontology


def ontology_settings(config, sample_id):
    """Load one HEST sample's mapping and conformal graph settings."""
    settings = config["conformal"]
    mapping = hest.load_sample_mapping(
        config["data"]["mapping_dir"], sample_id, config["dataset"]
    )
    ontology = load_ontology(settings["ontology"])
    mapping.validate_ontology(ontology)

    alpha = float(settings.get("alpha", 0.1))
    if not 0 < alpha < 1:
        raise ValueError("conformal alpha must be between zero and one")
    configured_roots = settings.get("branch_roots", ())
    if isinstance(configured_roots, dict):
        configured_roots = configured_roots.get(str(sample_id), ())
    roots = tuple(map(str, configured_roots))
    if not roots or len(set(roots)) != len(roots):
        raise ValueError(
            f"{sample_id}: conformal branch_roots must be non-empty and unique"
        )
    unknown = sorted(set(roots) - set(ontology.names))
    if unknown:
        raise ValueError(f"{sample_id}: unknown conformal branch roots: {unknown}")
    return mapping, ontology, alpha, roots


def _with_context(result, context):
    return replace(
        result,
        metrics=tuple({**context, **row} for row in result.metrics),
        per_class_coverage=tuple(
            {**context, **row} for row in result.per_class_coverage
        ),
        calibration=tuple({**context, **row} for row in result.calibration),
    )


def _save_aggregate(output, metrics, coverage, calibration_rows):
    output.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame(metrics)
    metrics.to_csv(output / "metrics.csv", index=False)
    pd.DataFrame(coverage).to_csv(output / "per_class_coverage.csv", index=False)
    pd.DataFrame(calibration_rows).to_csv(output / "calibration.csv", index=False)
    summary = (
        metrics.groupby(
            ["dataset", "method", "arm", "task", "probe", "conformal_method"],
            sort=False,
        )
        .agg(
            n_samples=("sample_id", "size"),
            marginal_coverage_mean=("marginal_coverage", "mean"),
            marginal_coverage_std=("marginal_coverage", "std"),
            coverage_gap_mean=("coverage_gap", "mean"),
            coverage_gap_std=("coverage_gap", "std"),
            mean_prediction_set_size_mean=("mean_prediction_set_size", "mean"),
            mean_prediction_set_size_std=("mean_prediction_set_size", "std"),
            singleton_rate_mean=("singleton_rate", "mean"),
            root_inconclusive_rate_mean=("root_inconclusive_rate", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(output / "summary.csv", index=False)


def evaluate(config, methods, probes, tasks):
    """Evaluate every requested HEST representation on fixed spatial stripes."""
    if tasks != ["all"]:
        raise ValueError("HEST supports only the all-cell-type task")

    data = config["data"]
    source = sources.load(data)
    sample_ids = source.sample_ids(data)
    probe_settings = config["probe"]
    settings = config["conformal"]
    output_subdir = settings.get("output_subdir", "conformal")
    embedding_dir = Path(config["output_dir"]) / "embeddings"
    evaluation_dir = Path(config["output_dir"]) / "evaluation"
    seed = int(probe_settings.get("seed", 0))

    for method in methods:
        for arm in embedding_methods.arms(method):
            manifest = embedding_dir / arm / "manifest.csv"
            tables = {
                probe: {"metrics": [], "coverage": [], "calibration": []}
                for probe in probes
            }
            for sample_id in sample_ids:
                mapping, ontology, alpha, branch_roots = ontology_settings(
                    config, sample_id
                )
                loaded = load_manifest_sample(manifest, sample_id)
                metadata = hest.align_metadata(
                    source, data, loaded, sample_id
                )
                labels, coordinates = source.evaluation_paths(data, sample_id)
                task = hest.prepare_task(
                    loaded.embeddings,
                    metadata["cell_id"].to_numpy(),
                    labels,
                    coordinates,
                    id_column=probe_settings.get("label_id_column", "cell_id"),
                    label_column=probe_settings.get("label_column", "group"),
                    min_class_count=int(probe_settings.get("min_class_count", 25)),
                    test_fraction=float(probe_settings.get("test_fraction", 0.2)),
                    split_axis=probe_settings.get("split_axis", "y"),
                    arm=arm,
                    mapping=mapping,
                    calibration_fraction_within_fit=float(
                        settings.get("calibration_fraction_within_fit", 0.2)
                    ),
                    min_class_count_per_split=int(
                        settings.get("min_class_count_per_split", 20)
                    ),
                    pca_components=int(probe_settings.get("pca_components", 256)),
                    seed=seed,
                )
                for probe in probes:
                    current_probe_settings = dict(probe_settings.get(probe, {}))
                    current_probe_settings.setdefault("seed", seed)
                    result = conformal.evaluate(
                        task,
                        probe,
                        current_probe_settings,
                        mapping,
                        ontology,
                        branch_roots=branch_roots,
                        alpha=alpha,
                    )
                    context = {
                        "dataset": config["dataset"],
                        "sample_id": sample_id,
                        "method": method,
                        "arm": arm,
                        "task": "all",
                        "probe": probe,
                    }
                    result = _with_context(result, context)
                    output = (
                        evaluation_dir
                        / arm
                        / "all"
                        / probe
                        / sample_id
                        / output_subdir
                    )
                    conformal.save_result(output, result)
                    tables[probe]["metrics"].extend(result.metrics)
                    tables[probe]["coverage"].extend(result.per_class_coverage)
                    tables[probe]["calibration"].extend(result.calibration)
                    graph = result.metrics[1]
                    print(
                        f"[{method}/{arm}/{sample_id}/{probe}/{output_subdir}] "
                        f"coverage={graph['marginal_coverage']:.4f}, "
                        f"size={graph['mean_prediction_set_size']:.2f}"
                    )
                del task, loaded, metadata
                gc.collect()
            for probe, rows in tables.items():
                output = evaluation_dir / arm / "all" / probe / output_subdir
                _save_aggregate(
                    output,
                    rows["metrics"],
                    rows["coverage"],
                    rows["calibration"],
                )
