"""Re-score finished HEST probes at reviewed hierarchy levels.

Reads the ``predictions.npz`` artifacts an evaluation already wrote and adds
identity-, lineage-, and compartment-level scores. Nothing is refitted, so this
runs on CPU in seconds against any completed run.

    PYTHONPATH=src python -m terra_multi.evaluation.rescore_hierarchy \
      --mapping outputs/hierarchical/hest_label_test/source_label_mapping.csv \
      --run-dir outputs/hierarchical/lora_baseline \
      --combined outputs/hierarchical/summary_hierarchy_all_runs.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from terra_multi.evaluation.hierarchy_metrics import (
    load_hierarchy_mapping,
    score_prediction_file,
)
from terra_multi.evaluation.storage import write_table

# PhenoCell and PanNuke use their own class vocabularies and have no reviewed
# Cell Ontology mapping, so only HEST cohorts can be re-scored here.
HEST_COHORT_PREFIX = "hest_"


def _parse_artifact_path(path: Path, evaluation_dir: Path) -> dict | None:
    """Read cohort, sample, arm, and method out of a predictions path."""

    parts = path.relative_to(evaluation_dir).parts
    if len(parts) == 5:
        cohort, _, sample_id, arm, _ = parts
        method = "linear"
    elif len(parts) == 6 and parts[4] == "knn":
        cohort, _, sample_id, arm, method, _ = parts
    else:
        return None
    if not cohort.startswith(HEST_COHORT_PREFIX):
        return None
    return {
        "cohort": cohort.removeprefix(HEST_COHORT_PREFIX),
        "sample_id": sample_id,
        "arm": arm,
        "method": method,
    }


def score_run(run_dir, mapping: pd.DataFrame) -> pd.DataFrame:
    """Score every HEST prediction artifact under one run directory."""

    run_dir = Path(run_dir)
    evaluation_dir = run_dir / "evaluation"
    if not evaluation_dir.is_dir():
        raise FileNotFoundError(f"{run_dir}: no evaluation directory")
    rows = []
    for path in sorted(evaluation_dir.rglob("predictions.npz")):
        context = _parse_artifact_path(path, evaluation_dir)
        if context is None:
            continue
        for scores in score_prediction_file(path, mapping):
            rows.append({"run_name": run_dir.name, **context, **scores})
    if not rows:
        raise ValueError(f"{run_dir}: no HEST prediction artifacts were scored")
    return pd.DataFrame(rows)


def aggregate(table: pd.DataFrame) -> pd.DataFrame:
    """Average per-sample scores into one row per cohort, arm, level."""

    keys = ["run_name", "cohort", "arm", "method", "level"]
    values = ["macro_f1", "balanced_accuracy", "macro_f1_source_labels", "coverage"]
    grouped = table.groupby(keys, as_index=False).agg(
        n_samples=("sample_id", "nunique"),
        **{name: (name, "mean") for name in values},
        macro_f1_std=("macro_f1", "std"),
    )
    return grouped.sort_values(keys).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mapping", required=True, help="source_label_mapping.csv")
    parser.add_argument("--run-dir", required=True, nargs="+", help="run output dirs")
    parser.add_argument("--combined", help="optional path for one pooled summary")
    args = parser.parse_args()

    mapping = load_hierarchy_mapping(args.mapping)
    tables = []
    for run_dir in args.run_dir:
        table = score_run(run_dir, mapping)
        summary = aggregate(table)
        write_table(Path(run_dir) / "evaluation" / "per_sample_hierarchy.csv", table)
        write_table(Path(run_dir) / "evaluation" / "summary_hierarchy.csv", summary)
        print(f"scored {len(table)} artifacts -> {run_dir}/evaluation/summary_hierarchy.csv")
        tables.append(summary)

    if args.combined:
        write_table(args.combined, pd.concat(tables, ignore_index=True))
        print(f"wrote pooled summary -> {args.combined}")


if __name__ == "__main__":
    main()
