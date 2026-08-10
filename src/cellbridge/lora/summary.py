"""Evaluation summary tables for LoRA experiments."""

from pathlib import Path

import pandas as pd

from cellbridge.evaluation.artifacts import write_table


def write_evaluation_summary(
    cfg,
    output_dir,
    hest_metrics=None,
    phenocell_metrics=None,
    method="linear",
):
    run_name = cfg.get("run_name", Path(cfg["output_dir"]).name)
    rows = []
    hest_tables = (
        hest_metrics.items()
        if isinstance(hest_metrics, dict)
        else [("label_test_fast", hest_metrics)]
    )
    for cohort, table in hest_tables:
        if table is None or table.empty:
            continue
        metric_columns = ["macro_f1", "balanced_accuracy"]
        if "norm_correctness_spearman" in table:
            metric_columns.append("norm_correctness_spearman")
        means = table.groupby("arm")[metric_columns].mean()
        task = (
            "label_test_fast_5_sample_mean" if cohort == "label_test_fast" else cohort
        )
        rows.extend(
            {
                "run_name": run_name,
                "dataset": "HEST",
                "task": task,
                "arm": arm,
                "macro_f1": values["macro_f1"],
                "balanced_accuracy": values["balanced_accuracy"],
                "norm_correctness_spearman": values.get("norm_correctness_spearman"),
            }
            for arm, values in means.iterrows()
        )
    if phenocell_metrics is not None and not phenocell_metrics.empty:
        for row in phenocell_metrics.itertuples(index=False):
            record = {
                "run_name": run_name,
                "dataset": "PhenoCell",
                "task": row.task,
                "arm": row.arm,
                "macro_f1": row.test_macro_f1,
                "balanced_accuracy": row.test_balanced_accuracy,
                "norm_correctness_spearman": getattr(
                    row,
                    "norm_correctness_spearman",
                    float("nan"),
                ),
            }
            if hasattr(row, "protocol"):
                record["protocol"] = row.protocol
            rows.append(record)
    if not rows:
        return None
    suffix = "" if method == "linear" else f"_{method}"
    path = Path(output_dir) / f"summary{suffix}.csv"
    write_table(path, rows)
    print(f"wrote evaluation summary -> {path}")
    return pd.DataFrame(rows)
