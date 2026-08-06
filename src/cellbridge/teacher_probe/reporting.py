"""Tables and compact report text for the teacher-only probe."""

from pathlib import Path

import pandas as pd

METRICS = ("macro_f1", "accuracy", "balanced_accuracy", "weighted_f1")


def summarize_results(metrics, per_class, baseline_arm=None, token_arm=None):
    baseline_arm = baseline_arm or "cell_emb_expression"
    token_arm = token_arm or "cell_emb_expression_gene_tokens"
    summary_rows = []
    for arm, table in metrics.groupby("arm", sort=False):
        row = {"arm": arm, "n_folds": len(table)}
        for metric in METRICS:
            row[f"{metric}_mean"] = table[metric].mean()
            row[f"{metric}_std"] = table[metric].std(ddof=1)
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    class_summary = (
        per_class.groupby(["arm", "class"], sort=False)["f1"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "f1_mean", "std": "f1_std"})
    )
    baseline = metrics[metrics["arm"] == baseline_arm].set_index("fold")
    tokens = metrics[metrics["arm"] == token_arm].set_index("fold")
    delta = pd.DataFrame({"fold": baseline.index})
    for metric in METRICS:
        delta[f"delta_{metric}"] = (
            tokens.loc[baseline.index, metric].to_numpy() - baseline[metric].to_numpy()
        )
    return summary, class_summary, delta


def write_markdown(path, summary, delta, baseline_arm, token_arm):
    columns = [
        "arm",
        "macro_f1_mean",
        "macro_f1_std",
        "accuracy_mean",
        "accuracy_std",
    ]
    lines = [
        "# TERRA contextual gene-token teacher probe",
        "",
        summary[columns].to_markdown(index=False, floatfmt=".4f"),
        "",
    ]
    values = delta["delta_macro_f1"]
    lines.append(
        f"Key delta `({token_arm}) - ({baseline_arm})`: "
        f"macro-F1 **{values.mean():+.4f} ± {values.std(ddof=1):.4f}** "
        f"across {len(values)} folds."
    )
    Path(path).write_text("\n".join(lines) + "\n")
