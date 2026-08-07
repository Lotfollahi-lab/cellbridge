"""Aggregate completed TERRA cell-flow seeds and apply the scale-up rule."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def aggregate(output_dir, seeds) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    rows, per_class = [], []
    for seed in seeds:
        run_dir = Path(output_dir) / f"seed{seed}"
        base = json.loads((run_dir / "base_diagnostic.json").read_text())
        prompt = json.loads((run_dir / "prompt_diagnostic.json").read_text())
        phenotype = pd.read_csv(run_dir / "phenotype_metrics.csv").set_index("arm")
        latent = pd.read_csv(run_dir / "latent_metrics.csv").set_index("arm")
        rows.append(
            {
                "seed": seed,
                "base_error_separation": base["relative_error_reduction"],
                "prompt_error_separation": prompt["relative_error_reduction"],
                "base_retrieval_top1": base["retrieval_top1"],
                "prompt_retrieval_top1": prompt["retrieval_top1"],
                "base_retrieval_top1_ci95_high": base.get(
                    "retrieval_top1_ci95_high", float("inf")
                ),
                "prompt_retrieval_top1_ci95_low": prompt.get(
                    "retrieval_top1_ci95_low", float("-inf")
                ),
                "real_terra_macro_f1": phenotype.loc["real_terra", "macro_f1"],
                "direct_he_macro_f1": phenotype.loc["direct_he", "macro_f1"],
                "flow_macro_f1": phenotype.loc["flow", "macro_f1"],
                "prompt_macro_f1": phenotype.loc["flow_softrepa", "macro_f1"],
                "flow_balanced_accuracy": phenotype.loc[
                    "flow", "balanced_accuracy"
                ],
                "prompt_balanced_accuracy": phenotype.loc[
                    "flow_softrepa", "balanced_accuracy"
                ],
                "flow_latent_cosine": latent.loc["flow", "latent_cosine"],
                "prompt_latent_cosine": latent.loc[
                    "flow_softrepa", "latent_cosine"
                ],
                "flow_latent_mse": latent.loc["flow", "latent_mse"],
                "prompt_latent_mse": latent.loc["flow_softrepa", "latent_mse"],
            }
        )
        scores = pd.read_csv(run_dir / "per_class_f1.csv")
        scores["seed"] = seed
        per_class.append(scores)

    table = pd.DataFrame(rows)
    class_table = (
        pd.concat(per_class)
        .groupby(["arm", "class"])["f1"]
        .agg(["mean", "std"])
        .reset_index()
    )
    means = table.mean(numeric_only=True)
    retrieval_robust = bool(
        (
            table["prompt_retrieval_top1_ci95_low"]
            > table["base_retrieval_top1_ci95_high"]
        ).all()
    )
    compatibility = (
        means["prompt_error_separation"] > means["base_error_separation"]
        and means["prompt_retrieval_top1"] > means["base_retrieval_top1"]
        and retrieval_robust
    )
    phenotype = (
        means["prompt_macro_f1"] > means["flow_macro_f1"]
        and means["prompt_balanced_accuracy"] > means["flow_balanced_accuracy"]
    )
    gate = {
        "seeds": list(map(int, seeds)),
        "retrieval_robustly_improved": retrieval_robust,
        "compatibility_improved": bool(compatibility),
        "phenotype_improved": bool(phenotype),
        "scale_up": bool(compatibility and phenotype),
        "mean_prompt_minus_base_error_separation": float(
            means["prompt_error_separation"] - means["base_error_separation"]
        ),
        "mean_prompt_minus_base_retrieval_top1": float(
            means["prompt_retrieval_top1"] - means["base_retrieval_top1"]
        ),
        "mean_prompt_minus_base_macro_f1": float(
            means["prompt_macro_f1"] - means["flow_macro_f1"]
        ),
        "mean_prompt_minus_base_balanced_accuracy": float(
            means["prompt_balanced_accuracy"] - means["flow_balanced_accuracy"]
        ),
    }
    return table, class_table, gate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    args = parser.parse_args()
    table, per_class, gate = aggregate(args.output_dir, args.seeds)
    output_dir = Path(args.output_dir)
    table.to_csv(output_dir / "three_seed_summary.csv", index=False)
    per_class.to_csv(output_dir / "three_seed_per_class_f1.csv", index=False)
    (output_dir / "three_seed_gate.json").write_text(
        json.dumps(gate, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(gate, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
