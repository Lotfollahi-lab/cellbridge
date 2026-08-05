"""Reproduce the TERRA phenotype-ordering analysis with frozen NOVAE embeddings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from terra_multi.config import load_config
from terra_multi.exploration.analyze_terra_phenotypes import analyze_embeddings
from terra_multi.exploration.novae_embeddings import (
    load_aligned_novae_embeddings,
    write_model_comparison,
)
from terra_multi.exploration.phenotype_labels import load_sampled_embeddings
from terra_multi.exploration.phenotype_pairs import (
    relatedness_correlation,
    sample_cross_sample_pairs,
    sample_ordering_triplets,
    summarize_ordering,
    summarize_pairs,
)


def _score_embeddings(embeddings, cells, sampling):
    seed = int(sampling["seed"])
    pairs = sample_cross_sample_pairs(
        embeddings, cells, int(sampling["pairs_per_relation"]), seed
    )
    triplets = sample_ordering_triplets(
        embeddings, cells, int(sampling["ordering_triplets"]), seed + 1
    )
    metrics = {
        "correlation": relatedness_correlation(pairs),
        "ordering": summarize_ordering(triplets),
    }
    return metrics, summarize_pairs(pairs)


def _append_comparison_summary(
    output,
    comparison,
    *,
    n_original,
    n_shared,
    n_missing,
    n_invalid,
):
    indexed = comparison.set_index("metric")
    violation = indexed.loc["any_ordering_violation"]
    rho = indexed.loc["spearman_rho"]
    lines = [
        "",
        "## Direct comparison with TERRA",
        "",
        (
            f"NOVAE retained {n_shared:,} of the original {n_original:,} sampled "
            f"cells. Alignment found {n_missing:,} missing cell IDs and "
            f"{n_invalid:,} cells without a valid NOVAE spatial neighborhood. "
            "Excluded cells were removed from both matrices before scoring."
        ),
        "",
        (
            "On this exact shared cohort, the any-ordering-violation rate was "
            f"{violation['novae']:.3%} for NOVAE and "
            f"{violation['terra_shared']:.3%} for TERRA, a "
            f"{100 * violation['novae_minus_terra_shared']:+.3f} percentage-point "
            "difference. The original full-cohort TERRA estimate was "
            f"{violation['terra_original']:.3%}."
        ),
        "",
        (
            f"Spearman ρ was {rho['novae']:.4f} for NOVAE and "
            f"{rho['terra_shared']:.4f} for TERRA on the shared cohort."
        ),
        "",
        "See `novae_alignment_audit.csv`, `shared_cohort_exclusions.csv`, and "
        "`terra_novae_comparison.csv` for the supporting audit and metrics.",
    ]
    with (output / "RESULTS.md").open("a") as handle:
        handle.write("\n".join(lines) + "\n")


def run(cfg):
    terra_embeddings, original_cells, audit = load_sampled_embeddings(cfg)
    novae_embeddings, alignment, usable = load_aligned_novae_embeddings(
        cfg, original_cells
    )
    excluded = original_cells.loc[~usable].copy()
    cells = original_cells.loc[usable].copy().reset_index(drop=True)
    cells["embedding_index"] = np.arange(len(cells))
    terra_embeddings = terra_embeddings[usable]
    novae_embeddings = novae_embeddings[usable]

    output = Path(cfg["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    excluded.to_csv(output / "shared_cohort_exclusions.csv", index=False)
    alignment.to_csv(output / "novae_alignment_audit.csv", index=False)

    terra_metrics, terra_distribution = _score_embeddings(
        terra_embeddings, cells, cfg["sampling"]
    )
    (output / "terra_shared_metrics.json").write_text(
        json.dumps(terra_metrics, indent=2) + "\n"
    )
    terra_distribution.to_csv(
        output / "terra_shared_similarity_distribution_summary.csv", index=False
    )
    pd.DataFrame([terra_metrics["ordering"]]).to_csv(
        output / "terra_shared_ordering_violations.csv", index=False
    )

    metrics = analyze_embeddings(cfg, novae_embeddings, cells, audit)
    comparison = write_model_comparison(
        cfg,
        metrics,
        terra_metrics,
        terra_distribution,
    )
    _append_comparison_summary(
        output,
        comparison,
        n_original=len(original_cells),
        n_shared=len(cells),
        n_missing=int(alignment["n_missing"].sum()),
        n_invalid=int(alignment["n_invalid_neighborhood"].sum()),
    )
    any_row = comparison[
        comparison["metric"] == "any_ordering_violation"
    ].iloc[0]
    print(
        f"Shared-valid cohort: {len(cells):,}/{len(original_cells):,} cells; "
        f"NOVAE any-ordering violation: {any_row['novae']:.3%}; "
        f"TERRA shared-cohort: {any_row['terra_shared']:.3%}; "
        "difference: "
        f"{100 * any_row['novae_minus_terra_shared']:+.2f} percentage points"
    )
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()
