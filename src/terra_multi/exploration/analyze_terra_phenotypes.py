"""Measure phenotype structure in unchanged frozen TERRA embeddings.

Run from the repository root:

    PYTHONPATH=src python -m terra_multi.exploration.analyze_terra_phenotypes \
        --config configs/exploration/terra_phenotype_structure.yaml
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from terra_multi.config import load_config
from terra_multi.exploration.phenotype_labels import load_sampled_embeddings
from terra_multi.exploration.phenotype_pairs import (
    RELATIONS,
    relatedness_correlation,
    sample_cross_sample_pairs,
    sample_ordering_triplets,
    summarize_ordering,
    summarize_pairs,
)


DISPLAY = {
    "unrelated": "Unrelated",
    "related_lineage": "Related lineage",
    "same_type": "Same phenotype",
}
COLORS = {
    "unrelated": "#777777",
    "related_lineage": "#E69F00",
    "same_type": "#0072B2",
}


def _distribution_plot(pairs, path, seed, analysis_name="TERRA"):
    values = [
        pairs.loc[pairs["relation"] == relation, "cosine_similarity"].to_numpy()
        for relation in RELATIONS
    ]
    limits = (min(map(np.min, values)), max(map(np.max, values)))
    bins = np.linspace(*limits, 80)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for relation, scores in zip(RELATIONS, values):
        axes[0].hist(
            scores,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=2,
            color=COLORS[relation],
            label=DISPLAY[relation],
        )
    axes[0].set(xlabel="Cosine similarity", ylabel="Density")
    axes[0].legend(frameon=False)

    rng = np.random.default_rng(seed)
    shown = [
        rng.choice(scores, min(10_000, len(scores)), replace=False) for scores in values
    ]
    violin = axes[1].violinplot(shown, showmedians=True, showextrema=False)
    for body, relation in zip(violin["bodies"], RELATIONS):
        body.set_facecolor(COLORS[relation])
        body.set_alpha(0.75)
    axes[1].set_xticks(range(1, 4), [DISPLAY[r] for r in RELATIONS], rotation=15)
    axes[1].set_ylabel("Cosine similarity")
    fig.suptitle(
        f"Frozen {analysis_name} similarity by cross-sample phenotype relation"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _ordering_plot(triplets, path, analysis_name="TERRA"):
    margins = {
        "Same − related": triplets["same_minus_related"].to_numpy(),
        "Related − unrelated": triplets["related_minus_unrelated"].to_numpy(),
    }
    lower = min(np.quantile(values, 0.005) for values in margins.values())
    upper = max(np.quantile(values, 0.995) for values in margins.values())
    bins = np.linspace(lower, upper, 80)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for label, values in margins.items():
        violation = np.mean(values <= 0)
        ax.hist(
            values,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=2,
            label=f"{label} (violation {violation:.1%})",
        )
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Anchor-matched cosine margin")
    ax.set_ylabel("Density")
    ax.set_title(f"{analysis_name} cross-sample phenotype-ordering margins")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _heatmap_plot(pairs, cells, path, analysis_name="TERRA"):
    order = (
        cells[["phenotype", "lineage"]]
        .drop_duplicates()
        .sort_values(["lineage", "phenotype"])["phenotype"]
        .tolist()
    )
    position = {phenotype: index for index, phenotype in enumerate(order)}
    matrix = np.full((len(order), len(order)), np.nan)
    means = pairs.groupby(["phenotype_a", "phenotype_b"])[
        "cosine_similarity"
    ].mean()
    for (left, right), value in means.items():
        i, j = position[left], position[right]
        matrix[i, j] = matrix[j, i] = value

    cmap = plt.colormaps["viridis"].copy()
    cmap.set_bad("#eeeeee")
    finite = matrix[np.isfinite(matrix)]
    fig, ax = plt.subplots(figsize=(11, 9))
    image = ax.imshow(
        matrix,
        cmap=cmap,
        vmin=np.quantile(finite, 0.02),
        vmax=np.quantile(finite, 0.98),
    )
    ax.set_xticks(range(len(order)), order, rotation=90, fontsize=7)
    ax.set_yticks(range(len(order)), order, fontsize=7)
    ax.set_title(f"Mean {analysis_name} cross-sample cosine by phenotype pair")
    fig.colorbar(image, ax=ax, label="Mean cosine similarity", shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_summary(path, cfg, cells, distribution, ordering, correlation):
    analysis_name = cfg.get("analysis_name", "TERRA")
    embedding_name = cfg.get(
        "embedding_description", cfg.get("embedding_key", "cell_emb")
    )
    medians = distribution.set_index("relation")["median"]
    ordered = medians["same_type"] > medians["related_lineage"] > medians["unrelated"]
    lines = [
        f"# Frozen {analysis_name} phenotype-structure analysis",
        "",
        (
            f"This descriptive analysis used {len(cells):,} sampled cells from "
            f"{cells['sample_id'].nunique()} annotated breast samples and the unchanged "
            f"`{embedding_name}` embeddings. All compared cell "
            "pairs came from different samples."
        ),
        "",
        "## Results",
        "",
        (
            "Median cosine similarity was "
            f"{medians['same_type']:.4f} for the same phenotype, "
            f"{medians['related_lineage']:.4f} for related lineages, and "
            f"{medians['unrelated']:.4f} for unrelated cells. The expected ordering "
            f"was {'observed' if ordered else 'not observed'}."
        ),
        "",
        (
            "Ordinal phenotype relatedness and cosine similarity had Spearman "
            f"ρ = {correlation['spearman_rho']:.4f}."
        ),
        "",
        (
            f"Among {ordering['n_triplets']:,} anchor-matched triplets, "
            f"same≤related occurred in {ordering['same_not_above_related']:.1%}, "
            "related≤unrelated occurred in "
            f"{ordering['related_not_above_unrelated']:.1%}, and at least one ordering "
            f"violation occurred in {ordering['any_ordering_violation']:.1%}."
        ),
        "",
        "These estimates are descriptive and balanced over eligible sample/phenotype "
        "strata; they are not population-frequency estimates. No embedding was trained "
        "or modified.",
        "",
        "See `label_mapping_audit.csv` for every inclusion and exclusion decision.",
    ]
    path.write_text("\n".join(lines) + "\n")


def analyze_embeddings(cfg, embeddings, cells, audit):
    """Run the shared phenotype-structure analysis on an aligned embedding matrix."""
    output = Path(cfg["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    analysis_name = cfg.get("analysis_name", "TERRA")
    sampling = cfg["sampling"]
    seed = int(sampling["seed"])
    pairs = sample_cross_sample_pairs(
        embeddings, cells, int(sampling["pairs_per_relation"]), seed
    )
    triplets = sample_ordering_triplets(
        embeddings, cells, int(sampling["ordering_triplets"]), seed + 1
    )
    distribution = summarize_pairs(pairs)
    ordering = summarize_ordering(triplets)
    correlation = relatedness_correlation(pairs)

    cells.to_csv(output / "sampled_cells.csv.gz", index=False)
    audit.to_csv(output / "label_mapping_audit.csv", index=False)
    cells.groupby(["sample_id", "lineage", "phenotype"]).size().rename(
        "n_sampled"
    ).reset_index().to_csv(output / "sampled_cell_counts.csv", index=False)
    pairs.to_csv(output / "pair_similarities.csv.gz", index=False)
    triplets.to_csv(output / "ordering_triplets.csv.gz", index=False)
    distribution.to_csv(output / "similarity_distribution_summary.csv", index=False)
    pd.DataFrame([ordering]).to_csv(output / "ordering_violations.csv", index=False)
    metrics = {"correlation": correlation, "ordering": ordering}
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    (output / "resolved_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    _distribution_plot(
        pairs, output / "similarity_distributions.png", seed, analysis_name
    )
    _ordering_plot(triplets, output / "ordering_margins.png", analysis_name)
    _heatmap_plot(
        pairs, cells, output / "phenotype_similarity_heatmap.png", analysis_name
    )
    _write_summary(output / "RESULTS.md", cfg, cells, distribution, ordering, correlation)
    print((output / "RESULTS.md").read_text())
    print(f"Artifacts written to {output}")
    return metrics


def run(cfg):
    embeddings, cells, audit = load_sampled_embeddings(cfg)
    return analyze_embeddings(cfg, embeddings, cells, audit)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze phenotype structure in frozen TERRA embeddings."
    )
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run(load_config(args.config))


if __name__ == "__main__":
    main()
