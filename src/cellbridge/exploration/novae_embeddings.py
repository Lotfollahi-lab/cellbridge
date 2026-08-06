"""Load NOVAE embeddings aligned to the deterministic frozen-TERRA cohort."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def _embedding_path(cfg, sample_id):
    directory = Path(cfg["novae"]["embedding_dir"])
    pattern = cfg["novae"].get("embedding_pattern", "{sample_id}_novae.npz")
    return directory / pattern.format(sample_id=sample_id)


def load_aligned_novae_embeddings(cfg, cells):
    """Return NOVAE rows, an audit, and the shared-valid cohort mask.

    Rows are aligned to the supplied deterministic TERRA cell order. Missing
    cells and invalid neighborhoods are represented by ``False`` in the mask;
    callers must apply the same mask to both NOVAE and TERRA before scoring.
    """
    if not cells.index.equals(pd.RangeIndex(len(cells))):
        raise ValueError("sampled cells must have a contiguous zero-based index")
    key = cfg["novae"].get("embedding_key", "novae_latent")
    valid_key = cfg["novae"].get("valid_key", "neighborhood_valid")
    exact = bool(cfg["novae"].get("require_exact_cohort", False))
    embeddings = None
    usable_global = np.zeros(len(cells), dtype=bool)
    dimension = None
    audits = []

    for sample_id, requested in cells.groupby("sample_id", sort=False):
        path = _embedding_path(cfg, sample_id)
        if not path.exists():
            raise FileNotFoundError(f"{sample_id}: missing NOVAE embeddings: {path}")
        with np.load(path, allow_pickle=False) as data:
            if key not in data:
                raise KeyError(f"{sample_id}: missing embedding key {key!r} in {path}")
            if "cell_ids" not in data:
                raise KeyError(f"{sample_id}: missing cell_ids in {path}")
            if valid_key not in data:
                raise KeyError(f"{sample_id}: missing validity key {valid_key!r} in {path}")
            ids = pd.Index(data["cell_ids"].astype(str))
            if ids.has_duplicates:
                raise ValueError(f"{sample_id}: duplicate NOVAE cell IDs")
            positions = ids.get_indexer(requested["cell_id"].astype(str))
            missing = positions < 0
            present = ~missing
            valid = np.zeros(len(requested), dtype=bool)
            valid[present] = np.asarray(data[valid_key], dtype=bool)[positions[present]]
            matrix = np.asarray(data[key], dtype=np.float32)
            if matrix.ndim != 2 or matrix.shape[0] != len(ids):
                raise ValueError(
                    f"{sample_id}: {key} must have one matrix row per cell ID"
                )
            dimension = dimension or matrix.shape[1]
            if matrix.shape[1] != dimension:
                raise ValueError(f"{sample_id}: inconsistent NOVAE embedding dimension")

            unusable = missing | ~valid
            audits.append(
                {
                    "sample_id": sample_id,
                    "embedding_path": str(path),
                    "n_requested": len(requested),
                    "n_missing": int(missing.sum()),
                    "n_invalid_neighborhood": int((present & ~valid).sum()),
                    "n_usable": int((~unusable).sum()),
                }
            )
            if exact and unusable.any():
                raise ValueError(
                    f"{sample_id}: exact TERRA cohort unavailable in NOVAE "
                    f"(missing={missing.sum()}, invalid={(present & ~valid).sum()})"
                )

            if embeddings is None:
                embeddings = np.full(
                    (len(cells), dimension), np.nan, dtype=np.float32
                )
            local_usable = ~unusable
            selected = matrix[positions[local_usable]]
            if not np.isfinite(selected).all():
                raise ValueError(f"{sample_id}: non-finite NOVAE embeddings")
            norms = np.linalg.norm(selected, axis=1)
            if np.any(norms == 0):
                raise ValueError(f"{sample_id}: zero-norm NOVAE embeddings")
            selected = selected / norms[:, None]
            global_rows = requested.index.to_numpy()[local_usable]
            embeddings[global_rows] = selected
            usable_global[global_rows] = True

    if embeddings is None:
        raise ValueError("no sampled cells supplied")
    return embeddings, pd.DataFrame(audits), usable_global


def _distribution_median(distribution, relation):
    return float(
        distribution.loc[distribution["relation"] == relation, "median"].iloc[0]
    )


def write_model_comparison(
    cfg,
    novae_metrics,
    terra_shared_metrics,
    terra_shared_distribution,
):
    """Compare NOVAE with TERRA on shared cells and the original benchmark."""
    output = Path(cfg["output_dir"])
    reference = Path(cfg["terra_reference_dir"])
    terra_original = json.loads((reference / "metrics.json").read_text())
    original_distribution = pd.read_csv(
        reference / "similarity_distribution_summary.csv"
    )
    novae_distribution = pd.read_csv(
        output / "similarity_distribution_summary.csv"
    )

    rows = []
    for name in (
        "same_not_above_related",
        "related_not_above_unrelated",
        "any_ordering_violation",
    ):
        original = float(terra_original["ordering"][name])
        terra = float(terra_shared_metrics["ordering"][name])
        novae = float(novae_metrics["ordering"][name])
        rows.append(
            {
                "metric": name,
                "terra_original": original,
                "terra_shared": terra,
                "novae": novae,
                "novae_minus_terra_shared": novae - terra,
            }
        )
    original_rho = float(terra_original["correlation"]["spearman_rho"])
    terra_rho = float(terra_shared_metrics["correlation"]["spearman_rho"])
    novae_rho = float(novae_metrics["correlation"]["spearman_rho"])
    rows.append(
        {
            "metric": "spearman_rho",
            "terra_original": original_rho,
            "terra_shared": terra_rho,
            "novae": novae_rho,
            "novae_minus_terra_shared": novae_rho - terra_rho,
        }
    )
    for relation in ("unrelated", "related_lineage", "same_type"):
        original = _distribution_median(original_distribution, relation)
        terra = _distribution_median(terra_shared_distribution, relation)
        novae = _distribution_median(novae_distribution, relation)
        rows.append(
            {
                "metric": f"median_cosine_{relation}",
                "terra_original": original,
                "terra_shared": terra,
                "novae": novae,
                "novae_minus_terra_shared": novae - terra,
            }
        )

    comparison = pd.DataFrame(rows)
    comparison.to_csv(output / "terra_novae_comparison.csv", index=False)
    payload = {
        "terra_original_reference_dir": str(reference),
        "terra_original": terra_original,
        "terra_shared_cohort": terra_shared_metrics,
        "novae_shared_cohort": novae_metrics,
        "any_violation_difference_percentage_points": 100
        * (
            novae_metrics["ordering"]["any_ordering_violation"]
            - terra_shared_metrics["ordering"]["any_ordering_violation"]
        ),
    }
    (output / "terra_novae_comparison.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    return comparison
