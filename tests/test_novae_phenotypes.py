import json

import numpy as np
import pandas as pd
import pytest

from cellbridge.exploration.novae_embeddings import (
    load_aligned_novae_embeddings,
    write_model_comparison,
)


def _cfg(tmp_path):
    return {
        "output_dir": str(tmp_path / "output"),
        "terra_reference_dir": str(tmp_path / "terra"),
        "novae": {
            "embedding_dir": str(tmp_path / "embeddings"),
            "require_exact_cohort": True,
        },
    }


def _save(path, ids, matrix, valid=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        cell_ids=np.asarray(ids, dtype=str),
        novae_latent=np.asarray(matrix, dtype=np.float32),
        neighborhood_valid=np.ones(len(ids), dtype=bool)
        if valid is None
        else np.asarray(valid, dtype=bool),
    )


def test_novae_rows_align_to_exact_sampled_cell_order_and_are_normalized(tmp_path):
    cfg = _cfg(tmp_path)
    cells = pd.DataFrame(
        {"sample_id": ["A", "A", "B"], "cell_id": ["x", "y", "z"]}
    )
    directory = tmp_path / "embeddings"
    _save(directory / "A_novae.npz", ["y", "unused", "x"], [[0, 2], [1, 1], [3, 0]])
    _save(directory / "B_novae.npz", ["z"], [[4, 3]])

    embeddings, audit, usable = load_aligned_novae_embeddings(cfg, cells)
    assert usable.tolist() == [True, True, True]

    np.testing.assert_allclose(embeddings, [[1, 0], [0, 1], [0.8, 0.6]])
    assert audit["n_requested"].tolist() == [2, 1]
    assert audit["n_usable"].tolist() == [2, 1]
    assert audit[["n_missing", "n_invalid_neighborhood"]].to_numpy().sum() == 0


@pytest.mark.parametrize(
    "ids,valid,expected",
    [(["x"], [True], "missing=1"), (["x", "y"], [True, False], "invalid=1")],
)
def test_novae_alignment_refuses_to_change_the_terra_cohort(
    tmp_path, ids, valid, expected
):
    cfg = _cfg(tmp_path)
    cells = pd.DataFrame({"sample_id": ["A", "A"], "cell_id": ["x", "y"]})
    _save(tmp_path / "embeddings" / "A_novae.npz", ids, np.eye(len(ids), 2), valid)

    with pytest.raises(ValueError, match=expected):
        load_aligned_novae_embeddings(cfg, cells)


def test_novae_alignment_exposes_shared_valid_mask(tmp_path):
    cfg = _cfg(tmp_path)
    cfg["novae"]["require_exact_cohort"] = False
    cells = pd.DataFrame({"sample_id": ["A", "A"], "cell_id": ["x", "y"]})
    _save(
        tmp_path / "embeddings" / "A_novae.npz",
        ["x", "y"],
        np.eye(2),
        [True, False],
    )

    embeddings, _, usable = load_aligned_novae_embeddings(cfg, cells)
    assert usable.tolist() == [True, False]
    assert np.isnan(embeddings[1]).all()

def test_model_comparison_reports_percentage_point_difference(tmp_path):
    cfg = _cfg(tmp_path)
    reference = tmp_path / "terra"
    output = tmp_path / "output"
    reference.mkdir()
    output.mkdir()
    terra = {
        "correlation": {"spearman_rho": 0.4, "pvalue": 0.0},
        "ordering": {
            "n_triplets": 50_000,
            "same_not_above_related": 0.3,
            "related_not_above_unrelated": 0.2,
            "any_ordering_violation": 0.55,
        },
    }
    novae = {
        "correlation": {"spearman_rho": 0.5, "pvalue": 0.0},
        "ordering": {
            "n_triplets": 50_000,
            "same_not_above_related": 0.25,
            "related_not_above_unrelated": 0.15,
            "any_ordering_violation": 0.45,
        },
    }
    (reference / "metrics.json").write_text(json.dumps(terra))
    distribution = pd.DataFrame(
        {"relation": ["unrelated", "related_lineage", "same_type"], "median": [0.1, 0.2, 0.3]}
    )
    distribution.to_csv(reference / "similarity_distribution_summary.csv", index=False)
    (distribution.assign(median=[0.2, 0.4, 0.6])).to_csv(
        output / "similarity_distribution_summary.csv", index=False
    )

    comparison = write_model_comparison(cfg, novae, terra, distribution)

    row = comparison[comparison["metric"] == "any_ordering_violation"].iloc[0]
    assert row["novae_minus_terra_shared"] == pytest.approx(-0.1)
    payload = json.loads((output / "terra_novae_comparison.json").read_text())
    assert payload["any_violation_difference_percentage_points"] == pytest.approx(-10)
