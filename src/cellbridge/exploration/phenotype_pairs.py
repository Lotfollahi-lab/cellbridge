"""Balanced cross-sample similarity and ordering analyses."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


RELATIONS = ("unrelated", "related_lineage", "same_type")
RELATION_SCORE = {name: score for score, name in enumerate(RELATIONS)}


def _paired_cosine(left, right):
    return np.einsum("ij,ij->i", left, right)


def _cell_pools(cells):
    pools = []
    for (sample, phenotype), group in cells.groupby(
        ["sample_id", "phenotype"], sort=True
    ):
        pools.append(
            {
                "sample": sample,
                "phenotype": phenotype,
                "lineage": group["lineage"].iloc[0],
                "rows": group["embedding_index"].to_numpy(),
            }
        )
    return pools


def _relation(left, right):
    if left["phenotype"] == right["phenotype"]:
        return "same_type"
    if left["lineage"] == right["lineage"]:
        return "related_lineage"
    return "unrelated"


def _pair_strata(pools):
    strata = {relation: [] for relation in RELATIONS}
    for left in range(len(pools)):
        for right in range(left + 1, len(pools)):
            if pools[left]["sample"] == pools[right]["sample"]:
                continue
            strata[_relation(pools[left], pools[right])].append((left, right))
    missing = [relation for relation, values in strata.items() if not values]
    if missing:
        raise ValueError(f"no eligible cross-sample strata for {missing}")
    return strata


def _draw_rows(strata, pools, count, rng):
    selected = rng.integers(len(strata), size=count)
    left_rows = np.empty(count, dtype=np.int64)
    right_rows = np.empty(count, dtype=np.int64)
    for stratum_id in np.unique(selected):
        positions = np.flatnonzero(selected == stratum_id)
        left, right = strata[stratum_id]
        left_rows[positions] = rng.choice(pools[left]["rows"], size=len(positions))
        right_rows[positions] = rng.choice(pools[right]["rows"], size=len(positions))
    return selected, left_rows, right_rows


def sample_cross_sample_pairs(embeddings, cells, pairs_per_relation, seed):
    """Sample each relation uniformly over eligible phenotype/sample strata."""
    pools = _cell_pools(cells)
    strata_by_relation = _pair_strata(pools)
    rng = np.random.default_rng(seed)
    frames = []
    for relation in RELATIONS:
        strata = strata_by_relation[relation]
        selected, left_rows, right_rows = _draw_rows(
            strata, pools, pairs_per_relation, rng
        )
        similarities = np.einsum(
            "ij,ij->i", embeddings[left_rows], embeddings[right_rows]
        )
        left_pool = np.fromiter(
            (strata[index][0] for index in selected), dtype=np.int64
        )
        right_pool = np.fromiter(
            (strata[index][1] for index in selected), dtype=np.int64
        )
        frames.append(
            pd.DataFrame(
                {
                    "relation": relation,
                    "relatedness": RELATION_SCORE[relation],
                    "cosine_similarity": similarities,
                    "sample_a": [pools[index]["sample"] for index in left_pool],
                    "sample_b": [pools[index]["sample"] for index in right_pool],
                    "phenotype_a": [pools[index]["phenotype"] for index in left_pool],
                    "phenotype_b": [pools[index]["phenotype"] for index in right_pool],
                    "lineage_a": [pools[index]["lineage"] for index in left_pool],
                    "lineage_b": [pools[index]["lineage"] for index in right_pool],
                }
            )
        )
    pairs = pd.concat(frames, ignore_index=True)
    if (pairs["sample_a"] == pairs["sample_b"]).any():
        raise AssertionError("within-sample pair escaped filtering")
    return pairs


def _triplet_candidates(pools):
    eligible = []
    for anchor, source in enumerate(pools):
        candidates = {relation: [] for relation in RELATIONS}
        for target, other in enumerate(pools):
            if source["sample"] == other["sample"]:
                continue
            candidates[_relation(source, other)].append(target)
        if all(candidates.values()):
            eligible.append((anchor, candidates))
    if not eligible:
        raise ValueError("no anchor pool has all three cross-sample relations")
    return eligible


def sample_ordering_triplets(embeddings, cells, count, seed, similarity=None):
    """Sample anchor-matched same, related, and unrelated cross-sample pairs."""
    pools = _cell_pools(cells)
    eligible = _triplet_candidates(pools)
    rng = np.random.default_rng(seed)
    selected_anchors = rng.integers(len(eligible), size=count)
    anchor_rows = np.empty(count, dtype=np.int64)
    target_rows = {relation: np.empty(count, dtype=np.int64) for relation in RELATIONS}
    target_pools = {relation: np.empty(count, dtype=np.int64) for relation in RELATIONS}

    for eligible_id in np.unique(selected_anchors):
        positions = np.flatnonzero(selected_anchors == eligible_id)
        anchor, candidates = eligible[eligible_id]
        anchor_rows[positions] = rng.choice(pools[anchor]["rows"], len(positions))
        for relation in RELATIONS:
            chosen_pools = rng.choice(candidates[relation], len(positions))
            target_pools[relation][positions] = chosen_pools
            for pool_id in np.unique(chosen_pools):
                subset = positions[chosen_pools == pool_id]
                target_rows[relation][subset] = rng.choice(
                    pools[pool_id]["rows"], len(subset)
                )

    score = similarity or _paired_cosine
    similarities = {
        relation: score(embeddings[anchor_rows], embeddings[target_rows[relation]])
        for relation in RELATIONS
    }
    anchor_pool_ids = np.fromiter(
        (eligible[index][0] for index in selected_anchors), dtype=np.int64
    )
    triplets = pd.DataFrame(
        {
            "anchor_sample": [pools[index]["sample"] for index in anchor_pool_ids],
            "anchor_phenotype": [
                pools[index]["phenotype"] for index in anchor_pool_ids
            ],
            "same_sample": [
                pools[index]["sample"] for index in target_pools["same_type"]
            ],
            "related_sample": [
                pools[index]["sample"] for index in target_pools["related_lineage"]
            ],
            "unrelated_sample": [
                pools[index]["sample"] for index in target_pools["unrelated"]
            ],
            "similarity_same": similarities["same_type"],
            "similarity_related": similarities["related_lineage"],
            "similarity_unrelated": similarities["unrelated"],
        }
    )
    triplets["same_minus_related"] = (
        triplets["similarity_same"] - triplets["similarity_related"]
    )
    triplets["related_minus_unrelated"] = (
        triplets["similarity_related"] - triplets["similarity_unrelated"]
    )
    triplets["same_not_above_related"] = triplets["same_minus_related"] <= 0
    triplets["related_not_above_unrelated"] = triplets["related_minus_unrelated"] <= 0
    triplets["any_ordering_violation"] = triplets[
        ["same_not_above_related", "related_not_above_unrelated"]
    ].any(axis=1)
    return triplets


def summarize_pairs(pairs):
    rows = []
    for relation in RELATIONS:
        values = pairs.loc[
            pairs["relation"] == relation, "cosine_similarity"
        ].to_numpy()
        rows.append(
            {
                "relation": relation,
                "n_pairs": len(values),
                "mean": np.mean(values),
                "std": np.std(values),
                "q05": np.quantile(values, 0.05),
                "median": np.median(values),
                "q95": np.quantile(values, 0.95),
            }
        )
    return pd.DataFrame(rows)


def summarize_ordering(triplets):
    return {
        "n_triplets": len(triplets),
        "same_not_above_related": float(triplets["same_not_above_related"].mean()),
        "related_not_above_unrelated": float(
            triplets["related_not_above_unrelated"].mean()
        ),
        "any_ordering_violation": float(triplets["any_ordering_violation"].mean()),
    }


def relatedness_correlation(pairs):
    result = spearmanr(pairs["relatedness"], pairs["cosine_similarity"])
    return {"spearman_rho": float(result.statistic), "pvalue": float(result.pvalue)}
