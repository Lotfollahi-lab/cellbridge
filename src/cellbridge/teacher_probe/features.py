"""Fold-local contextual gene-token PCA feature construction."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from uuid import uuid4

import numpy as np
from sklearn.decomposition import PCA

from cellbridge.teacher_probe.store import read_gene_embeddings


def token_feature_names(genes, n_components):
    names = []
    for gene in genes:
        names.extend(f"{gene}:PC{component + 1}" for component in range(n_components))
        names.append(f"{gene}:present")
    return names


def gene_feature_slice(gene_index, n_components):
    start = gene_index * (n_components + 1)
    return slice(start, start + n_components), start + n_components


def _cache_signature(data, train, fold, n_components, seed):
    digest = hashlib.sha256()
    digest.update(data.token_store_signature.encode())
    digest.update(np.asarray(data.cell_ids).astype("S").tobytes())
    digest.update(np.asarray(train, dtype=np.int64).tobytes())
    settings = {
        "fold": int(fold),
        "genes": data.genes.tolist(),
        "n_components": int(n_components),
        "seed": int(seed),
    }
    digest.update(json.dumps(settings, sort_keys=True).encode())
    return digest.hexdigest()


def _load_cached(path, signature):
    if not Path(path).exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as cache:
            if str(cache["signature"].item()) != signature:
                return None
            return cache["features"].astype(np.float32), json.loads(
                str(cache["stats_json"].item())
            )
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None


def _save_cache(path, signature, features, stats):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp.npz")
    try:
        np.savez(
            temporary,
            signature=np.asarray(signature),
            features=np.asarray(features, dtype=np.float32),
            stats_json=np.asarray(json.dumps(stats, sort_keys=True)),
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build_fold_token_features(
    data, train, fold, eval_cfg, cache_dir, *, overwrite=False
):
    """Fit each gene PCA on present training rows and transform all present rows."""
    n_components = int(eval_cfg["gene_pca_components"])
    seed = int(eval_cfg.get("seed", 0))
    train = np.asarray(train, dtype=int)
    signature = _cache_signature(data, train, fold, n_components, seed)
    cache_path = Path(cache_dir) / f"fold_{fold}_gene_token_features.npz"
    if not overwrite:
        cached = _load_cached(cache_path, signature)
        if cached is not None:
            return cached

    width = len(data.genes) * (n_components + 1)
    features = np.zeros((len(data.cell_ids), width), dtype=np.float32)
    train_mask = np.zeros(len(data.cell_ids), dtype=bool)
    train_mask[train] = True
    stats = []
    for gene_index, gene in enumerate(data.genes):
        rows, embeddings = read_gene_embeddings(
            data.token_store_path, str(gene), data.store_rows
        )
        observed = np.zeros(len(data.cell_ids), dtype=bool)
        observed[rows] = True
        if not np.array_equal(observed, data.presence[:, gene_index]):
            raise ValueError(f"{gene}: sparse token rows disagree with presence mask")
        fit_rows = rows[train_mask[rows]]
        if len(fit_rows) <= n_components:
            raise ValueError(
                f"fold {fold} {gene}: only {len(fit_rows)} present training rows"
            )
        row_to_embedding = np.full(len(data.cell_ids), -1, dtype=int)
        row_to_embedding[rows] = np.arange(len(rows))
        fit_embeddings = embeddings[row_to_embedding[fit_rows]]
        pca = PCA(
            n_components=n_components,
            svd_solver="randomized",
            random_state=seed + 1000 * int(fold) + gene_index,
        )
        pca.fit(fit_embeddings)
        scores = pca.transform(embeddings).astype(np.float32)

        # Orient arbitrary PCA signs using training expression only. This does
        # not affect probe performance, but makes fold-local plots comparable.
        fit_scores = scores[row_to_embedding[fit_rows]]
        fit_expression = data.expression[fit_rows, gene_index]
        orientations = np.ones(n_components, dtype=np.float32)
        for component in range(n_components):
            if np.std(fit_scores[:, component]) and np.std(fit_expression):
                correlation = np.corrcoef(fit_scores[:, component], fit_expression)[
                    0, 1
                ]
                if np.isfinite(correlation) and correlation < 0:
                    orientations[component] = -1.0
        scores *= orientations

        pc_slice, mask_column = gene_feature_slice(gene_index, n_components)
        features[rows, pc_slice] = scores
        features[:, mask_column] = observed.astype(np.float32)
        stats.append(
            {
                "fold": int(fold),
                "gene": str(gene),
                "n_present": int(len(rows)),
                "n_present_train": int(len(fit_rows)),
                "explained_variance": float(pca.explained_variance_ratio_.sum()),
                "explained_variance_by_pc": pca.explained_variance_ratio_.tolist(),
                "orientation": orientations.tolist(),
            }
        )
    _save_cache(cache_path, signature, features, stats)
    return features, stats
