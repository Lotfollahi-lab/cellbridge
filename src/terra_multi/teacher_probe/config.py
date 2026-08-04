"""Configuration validation for the teacher-only gene-token experiment."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from terra_multi.config import load_config


def load_teacher_probe_config(path, *, smoke=False):
    """Load one experiment config and resolve full/smoke paths and limits."""
    cfg = deepcopy(load_config(path))
    if cfg.get("experiment") != "terra_gene_token_probe_v1":
        raise ValueError("teacher probe requires experiment: terra_gene_token_probe_v1")

    genes = cfg.get("genes")
    if not isinstance(genes, list) or not 20 <= len(genes) <= 30:
        raise ValueError("genes must contain 20--30 gene symbols")
    genes = [str(gene) for gene in genes]
    if len(set(genes)) != len(genes):
        raise ValueError("genes must be unique")
    cfg["genes"] = genes

    required = (
        "sample_id",
        "adata_path",
        "labels_path",
        "terra_embeddings",
        "gene_tokens_path",
        "output_dir",
    )
    missing = [name for name in required if not cfg.get(name)]
    if missing:
        raise ValueError(f"teacher probe config is missing: {missing}")

    eval_cfg = cfg.setdefault("eval", {})
    n_splits = int(eval_cfg.get("n_splits", 5))
    block_size = float(eval_cfg.get("spatial_block_size_um", 500.0))
    n_components = int(eval_cfg.get("gene_pca_components", 4))
    if n_splits < 2 or block_size <= 0 or not 3 <= n_components <= 5:
        raise ValueError("invalid spatial CV or gene PCA settings")
    eval_cfg.update(
        n_splits=n_splits,
        spatial_block_size_um=block_size,
        gene_pca_components=n_components,
    )
    expression_source = str(eval_cfg.get("expression_source", "focal"))
    if expression_source not in {"focal", "neighborhood"}:
        raise ValueError("eval.expression_source must be focal or neighborhood")
    eval_cfg["expression_source"] = expression_source
    if expression_source == "neighborhood":
        neighborhood = cfg.setdefault("neighborhood_expression", {})
        if int(neighborhood.get("n_neighbors", 0)) < 1:
            raise ValueError("neighborhood_expression.n_neighbors must be positive")
        if not neighborhood.get("cache_path"):
            raise ValueError("neighborhood_expression.cache_path is required")

    mode = cfg.get("smoke", {}) if smoke else cfg.get("selection", {})
    max_cells = int(mode.get("max_cells", 0))
    if max_cells < 1:
        raise ValueError("selection.max_cells and smoke.max_cells must be positive")
    cfg["active_max_cells"] = max_cells
    cfg["active_gene_tokens_path"] = str(
        mode.get("gene_tokens_path", cfg["gene_tokens_path"])
    )
    cfg["active_output_dir"] = str(mode.get("output_dir", cfg["output_dir"]))
    if smoke and "min_class_count" in mode:
        eval_cfg["min_class_count"] = int(mode["min_class_count"])
    cfg["smoke_mode"] = bool(smoke)

    extraction = cfg.setdefault("extraction", {})
    if smoke and "batch_size" in mode:
        extraction["batch_size"] = int(mode["batch_size"])
    if int(extraction.get("batch_size", 0)) < 1:
        raise ValueError("extraction.batch_size must be positive")
    for name in ("model_name", "model_revision", "terra_version"):
        if not extraction.get(name):
            raise ValueError(f"extraction.{name} is required")
    return cfg


def ensure_output_directories(cfg):
    """Create only the active result and derived-feature cache directories."""
    output = Path(cfg["active_output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    cache = output / "feature_cache"
    cache.mkdir(parents=True, exist_ok=True)
    return output, cache
