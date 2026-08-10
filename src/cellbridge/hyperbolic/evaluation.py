"""Geometry diagnostics integrated with the LoRA evaluation artifacts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from cellbridge.config import load_config
from cellbridge.evaluation.artifacts import write_json
from cellbridge.exploration.phenotype_labels import load_sampled_embeddings
from cellbridge.exploration.phenotype_pairs import (
    sample_ordering_triplets,
    summarize_ordering,
)
from cellbridge.hierarchical.dataset import (
    encode_ontology_node_ids,
    load_aligned_hierarchy,
    load_ontology_nodes,
)
from cellbridge.hyperbolic.losses import entailment_cone_loss
from cellbridge.lora.hest import evaluation_cohorts, evaluation_samples


def _feature_paths(cfg, root, expected_samples):
    paths = {}
    expected_samples = set(map(str, expected_samples))
    cohorts = ("breast_train", *evaluation_cohorts(cfg))
    for cohort in cohorts:
        for sample_id in evaluation_samples(cfg, cohort):
            if sample_id not in expected_samples:
                continue
            path = Path(root) / f"hest_{cohort}" / "features" / f"{sample_id}.npz"
            if sample_id in paths:
                raise ValueError(f"duplicate hierarchy-metric sample {sample_id}")
            paths[sample_id] = path
    missing = sorted(set(expected_samples) - set(paths))
    if missing:
        raise ValueError(
            "hierarchy metric cohort differs from phenotype sampler: "
            f"missing={missing}"
        )
    return paths


def _artifact_loader(paths, key):
    def load(sample_id):
        path = paths[sample_id]
        if not path.exists():
            raise FileNotFoundError(f"missing hierarchy-metric features: {path}")
        with np.load(path, allow_pickle=False) as data:
            if key not in data:
                raise KeyError(f"{path}: missing feature arm {key!r}")
            return data["cell_ids"].astype(str), data[key].astype(np.float32)

    return load


def _geodesic_similarity(model, device, chunk_size):
    @torch.no_grad()
    def similarity(left, right):
        scores = []
        for start in range(0, len(left), chunk_size):
            stop = start + chunk_size
            left_tangent = torch.as_tensor(left[start:stop], device=device)
            right_tangent = torch.as_tensor(right[start:stop], device=device)
            left_point = model.lorentz.expmap0(left_tangent)
            right_point = model.lorentz.expmap0(right_tangent)
            scores.append(
                -model.lorentz.distance(left_point, right_point).float().cpu().numpy()
            )
        return np.concatenate(scores)

    return similarity


def _cone_pairs(cells, cfg, nodes, label_id):
    node_ids = np.full((len(cells), 4), -1, dtype=np.int64)
    depths = np.zeros(len(cells), dtype=np.int64)
    usable = np.zeros(len(cells), dtype=bool)
    labels_dir = Path(cfg["hierarchy"]["labels_dir"])
    for sample_id, group in cells.groupby("sample_id", sort=False):
        rows = group.index.to_numpy()
        table = load_aligned_hierarchy(
            labels_dir / f"{sample_id}.parquet",
            group[label_id].astype(str).to_numpy(),
            str(sample_id),
        )
        node_ids[rows] = encode_ontology_node_ids(table, nodes)
        depths[rows] = table["hierarchy_depth"].to_numpy(dtype=np.int64)
        usable[rows] = table["use_in_hierarchy"].to_numpy(dtype=bool)
    levels = np.arange(4)[None, :]
    valid = (
        usable[:, None] & (node_ids >= 0) & (levels > 0) & (levels <= depths[:, None])
    )
    cell_rows, level_rows = np.nonzero(valid)
    return cell_rows, node_ids[cell_rows, level_rows]


@torch.no_grad()
def _cone_metrics(model, tangents, cell_rows, node_ids, K, device, chunk_size):
    if model.ontology_points is None:
        return {
            "cone_pairs": 0,
            "cone_activation_rate": float("nan"),
            "cone_violation_rate": float("nan"),
            "cone_loss": float("nan"),
        }
    loss_sum = active = violations = pairs = 0.0
    for start in range(0, len(cell_rows), chunk_size):
        stop = start + chunk_size
        rows = cell_rows[start:stop]
        cells = model.lorentz.expmap0(torch.as_tensor(tangents[rows], device=device))
        indices = torch.as_tensor(node_ids[start:stop], device=device)
        result = entailment_cone_loss(
            cells,
            model.ontology_points[indices],
            model.lorentz,
            K,
        )
        count = float(result["cone_pairs"].item())
        loss_sum += float(result["loss_cone"].item()) * count
        active += float(result["cone_active_pairs"].item())
        violations += float(result["cone_violations"].item())
        pairs += count
    return {
        "cone_pairs": int(pairs),
        "cone_activation_rate": active / pairs if pairs else float("nan"),
        "cone_violation_rate": violations / pairs if pairs else float("nan"),
        "cone_loss": loss_sum / pairs if pairs else float("nan"),
    }


def evaluate_geometry(cfg, model, root):
    """Evaluate exact phenotype triplets and cone health on exported HEST cells."""

    metric_path = cfg.get("eval", {}).get("hierarchy_metric_config")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if not metric_path:
        raise ValueError("eval.hierarchy_metric_config is required")
    metric_cfg = load_config(metric_path)
    paths = _feature_paths(cfg, root, metric_cfg["sample_ids"])
    device = next(model.parameters()).device
    chunk_size = int(cfg["eval"].get("geometry_chunk_size", 8192))
    if chunk_size < 1:
        raise ValueError("eval.geometry_chunk_size must be positive")
    similarity = (
        None
        if model.lorentz is None
        else _geodesic_similarity(model, device, chunk_size)
    )
    rows = []
    sampled_cells = None
    cones = cfg.get("hierarchy", {}).get("cones", {})
    cone_enabled = bool(cones.get("enabled", False))
    nodes = (
        load_ontology_nodes(cfg["hierarchy"]["nodes_path"]) if cone_enabled else None
    )
    if cone_enabled:
        if model.lorentz is None or model.ontology_points is None:
            raise ValueError("cone evaluation requires a cone-enabled checkpoint")
        if len(nodes) != len(model.ontology_points):
            raise ValueError("checkpoint ontology differs from hierarchy.nodes_path")

    arms = (("h&e", "image", "lora_projected"), ("terra", "terra", "terra_projected"))
    for modality, slug, key in arms:
        embeddings, cells, _ = load_sampled_embeddings(
            metric_cfg,
            _artifact_loader(paths, key),
            normalize=model.lorentz is None,
        )
        if sampled_cells is None:
            sampled_cells = cells
        elif not cells.equals(sampled_cells):
            raise RuntimeError("geometry modalities sampled different phenotype cells")
        triplets = sample_ordering_triplets(
            embeddings,
            cells,
            int(metric_cfg["sampling"]["ordering_triplets"]),
            int(metric_cfg["sampling"]["seed"]) + 1,
            similarity=similarity,
        )
        ordering = summarize_ordering(triplets)
        cone_metrics = {
            "cone_pairs": 0,
            "cone_activation_rate": float("nan"),
            "cone_violation_rate": float("nan"),
            "cone_loss": float("nan"),
        }
        if cone_enabled:
            label_id = metric_cfg.get("label_id_column", "cell_id")
            cell_rows, node_ids = _cone_pairs(cells, cfg, nodes, label_id)
            cone_metrics = _cone_metrics(
                model,
                embeddings,
                cell_rows,
                node_ids,
                float(cones["K"]),
                device,
                chunk_size,
            )
        rows.append(
            {
                "modality": modality,
                "geometry": model.geometry_mode,
                "dimension": model.alignment_dim,
                "curvature": model.current_curvature
                if model.lorentz is not None
                else np.nan,
                "mean_embedding_norm": float(np.linalg.norm(embeddings, axis=1).mean()),
                "hierarchy_violation_rate": ordering["any_ordering_violation"],
                **ordering,
                **cone_metrics,
            }
        )
        triplets.to_csv(root / f"ordering_triplets_{slug}.csv.gz", index=False)

    result = pd.DataFrame(rows)
    result.to_csv(root / "hierarchy_metrics.csv", index=False)
    write_json(root / "hierarchy_metrics.json", {"modalities": rows})
    sampled_cells.to_csv(root / "hierarchy_sampled_cells.csv.gz", index=False)
    return result
