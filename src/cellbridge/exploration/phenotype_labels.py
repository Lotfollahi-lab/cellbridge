"""Conservative breast-cell labels and sampled frozen TERRA embeddings."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LabelSpec:
    cell_type: str
    lineage: str
    state: str = ""
    malignancy: str = "not_applicable"

    @property
    def phenotype(self):
        return self.cell_type if not self.state else f"{self.cell_type} ({self.state})"


def _s(cell_type, lineage, state="", malignancy="not_applicable"):
    return LabelSpec(cell_type, lineage, state, malignancy)


LABELS = {
    "Tumor_Cells": _s("tumor cell", "epithelial", malignancy="malignant"),
    "Proliferative_Tumor_Cells": _s(
        "tumor cell", "epithelial", "proliferating", "malignant"
    ),
    "Proliferating_Tumor_Cells": _s(
        "tumor cell", "epithelial", "proliferating", "malignant"
    ),
    "Proliferative_Tumor Cells": _s(
        "tumor cell", "epithelial", "proliferating", "malignant"
    ),
    "Inflammatory_Tumor_Cells": _s(
        "tumor cell", "epithelial", "inflammatory", "malignant"
    ),
    "Normal_Epithelial_Cells": _s(
        "epithelial cell", "epithelial", "normal", "non_malignant"
    ),
    "Alveolar_Epithelial_Cells": _s(
        "alveolar epithelial cell", "epithelial", malignancy="non_malignant"
    ),
    "Ductal_Epithelial_Cells": _s(
        "ductal epithelial cell", "epithelial", malignancy="non_malignant"
    ),
    "Myoepithelial": _s("myoepithelial cell", "epithelial", malignancy="non_malignant"),
    "Myoepithelial_Cells": _s(
        "myoepithelial cell", "epithelial", malignancy="non_malignant"
    ),
    "Normal_Myoepithelial_Cells": _s(
        "myoepithelial cell", "epithelial", "normal", "non_malignant"
    ),
    "Tumor_Adjacent_Myoepithelial_Cells": _s(
        "myoepithelial cell", "epithelial", "tumor-adjacent", "non_malignant"
    ),
    "T_Lymphocytes": _s("T cell", "lymphoid"),
    "Effector_T_Lymphocytes": _s("T cell", "lymphoid", "effector"),
    "B_Lymphocytes": _s("B cell", "lymphoid"),
    "Plasma_Cells": _s("plasma cell", "lymphoid"),
    "Macrophages": _s("macrophage", "myeloid"),
    "Activated_Macrophages": _s("macrophage", "myeloid", "activated"),
    "Dendritic_Cells": _s("dendritic cell", "myeloid"),
    "Mast_Cells": _s("mast cell", "myeloid"),
    "Stromal_Cells": _s("stromal cell", "stromal"),
    "MMP11+_CAFs": _s("cancer-associated fibroblast", "stromal", "MMP11+"),
    "CXCL14+_CAFs": _s("cancer-associated fibroblast", "stromal", "CXCL14+"),
    "Perivascular_Cells": _s("perivascular cell", "stromal"),
    "Adipocytes": _s("adipocyte", "stromal"),
    "Endothelial": _s("endothelial cell", "vascular"),
    "Endothelial_Cells": _s("endothelial cell", "vascular"),
    "Lymphatic_Endothelial_Cells": _s("lymphatic endothelial cell", "vascular"),
}

EXCLUSIONS = {
    "Epithelial_&_Tumor_Cells": "ambiguous epithelial/tumor composite",
    "Epithelial_&_Myoepithelial_Cells": "ambiguous epithelial composite",
    "Activated_Lymphoid_Cells": "lineage label without a cell identity",
    "Myeloid_Cells": "lineage label without a cell identity",
    "MMP9+_Myeloid_Cells": "lineage label without a cell identity",
    "Necrotic_Cells": "cell state without a viable cell identity",
}


def annotate_label(raw_label):
    """Return normalized label metadata and an inclusion decision."""
    normalized = re.sub(r"^\d+_", "", str(raw_label).strip())
    spec = LABELS.get(normalized)
    if spec:
        return {
            "normalized_label": normalized,
            **asdict(spec),
            "phenotype": spec.phenotype,
            "decision": "included",
        }
    return {
        "normalized_label": normalized,
        "cell_type": "",
        "lineage": "",
        "state": "",
        "malignancy": "",
        "phenotype": "",
        "decision": EXCLUSIONS.get(normalized, "unmapped label"),
    }


def _annotate_table(labels, label_column):
    annotations = {
        raw: annotate_label(raw) for raw in labels[label_column].astype(str).unique()
    }
    fields = next(iter(annotations.values())).keys()
    for field in fields:
        labels[field] = (
            labels[label_column]
            .astype(str)
            .map({raw: values[field] for raw, values in annotations.items()})
        )
    return labels


def _sample_rows(table, limit, rng):
    rows = []
    for _, group in table.groupby("phenotype", sort=True):
        index = group.index.to_numpy()
        take = min(limit, len(index))
        rows.extend(rng.choice(index, size=take, replace=False))
    return table.loc[rows].sort_values("_embedding_row").reset_index(drop=True)


def load_sampled_embeddings(cfg, embedding_loader=None, *, normalize=True):
    """Load the exact balanced phenotype cohort used by the ordering metric.

    ``embedding_loader(sample_id)`` may supply ``(cell_ids, matrix)`` from a
    model artifact. The default remains the original frozen-TERRA NPZ path.
    """
    root = Path(cfg["data_root"])
    key = cfg.get("embedding_key", "cell_emb")
    label_id = cfg.get("label_id_column", "cell_id")
    label_column = cfg.get("label_column", "group")
    limit = int(cfg["sampling"]["max_cells_per_phenotype_per_sample"])
    seed = int(cfg["sampling"]["seed"])
    all_embeddings, all_cells, audits = [], [], []
    dimension = None

    for sample_id in cfg["sample_ids"]:
        directory = root / sample_id
        labels = pd.read_csv(directory / f"{sample_id}_granular_cell_labels.csv")
        if labels[label_id].duplicated().any():
            raise ValueError(f"{sample_id}: duplicate label cell IDs")
        labels = labels[[label_id, label_column]].copy()
        labels[label_id] = labels[label_id].astype(str)
        labels["raw_label"] = labels[label_column].astype(str)
        labels = _annotate_table(labels, label_column)

        if embedding_loader is None:
            with np.load(
                directory / f"{sample_id}_terra.npz", allow_pickle=True
            ) as data:
                if key not in data:
                    raise KeyError(f"{sample_id}: missing embedding key {key!r}")
                embedding_ids = data["cell_ids"].astype(str)
                source_matrix = np.asarray(data[key], dtype=np.float32)
        else:
            embedding_ids, source_matrix = embedding_loader(str(sample_id))
            embedding_ids = np.asarray(embedding_ids).astype(str)
            source_matrix = np.asarray(source_matrix, dtype=np.float32)
        if source_matrix.ndim != 2 or len(source_matrix) != len(embedding_ids):
            raise ValueError(f"{sample_id}: embedding rows and cell IDs must align")

        labels["_embedding_row"] = pd.Index(embedding_ids).get_indexer(labels[label_id])
        labels["embedding_matched"] = labels["_embedding_row"] >= 0
        eligible = labels[
            (labels["decision"] == "included") & labels["embedding_matched"]
        ].copy()
        numeric_id = int(re.search(r"\d+$", sample_id).group())
        chosen = _sample_rows(
            eligible,
            limit,
            np.random.default_rng(np.random.SeedSequence([seed, numeric_id])),
        )
        matrix = source_matrix[chosen["_embedding_row"].to_numpy()]

        if matrix.ndim != 2 or not np.isfinite(matrix).all():
            raise ValueError(f"{sample_id}: embeddings must be a finite matrix")
        dimension = dimension or matrix.shape[1]
        if matrix.shape[1] != dimension:
            raise ValueError(f"{sample_id}: inconsistent embedding dimension")
        if normalize:
            norms = np.linalg.norm(matrix, axis=1)
            if np.any(norms == 0):
                raise ValueError(f"{sample_id}: zero-norm embeddings")
            matrix = matrix / norms[:, None]
        all_embeddings.append(matrix)
        chosen["sample_id"] = sample_id
        all_cells.append(chosen.drop(columns=[label_column, "_embedding_row"]))

        matched = labels[labels["embedding_matched"]].groupby("raw_label").size()
        sampled = chosen.groupby("raw_label").size()
        audit = labels.groupby("raw_label", sort=True).first().reset_index()
        audit["n_labelled"] = (
            labels.groupby("raw_label").size().reindex(audit["raw_label"]).to_numpy()
        )
        audit["n_embedding_matched"] = (
            audit["raw_label"].map(matched).fillna(0).astype(int)
        )
        audit["n_sampled"] = audit["raw_label"].map(sampled).fillna(0).astype(int)
        audit["sample_id"] = sample_id
        audits.append(
            audit[
                [
                    "sample_id",
                    "raw_label",
                    "normalized_label",
                    "cell_type",
                    "lineage",
                    "state",
                    "malignancy",
                    "phenotype",
                    "decision",
                    "n_labelled",
                    "n_embedding_matched",
                    "n_sampled",
                ]
            ]
        )

    cells = pd.concat(all_cells, ignore_index=True)
    cells["embedding_index"] = np.arange(len(cells))
    return np.vstack(all_embeddings), cells, pd.concat(audits, ignore_index=True)
