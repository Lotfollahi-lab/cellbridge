"""Score cell-type predictions at reviewed hierarchy levels.

HEST probes are trained and scored on raw source clusters such as
``0_Tumor_Cells``. Several of those clusters share one canonical identity, so
the raw macro-F1 also charges the model for within-identity confusions. This
module re-scores existing predictions at ``compartment``, ``lineage``, and
``identity`` using the reviewed mapping, without refitting any probe.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score

# Level name -> (hierarchy_depth required, mapping column).
LEVELS = {
    "compartment": (1, "level_1_compartment"),
    "lineage": (2, "level_2_lineage"),
    "identity": (3, "level_3_identity"),
}

# Stands in for a prediction that has no node at the scored level. It is never a
# valid truth value, so such predictions always count as errors.
UNRESOLVED = "<unresolved>"


def load_hierarchy_mapping(path) -> pd.DataFrame:
    """Load the reviewed source-label mapping, indexed by raw label."""

    columns = ["raw_label", "hierarchy_depth", *(column for _, column in LEVELS.values())]
    table = pd.read_csv(path, usecols=columns)
    table["raw_label"] = table["raw_label"].astype(str)
    if table["raw_label"].duplicated().any():
        raise ValueError(f"{path}: duplicate raw labels in the hierarchy mapping")
    return table.set_index("raw_label")


def _resolve(labels: np.ndarray, mapping: pd.DataFrame, level: str) -> pd.DataFrame:
    """Map raw labels to one level, flagging those that do not reach it."""

    depth, column = LEVELS[level]
    rows = mapping.reindex(pd.Index(labels, dtype=str))
    node = rows[column].astype("string")
    reached = (rows["hierarchy_depth"].fillna(0).to_numpy() >= depth) & node.notna()
    return pd.DataFrame(
        {"node": node.fillna(UNRESOLVED).to_numpy(), "reached": reached.to_numpy()}
    )


def score_level(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    mapping: pd.DataFrame,
    level: str,
) -> dict:
    """Score one hierarchy level on the cells whose truth reaches that level.

    Cells whose true label stops above ``level`` cannot be scored there and are
    excluded; ``coverage`` reports the share that survives. Predictions that
    stop above the level are kept and always scored wrong.
    """

    if level not in LEVELS:
        raise ValueError(f"unknown hierarchy level {level!r}")
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")

    truth = _resolve(np.asarray(y_true), mapping, level)
    prediction = _resolve(np.asarray(y_pred), mapping, level)
    keep = truth["reached"].to_numpy()
    scored_true = truth["node"].to_numpy()[keep]
    scored_pred = np.where(
        prediction["reached"].to_numpy()[keep],
        prediction["node"].to_numpy()[keep],
        UNRESOLVED,
    )
    if len(scored_true) == 0:
        return {
            "level": level,
            "n_scored": 0,
            "coverage": 0.0,
            "n_classes": 0,
            "macro_f1": float("nan"),
            "balanced_accuracy": float("nan"),
            "macro_f1_source_labels": float("nan"),
        }

    classes = sorted(set(scored_true))
    raw_true = np.asarray(y_true, dtype=str)[keep]
    raw_pred = np.asarray(y_pred, dtype=str)[keep]
    return {
        "level": level,
        "n_scored": int(keep.sum()),
        "coverage": float(keep.mean()),
        "n_classes": len(classes),
        "macro_f1": float(f1_score(scored_true, scored_pred, average="macro", labels=classes)),
        "balanced_accuracy": float(balanced_accuracy_score(scored_true, scored_pred)),
        # Raw-cluster score on the same cells, so the levels are comparable.
        "macro_f1_source_labels": float(
            f1_score(raw_true, raw_pred, average="macro", labels=sorted(set(raw_true)))
        ),
    }


def score_prediction_file(path, mapping: pd.DataFrame) -> list[dict]:
    """Score one ``predictions.npz`` artifact at every hierarchy level."""

    with np.load(Path(path), allow_pickle=False) as data:
        y_true = data["y_true"].astype(str)
        y_pred = data["y_pred"].astype(str)
    full = {
        "n_test": len(y_true),
        "macro_f1_source_labels_all_cells": float(
            f1_score(y_true, y_pred, average="macro", labels=sorted(set(y_true)))
        ),
    }
    return [{**full, **score_level(y_true, y_pred, mapping, level)} for level in LEVELS]
