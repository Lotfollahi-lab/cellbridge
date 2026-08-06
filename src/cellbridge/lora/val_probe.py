"""Labelled probe on the breast validation cohort, for checkpoint selection.

The paired contrastive loss plateaus long before the representation stops
improving, so it is a weak selection signal. This probe scores the embeddings
the contrastive validation pass has already computed -- no extra forward passes
-- using the same spatial split protocol as the reported HEST evaluation.

It reports raw source clusters (matching the headline metric) and canonical
identities (matching what the hierarchy objectives actually target).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from cellbridge.helpers.splits import spatial_split
from cellbridge.lora.data import canonical_id
from cellbridge.lora.hest import manifest_row, sample_dir

UNLABELLED = ""
LABEL_SETS = ("source", "identity")


@dataclass(frozen=True)
class _Section:
    """One validation section: its cells, labels, and fixed spatial split."""

    cell_ids: np.ndarray
    labels: dict[str, np.ndarray]
    is_train: np.ndarray
    is_test: np.ndarray


def _raw_labels(cfg, sample_id, cell_ids) -> np.ndarray:
    """Read the granular source clusters the reported HEST probe also uses."""

    settings = cfg["eval"]
    directory = sample_dir(cfg, sample_id, manifest_row(cfg, sample_id)[1])
    table = pd.read_csv(directory / f"{sample_id}_granular_cell_labels.csv")
    id_column = settings.get("label_id_column", "cell_id")
    label_column = settings.get("label_column", "group")
    table[id_column] = table[id_column].map(canonical_id)
    lookup = dict(zip(table[id_column], table[label_column].astype(str)))
    return np.array([lookup.get(cell_id, UNLABELLED) for cell_id in cell_ids])


def _identity_labels(dataset, rows: np.ndarray) -> np.ndarray:
    """Take canonical identities straight from the aligned hierarchy metadata."""

    codes = dataset.levels[rows, 3].numpy()
    resolved = (dataset.depth[rows].numpy() >= 3) & dataset.usable[rows].numpy()
    return np.where(resolved, codes.astype(str), UNLABELLED)


def _keep_frequent(labels: np.ndarray, minimum: int) -> np.ndarray:
    """Blank out classes too rare to score, so macro-F1 stays meaningful."""

    values, counts = np.unique(labels, return_counts=True)
    frequent = set(values[counts >= minimum]) - {UNLABELLED}
    return np.array([label if label in frequent else UNLABELLED for label in labels])


def build_validation_probe(cfg, dataset) -> "ValidationProbe":
    """Prepare per-section labels and spatial splits once, before training."""

    settings = cfg["eval"]
    minimum = int(settings.get("min_class_count", 25))
    base = dataset.base
    sections = {}
    for sample_id, rows in base.sample_indices.items():
        cell_ids = base.sample_cell_ids[sample_id]
        train, test = spatial_split(
            cell_ids,
            base.sample_centers[sample_id],
            float(settings.get("test_frac", 0.2)),
            settings.get("split_axis", "y"),
        )
        is_train = np.zeros(len(cell_ids), dtype=bool)
        is_test = np.zeros(len(cell_ids), dtype=bool)
        is_train[train] = True
        is_test[test] = True
        labels = {
            "source": _keep_frequent(_raw_labels(cfg, sample_id, cell_ids), minimum),
            "identity": _keep_frequent(_identity_labels(dataset, rows), minimum),
        }
        sections[sample_id] = _Section(cell_ids, labels, is_train, is_test)
    return ValidationProbe(sections)


def _macro_f1(features, labels, is_train, is_test) -> float | None:
    """Fit one section and return macro-F1, or None when it cannot be scored."""

    train = is_train & (labels != UNLABELLED)
    test = is_test & (labels != UNLABELLED)
    if not train.any() or not test.any() or len(set(labels[train])) < 2:
        return None
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(features[train], labels[train])
    predicted = model.predict(features[test])
    return float(f1_score(labels[test], predicted, average="macro"))


class ValidationProbe:
    """Score collected validation embeddings without refitting anything else."""

    def __init__(self, sections: dict[str, _Section]):
        if not sections:
            raise ValueError("the validation probe needs at least one section")
        self.sections = sections

    def score(self, cell_ids, features) -> dict[str, float]:
        """Average per-section macro-F1 over the sections that can be scored.

        ``cell_ids`` and ``features`` are the embeddings gathered during the
        contrastive validation pass, in whatever order the loader produced them.
        The loader drops a short tail per epoch, so cells it did not return are
        skipped rather than treated as an error.
        """

        position = {cell_id: row for row, cell_id in enumerate(cell_ids)}
        features = np.asarray(features, dtype=np.float32)
        scores: dict[str, list[float]] = {name: [] for name in LABEL_SETS}
        for section in self.sections.values():
            rows = np.array([position.get(c, -1) for c in section.cell_ids])
            present = rows >= 0
            embedded = features[rows[present]]
            for name in LABEL_SETS:
                value = _macro_f1(
                    embedded,
                    section.labels[name][present],
                    section.is_train[present],
                    section.is_test[present],
                )
                if value is not None:
                    scores[name].append(value)
        return {
            _metric_name(name): float(np.mean(values)) if values else float("nan")
            for name, values in scores.items()
        }


def _metric_name(label_set: str) -> str:
    return "probe_macro_f1" if label_set == "source" else f"probe_macro_f1_{label_set}"
