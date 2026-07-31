"""Shrink TERRA regression targets toward phenotype prototypes.

The ranking and negative-downweighting objectives added a constraint that
competes with the paired loss, and frozen TERRA violates the hierarchy ordering
in most sampled triplets, so the constraint loses. This module takes the other
route: move the structure into the target the alignment already regresses to.

Each cell's target becomes ``normalize((1 - alpha) * terra + alpha * prototype)``.
At ``alpha = 0`` this is the unchanged objective. It must stay well below 1 --
at ``alpha = 1`` every cell of one identity shares a target vector and the
exact-pair InfoNCE has no unique diagonal left to find.

Two prototype sources are supported so the mechanism can be separated from the
biology: canonical identities (labelled sections only) and unsupervised TERRA
clusters (every section). If both help equally, the gain is target denoising
rather than anything the hierarchy contributed.
"""

from __future__ import annotations

import numpy as np

from terra_multi.distill.prototype import assign_prototypes, fit_prototypes, l2_normalize

UNASSIGNED = -1


def fit_identity_prototypes(dataset) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Average normalized TERRA within each canonical identity.

    ``dataset`` is a ``HierarchyDataset``. Cells whose label stops above the
    identity level, or that are excluded, get ``UNASSIGNED`` and are left alone
    by :func:`shrink_targets`.
    """

    codes = dataset.levels[:, 3].numpy()
    resolved = (dataset.depth.numpy() >= 3) & dataset.usable.numpy()
    identities = np.unique(codes[resolved])
    if len(identities) == 0:
        raise ValueError("no cell resolves to a canonical identity")

    terra = l2_normalize(_stacked_terra(dataset.base))
    prototypes = np.stack(
        [terra[resolved & (codes == identity)].mean(axis=0) for identity in identities]
    )
    slot = {identity: index for index, identity in enumerate(identities)}
    assignment = np.where(
        resolved, np.array([slot.get(code, UNASSIGNED) for code in codes]), UNASSIGNED
    )
    return l2_normalize(prototypes), _split_by_sample(dataset.base, assignment)


def fit_cluster_prototypes(
    datasets, num_prototypes: int, seed: int, max_cells: int | None = 200_000
) -> tuple[np.ndarray, list[dict[str, np.ndarray]]]:
    """Fit unsupervised TERRA clusters as a label-free prototype control.

    Fitted across every training section jointly, so the clusters follow the
    organ mix in the data rather than a taxonomy borrowed from one tissue.
    """

    datasets = list(datasets)
    model, _ = fit_prototypes(
        np.concatenate([_stacked_terra(d) for d in datasets]),
        num_prototypes,
        max_cells,
        seed,
    )
    assignments = [
        _split_by_sample(d, assign_prototypes(model, _stacked_terra(d)).astype(np.int64))
        for d in datasets
    ]
    return l2_normalize(model.cluster_centers_), assignments


def shrink_targets(
    terra: np.ndarray,
    prototypes: np.ndarray,
    assignment: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Interpolate each target toward its prototype and renormalize."""

    if not 0 <= alpha < 1:
        raise ValueError("alpha must be in [0, 1); alpha=1 collapses the diagonal")
    if len(terra) != len(assignment):
        raise ValueError("targets and assignments must have the same length")
    shrunk = l2_normalize(terra)
    assigned = assignment != UNASSIGNED
    if not assigned.any() or alpha == 0:
        return shrunk
    rows = np.flatnonzero(assigned)
    shrunk[rows] = (1.0 - alpha) * shrunk[rows] + alpha * prototypes[assignment[rows]]
    return l2_normalize(shrunk)


def apply_shrinkage(dataset, prototypes, assignments, alpha: float) -> dict:
    """Rewrite a dataset's TERRA targets in place and report coverage.

    Mutates the training dataset only. Validation targets are deliberately left
    raw so the selection metric keeps the same meaning at every ``alpha``.
    """

    targets = dataset.sample_terra
    covered = total = 0
    for sample_id, values in targets.items():
        assignment = assignments[sample_id]
        values[:] = shrink_targets(values, prototypes, assignment, alpha)
        covered += int((assignment != UNASSIGNED).sum())
        total += len(assignment)
    return {
        "alpha": float(alpha),
        "n_prototypes": int(len(prototypes)),
        "assigned_cells": covered,
        "total_cells": total,
        "assigned_fraction": covered / total if total else 0.0,
    }


def configure_target_shrinkage(cfg, data) -> dict | None:
    """Fit prototypes and rewrite the training targets, returning provenance.

    ``identity`` prototypes exist only for the labelled breast sections, so that
    arm reaches breast batches alone. ``cluster`` prototypes are label-free and
    cover every training section, which is the only route to the cross-organ and
    OOD benchmarks. Validation targets are never touched.
    """

    settings = cfg["hierarchy"].get("target_shrinkage") or {}
    if not settings.get("enabled", False):
        return None
    alpha = float(settings["alpha"])
    source = settings.get("prototypes", "identity")
    if source == "identity":
        prototypes, assignments = fit_identity_prototypes(data.breast_train)
        applied = [(data.breast_train.base, assignments)]
    elif source == "cluster":
        datasets = [data.base_train, data.breast_train.base]
        prototypes, assignments = fit_cluster_prototypes(
            datasets,
            int(settings.get("num_prototypes", 256)),
            int(settings.get("seed", 0)),
            settings.get("max_cells_for_fitting", 200_000),
        )
        applied = list(zip(datasets, assignments))
    else:
        raise ValueError(f"unknown prototype source {source!r}")

    coverage = [apply_shrinkage(ds, prototypes, a, alpha) for ds, a in applied]
    assigned = sum(entry["assigned_cells"] for entry in coverage)
    reached = sum(entry["total_cells"] for entry in coverage)
    training_cells = len(data.base_train) + len(data.breast_train)
    record = {
        "prototypes": source,
        "alpha": alpha,
        "n_prototypes": int(len(prototypes)),
        "shrunk_cells": assigned,
        "training_cells": training_cells,
        # Share of ALL training cells whose target moved. This is the coverage
        # ceiling: identity prototypes exist only for the breast sections.
        "shrunk_fraction_of_training": assigned / training_cells if training_cells else 0.0,
        # Share of the cells this arm reaches that got a prototype at all.
        "assigned_fraction_of_reached": assigned / reached if reached else 0.0,
    }
    print(f"target shrinkage | {record}")
    return record


def _stacked_terra(dataset) -> np.ndarray:
    """Concatenate a dataset's TERRA targets in dataset row order."""

    return np.concatenate([dataset.sample_terra[name] for name in dataset.sample_ids])


def _split_by_sample(dataset, values: np.ndarray) -> dict[str, np.ndarray]:
    """Cut a dataset-length vector back into per-sample pieces."""

    pieces, start = {}, 0
    for sample_id in dataset.sample_ids:
        stop = start + len(dataset.sample_cell_ids[sample_id])
        pieces[sample_id] = values[start:stop]
        start = stop
    if start != len(values):
        raise ValueError("per-sample lengths do not cover the dataset")
    return pieces
