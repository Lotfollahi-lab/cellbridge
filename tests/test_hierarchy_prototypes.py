import numpy as np
import pytest
import torch

from cellbridge.hierarchical.prototypes import (
    UNASSIGNED,
    fit_identity_prototypes,
    shrink_targets,
)


class _Base:
    """Minimal stand-in for HESTCellDataset's target/ID accessors."""

    def __init__(self, terra_by_sample):
        self.sample_ids = list(terra_by_sample)
        self._terra = {k: np.asarray(v, dtype=np.float32) for k, v in terra_by_sample.items()}

    @property
    def sample_terra(self):
        return self._terra

    @property
    def sample_cell_ids(self):
        return {k: np.arange(len(v)) for k, v in self._terra.items()}


class _Hierarchy:
    def __init__(self, base, codes, depth, usable):
        self.base = base
        self.levels = torch.tensor([[0, 0, 0, c] for c in codes])
        self.depth = torch.tensor(depth)
        self.usable = torch.tensor(usable)


def _dataset():
    # Two identities, two cells each, plus one partial cell with no identity.
    terra = np.array(
        [[1.0, 0.0], [0.8, 0.6], [0.0, 1.0], [0.6, 0.8], [-1.0, 0.0]], dtype=np.float32
    )
    base = _Base({"S1": terra})
    return _Hierarchy(base, [0, 0, 1, 1, 0], [3, 3, 3, 3, 2], [True] * 5)


def test_prototypes_are_unit_norm_and_one_per_identity():
    prototypes, assignments = fit_identity_prototypes(_dataset())

    assert prototypes.shape == (2, 2)
    assert np.allclose(np.linalg.norm(prototypes, axis=1), 1.0, atol=1e-5)
    # The depth-2 cell cannot be assigned an identity.
    assert assignments["S1"].tolist() == [0, 0, 1, 1, UNASSIGNED]


def test_shrinkage_moves_targets_toward_their_prototype():
    dataset = _dataset()
    prototypes, assignments = fit_identity_prototypes(dataset)
    terra = dataset.base.sample_terra["S1"]
    assignment = assignments["S1"]

    shrunk = shrink_targets(terra, prototypes, assignment, alpha=0.5)

    before = np.sum(terra[0] / np.linalg.norm(terra[0]) * prototypes[0])
    after = np.sum(shrunk[0] * prototypes[0])
    assert after > before
    assert np.allclose(np.linalg.norm(shrunk, axis=1), 1.0, atol=1e-5)


def test_alpha_zero_only_normalizes():
    dataset = _dataset()
    prototypes, assignments = fit_identity_prototypes(dataset)
    terra = dataset.base.sample_terra["S1"]

    shrunk = shrink_targets(terra, prototypes, assignments["S1"], alpha=0.0)

    expected = terra / np.linalg.norm(terra, axis=1, keepdims=True)
    assert np.allclose(shrunk, expected, atol=1e-6)


def test_unassigned_cells_are_left_alone():
    dataset = _dataset()
    prototypes, assignments = fit_identity_prototypes(dataset)
    terra = dataset.base.sample_terra["S1"]

    shrunk = shrink_targets(terra, prototypes, assignments["S1"], alpha=0.9)

    expected = terra[-1] / np.linalg.norm(terra[-1])
    assert np.allclose(shrunk[-1], expected, atol=1e-6)


def test_alpha_one_is_rejected_because_it_collapses_the_diagonal():
    dataset = _dataset()
    prototypes, assignments = fit_identity_prototypes(dataset)

    with pytest.raises(ValueError, match="collapses the diagonal"):
        shrink_targets(
            dataset.base.sample_terra["S1"], prototypes, assignments["S1"], alpha=1.0
        )


def test_shrinkage_shrinks_within_identity_spread():
    """Same-identity targets must get closer; that is the whole mechanism."""

    dataset = _dataset()
    prototypes, assignments = fit_identity_prototypes(dataset)
    terra = dataset.base.sample_terra["S1"]
    normalized = terra / np.linalg.norm(terra, axis=1, keepdims=True)

    shrunk = shrink_targets(terra, prototypes, assignments["S1"], alpha=0.5)

    assert np.dot(shrunk[0], shrunk[1]) > np.dot(normalized[0], normalized[1])
