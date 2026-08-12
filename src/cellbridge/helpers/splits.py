"""Deterministic spatial train/test split."""

import numpy as np

_AXIS = {"x": 0, "y": 1, 0: 0, 1: 1}


def spatial_split(cell_ids, coords, test_frac, axis):
    """Hold out the top ``test_frac`` of cells along ``axis`` as the test set.

    ``coords`` is an ``(N, 2)`` array row-aligned to ``cell_ids``; ``axis`` is
    ``"x"``/``"y"`` (or ``0``/``1``). "Top" means the largest coordinate values,
    so train and test occupy different regions of the tissue -- a spatial split
    avoids the leakage a random split suffers from spatial autocorrelation.

    Deterministic. Returns ``(train_idx, test_idx)`` as sorted integer arrays
    indexing into the ``cell_ids`` row order.
    """
    coords = np.asarray(coords)
    n = coords.shape[0]
    if len(cell_ids) != n:
        raise ValueError(f"cell_ids ({len(cell_ids)}) and coords ({n}) length mismatch")
    if axis not in _AXIS:
        raise ValueError(f"axis must be one of {sorted(_AXIS, key=str)}, got {axis!r}")
    if not 0.0 < test_frac < 1.0:
        raise ValueError(f"test_frac must be in (0, 1), got {test_frac}")

    values = coords[:, _AXIS[axis]]
    n_test = int(round(test_frac * n))
    n_test = min(max(n_test, 1), n - 1)
    order = np.argsort(values, kind="stable")  # ascending coordinate
    test_idx = np.sort(order[n - n_test :])
    train_idx = np.sort(order[: n - n_test])
    return train_idx, test_idx
