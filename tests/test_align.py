"""
This test plants a per-cell marker, shuffles three arrays into different orders with disjoint
extra IDs, and checks that alignment recovers the intersection in one consistent
order.
"""

import numpy as np

from cellbridge.helpers.io import align_by_cell_id

# A distinct, ID-tied marker value for each shared cell (identical across arrays).
MARKER = {"c1": 1.0, "c2": 2.0, "c3": 3.0, "c4": 4.0}


def _make(order, extra):
    ids = list(order) + list(extra)
    col0 = [MARKER.get(c, -99.0) for c in ids]
    col1 = [10 + i for i in range(len(ids))]  # positional, array-specific
    return np.array([col0, col1], dtype=float).T, np.array(ids)


def test_align_by_cell_id():
    # Same four shared cells, three different orders, each with disjoint extras.
    a = _make(["c2", "c4", "c1", "c3"], ["x1"])
    b = _make(["c3", "c1", "c4", "c2"], ["y1", "y2"])
    c = _make(["c1", "c3", "c2", "c4"], ["z1"])

    aligned, shared_ids = align_by_cell_id([a, b, c])

    # Output IDs are exactly the intersection, in one consistent (sorted) order.
    assert list(shared_ids) == ["c1", "c2", "c3", "c4"]

    # Row i corresponds to the same cell (shared_ids[i]) in every output: the
    # ID-tied marker column must equal MARKER[shared_ids[i]] for all three.
    expected = np.array([MARKER[s] for s in shared_ids])
    for arr in aligned:
        assert arr.shape[0] == len(shared_ids)
        np.testing.assert_array_equal(arr[:, 0], expected)

    # The planted marker for a specific cell lands on the same row everywhere.
    row_c2 = list(shared_ids).index("c2")
    for arr in aligned:
        assert arr[row_c2, 0] == MARKER["c2"]

    assert not np.array_equal(a[0][:, 0], b[0][:, 0])
    marker_cols = np.stack([arr[:, 0] for arr in aligned])
    assert np.all(marker_cols == marker_cols[0])
