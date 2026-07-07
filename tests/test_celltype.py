"""transfer_labels: polygon-containment majority vote, nearest-centroid fallback
within the cutoff, and Unassigned beyond it. px_um=1 so distances are in pixels.
"""

import geopandas as gpd
import numpy as np
from shapely.geometry import Point, box

from terra_multi.helpers.labels import transfer_labels

# Three Xenium cells: A holds nuclei, B is empty but near A, C is empty and far.
CELLS = gpd.GeoDataFrame(
    {"cell_id": ["A", "B", "C"]},
    geometry=[box(0, 0, 10, 10), box(20, 0, 30, 10), box(100, 100, 110, 110)],
)
# Nuclei only inside A: majority Neoplastic (2) over Connective (1); (8,8) is nearest to B.
NUCLEI = gpd.GeoDataFrame(
    {"cellvit_class": ["Neoplastic", "Neoplastic", "Connective"]},
    geometry=[Point(2, 2), Point(3, 3), Point(8, 8)],
)
HE = np.array([[5.0, 5.0], [25.0, 5.0], [105.0, 105.0]])  # cell centroids (A, B, C)


def test_transfer_labels():
    labels = transfer_labels(
        ["A", "B", "C"], HE, CELLS, NUCLEI, fallback_max_um=20, px_um=1.0
    ).set_index("cell_id")

    # A: majority class among the 3 contained nuclei.
    assert labels.loc["A", "cell_type"] == "Neoplastic"
    assert labels.loc["A", "method"] == "polygon" and labels.loc["A", "n_nuclei"] == 3

    # B: no nucleus inside -> nearest centroid (8,8) is within the 20 px cutoff.
    assert labels.loc["B", "method"] == "fallback"
    assert labels.loc["B", "cell_type"] == "Connective"
    np.testing.assert_allclose(labels.loc["B", "fallback_dist_um"], np.hypot(17, 3))

    # C: nearest centroid is far beyond the cutoff -> Unassigned.
    assert labels.loc["C", "method"] == "fallback"
    assert labels.loc["C", "cell_type"] == "Unassigned"
