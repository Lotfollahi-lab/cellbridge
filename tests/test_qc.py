"""QC reconstruction filters transcripts and aggregates per-cell counts."""

import importlib

import numpy as np
import pandas as pd

reconstruct_cells = importlib.import_module(
    "terra_multi.pipeline.02_quality_check"
).reconstruct_cells

# (cell_id, feature_name, x, y, he_x, he_y, qv). Cell 1 keeps GENEA x2 + GENEB;
# cell 2 keeps GENEA only. The last three rows are each dropped for a reason.
_ROWS = [
    (1, b"GENEA", 10.0, 10.0, 100.0, 100.0, 30.0),
    (1, b"GENEA", 12.0, 10.0, 102.0, 100.0, 30.0),
    (1, b"GENEB", 11.0, 12.0, 101.0, 102.0, 30.0),
    (2, b"GENEA", 50.0, 50.0, 500.0, 500.0, 40.0),
    (1, b"NegControlProbe_5", 11.0, 11.0, 101.0, 101.0, 40.0),  # control probe
    (2, b"GENEB", 60.0, 60.0, 600.0, 600.0, 10.0),              # qv below cutoff
    (-1, b"GENEA", 0.0, 0.0, 0.0, 0.0, 40.0),                   # unassigned (int -1)
    ("UNASSIGNED", b"GENEA", 0.0, 0.0, 0.0, 0.0, 40.0),         # unassigned (string)
]
_COLS = ["cell_id", "feature_name", "x_location", "y_location", "he_x", "he_y", "qv"]


def test_reconstruct_cells():
    adata = reconstruct_cells(pd.DataFrame(_ROWS, columns=_COLS), qv_threshold=20)

    assert list(adata.obs_names) == ["1", "2"]
    assert list(adata.var_names) == ["GENEA", "GENEB"]  # control gene excluded

    x = pd.DataFrame(
        adata.X.toarray(), index=adata.obs_names, columns=adata.var_names
    )
    assert x.loc["1", "GENEA"] == 2 and x.loc["1", "GENEB"] == 1
    assert x.loc["2", "GENEA"] == 1 and x.loc["2", "GENEB"] == 0
    assert adata.obs.loc["1", "total_counts"] == 3

    # Centroids are the mean over each cell's kept transcripts, in both systems.
    np.testing.assert_allclose(adata.obsm["spatial"][0], [11.0, 32 / 3])
    np.testing.assert_allclose(
        adata.obs.loc["1", ["he_x", "he_y"]].to_numpy(dtype=float), [101.0, 302 / 3]
    )
    np.testing.assert_allclose(adata.obsm["spatial"][1], [50.0, 50.0])
