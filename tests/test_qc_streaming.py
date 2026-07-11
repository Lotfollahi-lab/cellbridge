"""The streaming QC reconstruction must match the standard one transcript-for-transcript."""

import importlib

import numpy as np
import pandas as pd

_qc = importlib.import_module("terra_multi.preprocess.02_quality_check")
_stream = importlib.import_module("terra_multi.preprocess.02_quality_check_streaming")

_COLS = ["cell_id", "feature_name", "x_location", "y_location", "he_x", "he_y", "qv"]

# Cell 1 straddles the batch split; the last four rows are each dropped for a
# reason (control probe, below-cutoff qv, int -1, string UNASSIGNED).
_ROWS = [
    (1, b"GENEA", 10.0, 10.0, 100.0, 100.0, 30.0),
    (1, b"GENEA", 12.0, 10.0, 102.0, 100.0, 30.0),
    (2, b"GENEB", 50.0, 50.0, 500.0, 500.0, 40.0),
    (10, b"GENEA", 5.0, 5.0, 55.0, 55.0, 25.0),
    # ---- batch boundary ----
    (1, b"GENEB", 11.0, 12.0, 101.0, 102.0, 30.0),
    (2, b"GENEA", 52.0, 50.0, 520.0, 500.0, 40.0),
    (10, b"GENEB", 6.0, 5.0, 60.0, 55.0, 40.0),
    (1, b"NegControlProbe_5", 11.0, 11.0, 101.0, 101.0, 40.0),
    (2, b"GENEB", 60.0, 60.0, 600.0, 600.0, 10.0),
    (-1, b"GENEA", 0.0, 0.0, 0.0, 0.0, 40.0),
    ("UNASSIGNED", b"GENEA", 0.0, 0.0, 0.0, 0.0, 40.0),
]


def _df(rows):
    return pd.DataFrame(rows, columns=_COLS)


def test_streaming_matches_standard():
    ref = _qc.reconstruct_cells(_df(_ROWS), qv_threshold=20)

    # Same transcripts, delivered as two separate batches.
    batches = [_df(_ROWS[:4]), _df(_ROWS[4:])]
    got = _stream.reconstruct_cells_streaming(lambda: iter(batches), qv_threshold=20)

    assert list(got.obs_names) == list(ref.obs_names)
    assert list(got.var_names) == list(ref.var_names)
    np.testing.assert_array_equal(got.X.toarray(), ref.X.toarray())
    np.testing.assert_allclose(got.obsm["spatial"], ref.obsm["spatial"])
    for col in ["he_x", "he_y", "total_counts", "n_genes_by_counts"]:
        np.testing.assert_allclose(
            got.obs[col].to_numpy(dtype=float), ref.obs[col].to_numpy(dtype=float)
        )
