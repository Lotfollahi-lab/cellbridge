import numpy as np

from cellbridge.inference.conformal.prediction import calibrate_split


def test_calibrate_split_accepts_float32_roundoff():
    probabilities = np.full((2, 5), 0.20000005, dtype=np.float32)

    q_hat = calibrate_split(
        probabilities,
        labels=np.array(["a", "b"]),
        classes=np.array(["a", "b", "c", "d", "e"]),
        alpha=0.5,
    )

    assert np.isclose(q_hat, 0.8)
