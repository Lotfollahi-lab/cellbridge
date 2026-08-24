"""Tests for linear-probe hyperparameter selection."""

import numpy as np
import pytest

from cellbridge.inference.evaluate import Task
from cellbridge.inference.evaluate import linear


def _rare_class_task():
    return Task(
        features=np.asarray([[-2.0], [0.0], [2.0]], dtype=np.float32),
        labels=np.asarray(["rare", "common", "common"]),
        ids=np.asarray(["0", "1", "2"]),
        train=np.asarray([0, 1, 2]),
        validation=None,
        test=np.asarray([0]),
    )


def test_fixed_c_allows_a_single_training_example_for_one_class():
    fitted = linear.fit(_rare_class_task(), Cs=[1], max_iter=100)

    assert fitted.C == 1.0
    assert np.isnan(fitted.selection_macro_f1)


def test_multiple_c_values_still_require_stratified_cross_validation():
    with pytest.raises(ValueError, match="two examples per class"):
        linear.fit(_rare_class_task(), Cs=[0.1, 1], max_iter=100)
