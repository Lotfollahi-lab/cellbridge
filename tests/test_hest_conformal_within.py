import numpy as np

from cellbridge.helpers.splits import spatial_split
from cellbridge.hierarchical_conformal.within_slide_data import (
    three_way_spatial_split,
)


def test_three_way_split_preserves_historical_test_stripe():
    cell_ids = np.asarray([f"cell-{index}" for index in range(100)])
    coordinates = np.column_stack(
        [np.zeros(100), np.arange(100, dtype=float)]
    )
    historical_fit, historical_test = spatial_split(
        cell_ids, coordinates, 0.2, "y"
    )
    train, calibration, test = three_way_spatial_split(
        cell_ids, coordinates, 0.2, 0.2, "y"
    )

    np.testing.assert_array_equal(test, historical_test)
    assert len(train) == 64
    assert len(calibration) == 16
    assert not set(train) & set(calibration)
    assert not set(train) & set(test)
    assert not set(calibration) & set(test)
    assert set(np.concatenate([train, calibration])) == set(historical_fit)
