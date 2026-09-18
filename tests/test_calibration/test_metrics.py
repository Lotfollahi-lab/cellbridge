import numpy as np
import pytest
from sklearn.metrics import accuracy_score, f1_score, log_loss

from cellbridge.inference.calibration.metrics import fit_temperature, log_probabilities, statistics
from cellbridge.inference.conformal.conformal_ood import _indices
from cellbridge.inference.evaluate import Task


def test_metrics_match_independent_calculations():
    p = np.array([[.83, .17], [.63, .37], [.07, .93], [.27, .73]])
    y = np.array([0, 1, 1, 0])
    point, draws, bins = statistics(np.log(p), y, ['a', 'a', 'b', 'b'], n_resamples=30)
    assert point['accuracy'] == accuracy_score(y, p.argmax(1))
    assert point['macro_f1'] == f1_score(y, p.argmax(1), average='macro')
    assert point['nll'] == pytest.approx(log_loss(y, p))
    assert point['brier'] == pytest.approx(np.square(p - np.eye(2)[y]).sum(1).mean())
    assert point['ece_10'] == pytest.approx((.17 + .63 + .07 + .73) / 4)
    assert len(draws) == 30
    assert bins.query('n_bins == 15')['n'].sum() == 4


def test_temperature_improves_heldout_nll_and_preserves_argmax():
    logits = np.tile([[0., 5.]], (100, 1))
    y = np.array([0] * 30 + [1] * 70)
    result = fit_temperature(logits, y)
    assert result['temperature'] > 1
    assert result['validation_nll_scaled'] < result['validation_nll_raw']
    scaled = np.exp(log_probabilities(logits, result['temperature']))
    assert scaled[:, 1].mean() == pytest.approx(.7, abs=1e-5)
    np.testing.assert_array_equal(scaled.argmax(1), logits.argmax(1))


def test_group_bootstrap_matches_explicit_resampling():
    logits = np.array([[3., 1., 0.], [0., 2., 1.], [1., 0., 2.], [1., 2., 0.]])
    y = np.array([0, 2, 2, 1])
    groups = np.array(['a', 'a', 'b', 'c'])
    _, draws, _ = statistics(logits, y, groups, n_resamples=10, seed=7)
    weights = np.random.default_rng(7).multinomial(3, [1/3] * 3, size=10)
    for index, weight in enumerate(weights):
        selected = np.repeat(np.arange(4), weight[np.searchsorted(np.unique(groups), groups)])
        point, _, _ = statistics(logits[selected], y[selected], np.repeat('one', len(selected)), n_resamples=0)
        for name in point:
            assert draws.iloc[index][name] == pytest.approx(point[name])


def test_extreme_logits_and_single_group():
    logits = np.array([[10000., -10000.], [-10000., 10000.]])
    point, draws, _ = statistics(logits, [0, 1], ['one', 'one'])
    assert point['ece_15'] == 0
    assert point['nll'] == 0
    assert draws.empty
    assert np.isfinite(log_probabilities(logits)).all()


def test_rejects_overlapping_temperature_and_training_rows():
    task = Task(np.zeros((6, 2)), np.array(['a', 'b'] * 3), np.arange(6),
                np.array([0, 1, 2]), np.array([2, 3]), np.array([4, 5]))
    with pytest.raises(ValueError, match='disjoint'):
        _indices(task)
