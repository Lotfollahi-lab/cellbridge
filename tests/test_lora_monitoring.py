import math

import pytest

from terra_multi.lora.monitoring import CheckpointSelector, Criterion, MetricWindow


def _window():
    return MetricWindow(
        gate_key="hierarchy_annotated_batch", gated_prefixes=("hierarchy_", "negative_")
    )


def test_ungated_metrics_average_over_every_step():
    window = _window()
    window.add({"loss_total": 1.0, "hierarchy_annotated_batch": 0.0})
    window.add({"loss_total": 3.0, "hierarchy_annotated_batch": 0.0})

    assert window.mean()["loss_total"] == pytest.approx(2.0)


def test_gated_metrics_ignore_the_steps_they_do_not_apply_to():
    window = _window()
    # Three unlabelled steps zero-fill the hierarchy metrics; one labelled step
    # carries the real value. Averaging over all four would report 0.1.
    for _ in range(3):
        window.add({"hierarchy_violation_rate": 0.0, "hierarchy_annotated_batch": 0.0})
    window.add({"hierarchy_violation_rate": 0.4, "hierarchy_annotated_batch": 1.0})

    averaged = window.mean()

    assert averaged["hierarchy_violation_rate"] == pytest.approx(0.4)
    assert averaged["hierarchy_annotated_batch"] == pytest.approx(0.25)
    assert averaged["window_steps"] == 4


def test_gated_metric_is_nan_when_the_gate_never_opens():
    window = _window()
    window.add({"hierarchy_violation_rate": 0.0, "hierarchy_annotated_batch": 0.0})

    assert math.isnan(window.mean()["hierarchy_violation_rate"])


def test_reset_clears_the_window():
    window = _window()
    window.add({"loss_total": 5.0, "hierarchy_annotated_batch": 1.0})
    window.reset()

    assert window.steps == 0
    with pytest.raises(ValueError):
        window.mean()


def test_each_criterion_keeps_its_own_best_step(tmp_path):
    selector = CheckpointSelector(
        [
            Criterion("val/loss", "min", tmp_path / "best.pt"),
            Criterion("val/f1", "max", tmp_path / "best_probe.pt"),
        ],
        patience=5,
    )

    selector.update({"val/loss": 2.0, "val/f1": 0.1}, step=1000)
    selector.update({"val/loss": 1.0, "val/f1": 0.05}, step=2000)
    improved = selector.update({"val/loss": 1.5, "val/f1": 0.3}, step=3000)

    summary = selector.summary()
    assert summary["val/loss"]["best_step"] == 2000
    assert summary["val/f1"]["best_step"] == 3000
    assert [criterion.name for criterion in improved] == ["val/f1"]


def test_training_stops_only_once_every_criterion_is_stale(tmp_path):
    selector = CheckpointSelector(
        [
            Criterion("val/loss", "min", tmp_path / "best.pt"),
            Criterion("val/f1", "max", tmp_path / "best_probe.pt"),
        ],
        patience=2,
    )

    selector.update({"val/loss": 1.0, "val/f1": 0.1}, step=1000)
    # The loss goes stale while the probe keeps improving.
    selector.update({"val/loss": 2.0, "val/f1": 0.2}, step=2000)
    selector.update({"val/loss": 2.0, "val/f1": 0.3}, step=3000)
    assert not selector.should_stop

    selector.update({"val/loss": 2.0, "val/f1": 0.1}, step=4000)
    selector.update({"val/loss": 2.0, "val/f1": 0.1}, step=5000)
    assert selector.should_stop


def test_missing_metric_is_an_error(tmp_path):
    selector = CheckpointSelector(
        [Criterion("val/loss", "min", tmp_path / "best.pt")], patience=2
    )

    with pytest.raises(KeyError):
        selector.update({"other": 1.0}, step=1000)


def test_non_finite_values_never_win(tmp_path):
    selector = CheckpointSelector(
        [Criterion("val/loss", "min", tmp_path / "best.pt")], patience=3
    )

    assert selector.update({"val/loss": float("nan")}, step=1000) == []
    assert selector.summary()["val/loss"]["best_step"] == 0


def test_rejects_duplicate_and_invalid_criteria(tmp_path):
    with pytest.raises(ValueError):
        Criterion("val/loss", "lowest", tmp_path / "best.pt")
    with pytest.raises(ValueError):
        CheckpointSelector(
            [
                Criterion("val/loss", "min", tmp_path / "a.pt"),
                Criterion("val/loss", "max", tmp_path / "b.pt"),
            ],
            patience=1,
        )
