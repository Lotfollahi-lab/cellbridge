from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from cellbridge.baselines import histoplus as hp
from cellbridge.baselines.histoplus_metrics import aggregate_metrics

ROOT = Path(__file__).resolve().parents[1]


def _config(tmp_path):
    repository = tmp_path / "histoplus"
    repository.mkdir(parents=True)
    (repository / "pyproject.toml").write_text("[project]\nname='histoplus'\n")
    return {
        "baseline": {
            "name": "histoplus",
            "inference_image_size": 784,
            "inference_overlap": 64,
        },
        "model": {"repository": str(repository)},
        "output_dir": str(tmp_path / "outputs"),
        "evaluation": {"match_radius_um": 5},
        "hest": {"model_mpp": 0.25},
        "phenocell": {"model_mpp": 0.5},
        "pannuke": {"model_mpp": 0.25},
    }


def test_config_pins_released_resolution_contracts(tmp_path):
    cfg = _config(tmp_path)
    hp.validate_config(cfg)

    cfg["phenocell"]["model_mpp"] = 0.25
    with pytest.raises(ValueError, match="PhenoCell"):
        hp.validate_config(cfg)


def test_radius_match_maximizes_cardinality_before_distance():
    reference = np.array([[0.0, 0.0], [1.5, 0.0]])
    prediction = np.array([[0.5, 0.0], [0.0, 0.0]])

    matches = hp.radius_match(reference, prediction, radius=1.0)

    assert {(int(r), int(p)) for r, p, _ in matches} == {(0, 1), (1, 0)}


def test_radius_match_uses_minimum_distance_for_equal_cardinality():
    reference = np.array([[0.0, 0.0], [0.9, 0.0]])
    prediction = np.array([[0.4, 0.0]])

    matches = hp.radius_match(reference, prediction, radius=1.0)

    assert matches.shape == (1, 3)
    assert tuple(matches[0, :2].astype(int)) == (0, 0)


def test_end_to_end_score_penalizes_misses_and_false_positives():
    reference = pd.DataFrame(
        {
            "reference_id": ["r0", "r1", "ignored"],
            "x": [0.0, 10.0, 30.0],
            "y": [0.0, 0.0, 0.0],
            "label": ["a", "b", "ignored"],
        }
    )
    predictions = pd.DataFrame(
        {
            "prediction_id": ["p0", "p1", "fp"],
            "x": [0.1, 10.1, 50.0],
            "y": [0.0, 0.0, 0.0],
            "cell_type": ["A", "A", "B"],
        }
    )

    metrics, matches, _ = hp.score_cells(
        reference,
        predictions,
        reference_map={"a": "a", "b": "b", "ignored": None},
        prediction_map={"A": "a", "B": "b"},
        radius_px=1.0,
    )

    assert metrics["coverage"] == 1.0
    assert metrics["n_geometric_matches"] == 2
    assert metrics["matched_macro_f1"] == pytest.approx(1 / 3)
    assert metrics["end_to_end_macro_f1"] == pytest.approx(1 / 3)
    assert matches["prediction_id"].notna().sum() == 2


def test_empty_official_test_field_scores_every_detection_as_a_false_positive():
    # PanNuke keeps official test fields whose metadata lists no cells, so
    # _ood_references yields an empty reference table for them.
    reference = pd.DataFrame(columns=("reference_id", "x", "y", "label", "contour"))
    predictions = pd.DataFrame(
        {
            "prediction_id": ["p0"],
            "x": [5.0],
            "y": [5.0],
            "cell_type": ["Cancer cell"],
            "contour": ['[[4,4],[6,4],[6,6],[4,6]]'],
        }
    )

    metrics, matches, match_indices = hp.score_cells(
        reference,
        predictions,
        reference_map=hp.PANNuke_REFERENCE,
        prediction_map=hp.HISTOPLUS_PANNUKE,
        radius_px=20.0,
    )

    assert metrics["n_reference"] == 0
    assert metrics["n_scorable_reference"] == 0
    assert metrics["n_geometric_matches"] == 0
    assert metrics["coverage"] == 0.0
    assert metrics["detection_recall"] == 0.0
    assert matches.empty
    neoplastic = next(
        row for row in metrics["per_class_end_to_end"] if row["class"] == "Neoplastic"
    )
    assert (neoplastic["tp"], neoplastic["fp"], neoplastic["fn"]) == (0, 1, 0)

    panoptic = hp.panoptic_quality(reference, predictions, match_indices)
    assert (panoptic["panoptic_tp"], panoptic["panoptic_fp"]) == (0, 1)
    assert panoptic["pq"] == 0.0


def test_image_tiles_have_disjoint_ownership_cores():
    image = Image.new("RGB", (1400, 900), "white")
    tiles = hp._image_tiles(image, scale=1.0, inference_size=784, overlap=64)

    cores = {
        (tile["core_x"], tile["core_y"], tile["core_x1"], tile["core_y1"])
        for tile in tiles
    }
    assert cores == {
        (0, 0, 656, 656),
        (656, 0, 1312, 656),
        (1312, 0, 1400, 656),
        (0, 656, 656, 900),
        (656, 656, 1312, 900),
        (1312, 656, 1400, 900),
    }


def test_aggregate_metrics_sums_confusions_and_panoptic_counts():
    common = {
        "n_reference": 2,
        "n_scorable_reference": 2,
        "n_predictions": 2,
        "n_geometric_matches": 1,
        "n_scorable_matches": 1,
        "per_class_matched": [{"class": "a", "tp": 1, "fp": 0, "fn": 0}],
        "per_class_end_to_end": [{"class": "a", "tp": 1, "fp": 0, "fn": 1}],
        "panoptic_tp": 1,
        "panoptic_fp": 1,
        "panoptic_fn": 1,
        "panoptic_iou_sum": 0.75,
    }

    result = aggregate_metrics([common, common], "pannuke")

    assert result["coverage"] == 0.5
    assert result["per_class_end_to_end"][0]["f1"] == pytest.approx(2 / 3)
    assert result["dq"] == 0.5
    assert result["sq"] == 0.75
    assert result["pq"] == 0.375


def test_launchers_use_isolated_uv_environment_and_shared_submission():
    scripts = ROOT / "scripts" / "baselines" / "histoplus"
    runner = (scripts / "run_in_env.sh").read_text()
    submitter = (scripts / "submit_parallel.sh").read_text()

    assert "UV_PROJECT_ENVIRONMENT" in runner
    assert "export PYTHONNOUSERSITE=1" in runner
    assert "--frozen" in runner
    assert "scripts/submit_job.sh" in submitter
    scheduler = submitter.index('"$SUBMIT" submit --job')
    config = submitter.index('"$CONFIG" --', scheduler)
    assert scheduler < config
    assert 'submit "$CONFIG" --job' not in submitter
    assert "infer-hest" in submitter
    assert "infer-ood" in submitter
