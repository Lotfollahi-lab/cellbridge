from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cellbridge.baselines import volta
from cellbridge.baselines.volta_crops import (
    crop_box,
    instance_bounds,
    locate_instance_boxes,
    square_bounds,
)
from cellbridge.baselines.volta_hest import _merge_bounds
from cellbridge.baselines.volta_model import _teacher_state_dict

ROOT = Path(__file__).resolve().parents[1]


def _config(tmp_path):
    repository = tmp_path / "VOLTA"
    backbone = repository / "backbones" / "resnet" / "model.py"
    backbone.parent.mkdir(parents=True)
    backbone.write_text("# released backbone\n")
    checkpoint = repository / "checkpoint.pth.tar"
    checkpoint.write_bytes(b"checkpoint")
    probe = {"methods": ["linear", "knn"], "pca": False}
    return {
        "baseline": {
            "name": "volta_ovarian",
            "embedding_dim": 512,
            "normalized": False,
            "save_dtype": "float16",
        },
        "model": {
            "repository": str(repository),
            "checkpoint": str(checkpoint),
            "checkpoint_domain": "ovarian",
            "checkpoint_size": checkpoint.stat().st_size,
        },
        "crop": {"scale": 2.0, "padding": "white"},
        "output_dir": str(tmp_path / "outputs"),
        "hest": {"probe": probe},
        "phenocell": {"probe": {**probe, "tasks": ["all", "immune"]}},
        "pannuke": {"probe": {**probe, "tasks": ["all"]}},
    }


def test_config_enforces_released_feature_contract(tmp_path):
    config = _config(tmp_path)
    volta.validate_config(config)

    config["baseline"]["normalized"] = True
    with pytest.raises(ValueError, match="not L2-normalized"):
        volta.validate_config(config)


def test_teacher_state_uses_only_momentum_encoder():
    state = _teacher_state_dict(
        {
            "state_dict": {
                "module.encoder_q.layer": 1,
                "module.encoder_k.layer": 2,
            }
        }
    )
    assert state == {"layer": 2}


def test_square_crop_has_white_boundary_padding():
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    assert square_bounds((0, 0, 2, 1), scale=2) == (-1, -2, 3, 2)

    crop = np.asarray(crop_box(image, (0, 0, 2, 1), scale=2))
    assert crop.shape == (4, 4, 3)
    assert np.all(crop[:2] == 255)
    assert np.all(crop[2:, 1:3] == 0)


def test_duplicate_cell_geometry_bounds_are_merged():
    bounds = pd.DataFrame(
        {
            "minx": [2, 1, 10],
            "miny": [4, 3, 20],
            "maxx": [6, 5, 12],
            "maxy": [8, 9, 22],
        }
    )
    merged = _merge_bounds(bounds, ["cell-a", "cell-a", "cell-b"])
    assert merged.loc["cell-a"].tolist() == [1, 3, 6, 9]


def test_instance_boxes_follow_tile_geometry_and_can_skip_missing():
    mask = np.zeros((4, 4), dtype=np.int32)
    mask[1:3, 2:4] = 7
    entries = {
        0: {"tile_name": "sample_TILE_0", "instance_mask": mask},
    }
    metadata = pd.DataFrame(
        {
            "tile_id": ["sample_TILE_0", "sample_TILE_0"],
            "cell_id": [7, 8],
        }
    )
    assert instance_bounds(mask)[7] == (2, 1, 4, 3)
    located = locate_instance_boxes(
        metadata, entries, image_shape=(4, 4), tile_size=4, skip_missing=True
    )
    assert located["cell_id"].tolist() == [7]
    assert located[["x0", "y0", "x1", "y1"]].iloc[0].tolist() == [2, 1, 4, 3]


def test_launchers_install_only_small_runtime_adapters():
    runner = (ROOT / "scripts/baselines/volta/run_in_env.sh").read_text()
    submitter = (ROOT / "scripts/baselines/volta/submit_parallel.sh").read_text()

    assert "--with lmdb --with opencv-python-headless" in runner
    assert "scripts/submit_job.sh" in submitter
    assert '"$SUBMIT" submit --job' in submitter
    assert 'submit "$CONFIG" --job' not in submitter
    assert "extract-pannuke" not in submitter
    assert "for dataset in phenocell pannuke" in submitter


def test_probe_cli_accepts_comma_separated_hest_samples():
    args = volta.build_parser().parse_args(
        ["probe", "--config", "config.yaml", "--dataset", "hest", "--samples", "A,B"]
    )

    assert volta._parse_samples(args.samples) == ["A", "B"]
