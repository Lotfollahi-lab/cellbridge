from pathlib import Path

import numpy as np
import pytest

from cellbridge.baselines import past
from cellbridge.baselines.past_crops import native_center_crop
from cellbridge.baselines.past_model import _image_state_dict

ROOT = Path(__file__).resolve().parents[1]


def _config(tmp_path):
    repository = tmp_path / "past_sc"
    repository.mkdir()
    (repository / "modules.py").write_text("# PAST model\n")
    (repository / "README.md").write_text("# PAST\n")
    checkpoint = repository / "checkpoint-context300.pth"
    checkpoint.write_bytes(b"checkpoint")
    probe = {"methods": ["linear", "knn"], "pca": False}
    return {
        "baseline": {
            "name": "past",
            "embedding_dim": 1024,
            "normalized": False,
            "save_dtype": "float16",
        },
        "model": {
            "repository": str(repository),
            "checkpoint": str(checkpoint),
            "checkpoint_size": checkpoint.stat().st_size,
            "checkpoint_sha256": "test-digest",
            "integration_paths": [str(Path(past.__file__))],
        },
        "crop": {
            "geometry": "cell_center",
            "source_size": 224,
            "padding": "black",
        },
        "output_dir": str(tmp_path / "outputs"),
        "hest": {"probe": probe},
        "phenocell": {"probe": {**probe, "tasks": ["all", "immune"]}},
        "pannuke": {"probe": {**probe, "tasks": ["all"]}},
    }


def test_config_enforces_released_feature_and_crop_contract(tmp_path):
    config = _config(tmp_path)
    past.validate_config(config)

    config["baseline"]["normalized"] = True
    with pytest.raises(ValueError, match="not L2-normalized"):
        past.validate_config(config)


def test_checkpoint_loader_selects_only_image_tower_weights():
    state = _image_state_dict(
        {
            "model": {
                "module.image_encoder.model.layer": 1,
                "module.image_projection.layer": 2,
                "module.cell_encoder.model.layer": 3,
            }
        }
    )
    assert state == {"layer": 1}

    with pytest.raises(ValueError, match="image_encoder.model"):
        _image_state_dict({"model": {"cell_encoder.model.layer": 1}})


def test_native_crop_matches_past_black_padding_and_integer_centering():
    image = np.full((4, 4, 3), 7, dtype=np.uint8)
    crop = np.asarray(native_center_crop(image, 0.9, 0.9, size=4))

    assert crop.shape == (4, 4, 3)
    assert np.all(crop[:2] == 0)
    assert np.all(crop[:, :2] == 0)
    assert np.all(crop[2:, 2:] == 7)


def test_launchers_use_shared_environment_and_parallel_probe_dependency():
    scripts = ROOT / "scripts" / "baselines" / "past"
    runner = (scripts / "run_in_env.sh").read_text()
    submitter = (scripts / "submit_parallel.sh").read_text()

    assert "--with 'timm==1.0.9'" in runner
    assert "scripts/submit_job.sh" in submitter
    assert "for dataset in phenocell pannuke" in submitter
    assert "--dependency" in submitter


def test_probe_cli_accepts_comma_separated_hest_samples():
    args = past.build_parser().parse_args(
        ["probe", "--config", "config.yaml", "--dataset", "hest", "--samples", "A,B"]
    )
    assert past._parse_samples(args.samples) == ["A", "B"]
