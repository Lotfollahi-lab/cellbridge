from pathlib import Path

import pytest

import terra_multi.evaluation.artifacts as artifact_utils
from terra_multi.baselines import spatialwhisperer as sw


ROOT = Path(__file__).resolve().parents[1]


def _config(tmp_path, embedding_dim=sw.RELEASED_PROJECTION_DIM):
    repository = tmp_path / "spatialwhisperer"
    repository.mkdir(parents=True)
    (repository / "pixi.toml").write_text("[workspace]\nname='test'\n")
    return {
        "baseline": {
            "name": "spatialwhisperer",
            "embedding_dim": embedding_dim,
            "normalized": True,
            "save_dtype": "float16",
        },
        "model": {
            "repository": str(repository),
            "hf_repo": "Good-Lab/spatialwhisperer",
        },
        "output_dir": str(tmp_path / "outputs"),
        "hest": {
            "crop": {"target_size": 224, "target_pixel_size_um": 0.5},
            "probe": {"pca": False},
        },
        "phenocell": {
            "crop": {"source_size": 96, "target_size": 224},
            "probe": {"pca": False},
        },
    }


def test_released_checkpoint_contract_is_2048_dimensional(tmp_path):
    sw.validate_config(_config(tmp_path))

    with pytest.raises(ValueError, match="2048-D"):
        sw.validate_config(_config(tmp_path / "wrong", embedding_dim=512))


def test_integration_provenance_tracks_moved_artifact_module():
    paths = tuple(path.resolve() for path in sw._integration_source_paths())

    assert Path(artifact_utils.__file__).resolve() in paths
    assert all(path.is_file() for path in paths)
    assert Path(sw.__file__).with_name("artifacts.py").resolve() not in paths


def test_launchers_use_isolated_pixi_and_uv_probe_environment():
    scripts = ROOT / "scripts" / "baselines" / "spatialwhisperer"
    runner = (scripts / "run_in_env.sh").read_text()
    submitter = (scripts / "submit_parallel.sh").read_text()

    assert "export PYTHONNOUSERSITE=1" in runner
    assert "module load cellgen/uv" in submitter
    assert 'source "${TERRA_ENV}/bin/activate"' in submitter
    assert "conda activate" not in submitter
