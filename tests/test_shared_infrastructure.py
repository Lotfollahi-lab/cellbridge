from argparse import Namespace

import numpy as np
import pandas as pd
import pytest

from terra_multi.cluster import render_job
from terra_multi.cohorts import cohort_in_split, samples_in_split
from terra_multi.config import load_config
from terra_multi.lora.features import feature_file_is_current, write_feature_file


def test_config_expands_environment_and_merges_overlay(tmp_path, monkeypatch):
    base = tmp_path / "base.yaml"
    base.write_text(
        "data_root: ${TEST_DATA_ROOT}/hest\n"
        "settings:\n  keep: true\n  value: base\n"
    )
    overlay = tmp_path / "local.yaml"
    overlay.write_text("settings:\n  value: local\n")
    monkeypatch.setenv("TEST_DATA_ROOT", str(tmp_path / "data"))

    config = load_config(base, overlay=overlay)

    assert config["data_root"] == str(tmp_path / "data" / "hest")
    assert config["settings"] == {"keep": True, "value": "local"}


def test_cohort_is_ordered_unique_and_held_out(tmp_path):
    splits = tmp_path / "splits.csv"
    cohort = tmp_path / "fast.csv"
    pd.DataFrame(
        {"sample_id": ["train", "B", "A"], "split": ["train", "test", "test"]}
    ).to_csv(splits, index=False)
    pd.DataFrame({"sample_id": ["A", "B"]}).to_csv(cohort, index=False)

    assert samples_in_split(splits, "test") == ["B", "A"]
    assert cohort_in_split(cohort, splits, "test") == ["A", "B"]

    pd.DataFrame({"sample_id": ["A", "A"]}).to_csv(cohort, index=False)
    with pytest.raises(ValueError, match="duplicate"):
        cohort_in_split(cohort, splits, "test")


def test_lora_artifacts_use_shared_alignment_contract(tmp_path):
    path = tmp_path / "features.npz"
    features = {"lora_adapted": np.ones((2, 3), dtype=np.float32)}
    write_feature_file(path, ["a", "b"], features, "sig")

    assert feature_file_is_current(
        path,
        "sig",
        required_keys=("lora_adapted",),
        expected_cells=2,
        expected_dims={"lora_adapted": 3},
    )
    assert not feature_file_is_current(
        path, "sig", required_keys=("lora_adapted",), expected_cells=3
    )


def test_lsf_render_uses_resolved_config_and_one_environment(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text(f"output_dir: {tmp_path / 'output'}\n")
    args = Namespace(
        config=str(config),
        output_key="output_dir",
        log_subdir="logs",
        log_prefix="train",
        job="run",
        cores=4,
        queue="training-normal",
        gpu="-",
        group="group",
        memory=16,
        dependency="done(7)",
        environment="none",
        command=["python3", "-m", "terra_multi.jepa.train", "--config", str(config)],
    )

    job = render_job(args)

    assert '#BSUB -gpu "-"' in job
    assert '#BSUB -w "done(7)"' in job
    assert f"#BSUB -o {tmp_path / 'output/logs/train_%J.out'}" in job
    assert "PYTHONPATH=src python3 -m terra_multi.jepa.train" in job
