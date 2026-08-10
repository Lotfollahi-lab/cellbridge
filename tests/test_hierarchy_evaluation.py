from pathlib import Path

import pandas as pd

from cellbridge.lora import evaluate as lora_evaluate
from cellbridge.lora.hest import evaluation_cohorts, evaluation_samples
from cellbridge.lora.summary import write_evaluation_summary


def _split_files(tmp_path):
    base = tmp_path / "base.csv"
    pd.DataFrame(
        {
            "sample_id": ["base_train", "base_val", "A", "B", "C", "X"],
            "split": [
                "distill_train",
                "distill_val",
                "label_test",
                "label_test",
                "label_test",
                "label_test",
            ],
        }
    ).to_csv(base, index=False)
    manifest = tmp_path / "phenotype.csv"
    pd.DataFrame(
        {
            "sample_id": ["A", "B", "C", "X"],
            "organ": ["breast", "breast", "breast", "pancreas"],
            "patient_id": ["p1", "p2", "p3", ""],
            "role": [
                "breast_train",
                "breast_val",
                "breast_test",
                "cross_organ_test",
            ],
        }
    ).to_csv(manifest, index=False)
    return base, manifest


def _config(tmp_path):
    base, manifest = _split_files(tmp_path)
    return {
        "output_dir": str(tmp_path / "run"),
        "splits_path": str(base),
        "hierarchy": {
            "enabled": True,
            "split_manifest": str(manifest),
        },
        "eval": {},
    }


def test_hierarchy_evaluation_cohorts_remain_separate(tmp_path):
    assert evaluation_cohorts(_config(tmp_path)) == {
        "breast_val": ["B"],
        "breast_test": ["C"],
        "cross_organ": ["X"],
    }


def test_hierarchy_evaluation_can_select_only_held_out_breast_test(tmp_path):
    cfg = _config(tmp_path)
    cfg["eval"]["hest_cohorts"] = ["breast_test"]

    assert evaluation_cohorts(cfg) == {"breast_test": ["C"]}


def test_breast_training_cohort_is_available_only_when_requested(tmp_path):
    assert evaluation_samples(_config(tmp_path), "breast_train") == ["A"]


def test_probe_only_routes_each_hierarchy_cohort_to_its_own_directory(
    tmp_path, monkeypatch
):
    cfg = _config(tmp_path)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"output_dir: {cfg['output_dir']}\n"
        f"splits_path: {cfg['splits_path']}\n"
        "hierarchy:\n"
        "  enabled: true\n"
        f"  split_manifest: {cfg['hierarchy']['split_manifest']}\n"
        "eval: {}\n"
    )
    calls = []

    def fake_probe(_cfg, output_dir, samples=None, cohort="label_test_fast"):
        calls.append((Path(output_dir).name, samples, cohort))

    monkeypatch.setattr(lora_evaluate, "probe_hest", fake_probe)
    lora_evaluate.main(["--config", str(config_path), "--probe-only"])

    assert calls == [
        ("hest_breast_val", None, "breast_val"),
        ("hest_breast_test", None, "breast_test"),
        ("hest_cross_organ", None, "cross_organ"),
    ]


def test_summary_reports_hest_cohorts_as_distinct_tasks(tmp_path):
    metrics = {
        cohort: pd.DataFrame(
            {
                "arm": ["uni_original", "lora_projected"],
                "macro_f1": [0.1, value],
                "balanced_accuracy": [0.2, value + 0.1],
            }
        )
        for cohort, value in {
            "breast_val": 0.4,
            "breast_test": 0.5,
            "cross_organ": 0.3,
        }.items()
    }
    summary = write_evaluation_summary(
        {"output_dir": str(tmp_path / "run"), "run_name": "test"},
        tmp_path,
        metrics,
    )

    assert set(summary["task"]) == {"breast_val", "breast_test", "cross_organ"}
