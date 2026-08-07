import json

import pandas as pd

from cellbridge.flow.evaluation import scale_up_gate
from cellbridge.flow.summary import aggregate


def _phenotype_rows():
    return [
        {"arm": "flow", "macro_f1": 0.4, "balanced_accuracy": 0.5},
        {"arm": "flow_softrepa", "macro_f1": 0.5, "balanced_accuracy": 0.6},
    ]


def test_scale_up_gate_rejects_retrieval_change_inside_uncertainty():
    base = {
        "relative_error_reduction": 0.1,
        "retrieval_top1": 0.10,
        "retrieval_top1_ci95_high": 0.15,
    }
    prompt = {
        "relative_error_reduction": 0.2,
        "retrieval_top1": 0.11,
        "retrieval_top1_ci95_low": 0.08,
    }
    gate = scale_up_gate(base, prompt, _phenotype_rows())
    assert not gate["retrieval_robustly_improved"]
    assert not gate["compatibility_improved"]
    assert not gate["scale_up"]


def test_scale_up_gate_accepts_clear_retrieval_and_phenotype_gains():
    base = {
        "relative_error_reduction": 0.1,
        "retrieval_top1": 0.10,
        "retrieval_top1_ci95_high": 0.15,
    }
    prompt = {
        "relative_error_reduction": 0.2,
        "retrieval_top1": 0.25,
        "retrieval_top1_ci95_low": 0.20,
    }
    gate = scale_up_gate(base, prompt, _phenotype_rows())
    assert gate["retrieval_robustly_improved"]
    assert gate["scale_up"]


def test_seed_summary_has_no_spurious_index_column(tmp_path):
    run_dir = tmp_path / "seed0"
    run_dir.mkdir()
    base = {
        "relative_error_reduction": 0.1,
        "retrieval_top1": 0.1,
        "retrieval_top1_ci95_high": 0.15,
    }
    prompt = {
        "relative_error_reduction": 0.2,
        "retrieval_top1": 0.25,
        "retrieval_top1_ci95_low": 0.2,
    }
    (run_dir / "base_diagnostic.json").write_text(json.dumps(base))
    (run_dir / "prompt_diagnostic.json").write_text(json.dumps(prompt))
    pd.DataFrame(
        [
            {"arm": "real_terra", "macro_f1": 0.8, "balanced_accuracy": 0.8},
            {"arm": "direct_he", "macro_f1": 0.3, "balanced_accuracy": 0.3},
            *_phenotype_rows(),
        ]
    ).to_csv(run_dir / "phenotype_metrics.csv", index=False)
    pd.DataFrame(
        [
            {"arm": "flow", "latent_cosine": 0.5, "latent_mse": 0.5},
            {"arm": "flow_softrepa", "latent_cosine": 0.6, "latent_mse": 0.4},
        ]
    ).to_csv(run_dir / "latent_metrics.csv", index=False)
    pd.DataFrame(
        [
            {"arm": arm, "class": "a", "f1": score}
            for arm, score in (("flow", 0.4), ("flow_softrepa", 0.5))
        ]
    ).to_csv(run_dir / "per_class_f1.csv", index=False)

    _, per_class, gate = aggregate(tmp_path, [0])
    assert list(per_class.columns) == ["arm", "class", "mean", "std"]
    assert gate["scale_up"]
