"""Run the gated one-tissue TERRA flow feasibility experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
import yaml

from cellbridge.config import load_config
from cellbridge.flow.checkpoints import load_base, save_base, save_prompts
from cellbridge.flow.data import build_pilot_datasets
from cellbridge.flow.evaluation import (
    evaluate_phenotypes,
    generate_embeddings,
    latent_metrics,
    scale_up_gate,
)
from cellbridge.flow.model import CellFlow, SoftPromptTokens
from cellbridge.flow.training import (
    diagnose,
    diagnostic_passes,
    set_seed,
    train_base,
    train_prompts,
)
from cellbridge.jepa.model import TargetNormalizer


def _write_json(path: Path, values: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(values, indent=2, sort_keys=True) + "\n")


def _set_run_seed(cfg: dict, seed: int, *, override_sampling: bool = True) -> None:
    if override_sampling:
        cfg["data"]["sampling_seed"] = seed
    sections = ("base_training", "diagnostic", "softrepa", "generation", "evaluation")
    for section in sections:
        cfg[section]["seed"] = seed


def _validate(cfg: dict) -> None:
    if cfg.get("flow_version") != "terra_cell_flow_v1":
        raise ValueError("flow config requires flow_version: terra_cell_flow_v1")
    for key in (
        "sample_id",
        "data_root",
        "manifest_path",
        "adata_path",
        "labels_path",
        "output_dir",
        "checkpoint_dir",
        "data",
        "model",
        "base_training",
        "diagnostic",
        "softrepa",
        "generation",
        "evaluation",
    ):
        if key not in cfg:
            raise ValueError(f"flow config is missing {key}")
    if int(cfg["data"]["neighbours"]) < 1:
        raise ValueError("data.neighbours must be positive")
    if int(cfg["model"]["target_dim"]) != 384:
        raise ValueError("the real TERRA cell embedding dimension must be 384")
    if float(cfg["softrepa"].get("diagonal_weight", 0.0)) < 0:
        raise ValueError("softrepa.diagonal_weight must be non-negative")


def _device(cfg: dict) -> torch.device:
    requested = str(cfg.get("device", "auto"))
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device



def _new_prompts(cfg, model, device):
    settings = cfg["softrepa"]
    return SoftPromptTokens(
        model.predictor_depth,
        int(settings["prompt_tokens"]),
        model.model_dim,
        time_dependent=bool(settings["time_dependent"]),
        time_bins=int(settings.get("time_bins", 100)),
    ).to(device)


def _diagnostic_stage(cfg, train_dataset, test_dataset, device, run_dir, ckpt_dir):
    settings = cfg["base_training"]
    model = CellFlow(**cfg["model"]).to(device)
    normalizer = TargetNormalizer.fit(train_dataset.anchor_terra).to(device)
    records = train_base(model, normalizer, train_dataset, settings, device)
    pd.DataFrame(records).to_csv(run_dir / "base_training.csv", index=False)
    save_base(ckpt_dir / "base.pt", model, normalizer)
    metrics = diagnose(
        model,
        normalizer,
        test_dataset,
        cfg["diagnostic"],
        device,
        exclusion_settings=cfg["softrepa"]["false_negative_exclusion"],
        score_settings=cfg["softrepa"],
    )
    metrics["gate_passed"] = diagnostic_passes(
        metrics, cfg["diagnostic"]["gate"]
    )
    _write_json(run_dir / "base_diagnostic.json", metrics)
    return model, normalizer, metrics


def _full_stage(
    cfg,
    train_dataset,
    test_dataset,
    device,
    run_dir,
    ckpt_dir,
    model=None,
    normalizer=None,
    base_metrics=None,
):
    if model is None:
        base_path = ckpt_dir / "base.pt"
        if not base_path.exists():
            raise RuntimeError("--stage full requires a completed diagnostic checkpoint")
        model, normalizer = load_base(base_path, device)
    if base_metrics is None:
        diagnostic_path = run_dir / "base_diagnostic.json"
        if not diagnostic_path.exists():
            raise RuntimeError(
                "--stage full requires base_diagnostic.json; run --stage diagnostic first"
            )
        base_metrics = json.loads(diagnostic_path.read_text())
    if "retrieval_top1_ci95_high" not in base_metrics:
        raise RuntimeError(
            "base diagnostic predates the corrected metrics; rerun --stage diagnostic"
        )
    if not base_metrics["gate_passed"]:
        _write_json(run_dir / "status.json", {"stopped_after_diagnostic": True})
        print("base diagnostic did not pass; stopping before SoftREPA", flush=True)
        return

    prompts = _new_prompts(cfg, model, device)
    prompt_records = train_prompts(
        model,
        prompts,
        normalizer,
        train_dataset,
        cfg["softrepa"],
        device,
    )
    pd.DataFrame(prompt_records).to_csv(run_dir / "prompt_training.csv", index=False)
    save_prompts(ckpt_dir / "prompts.pt", prompts)
    prompt_metrics = diagnose(
        model,
        normalizer,
        test_dataset,
        cfg["diagnostic"],
        device,
        prompts=prompts,
        exclusion_settings=cfg["softrepa"]["false_negative_exclusion"],
        score_settings=cfg["softrepa"],
    )
    _write_json(run_dir / "prompt_diagnostic.json", prompt_metrics)

    generation = cfg["generation"]
    seed = int(generation["seed"])
    base_train = generate_embeddings(
        model, None, normalizer, train_dataset, generation, device, seed=seed
    )
    base_test = generate_embeddings(
        model, None, normalizer, test_dataset, generation, device, seed=seed + 1
    )
    prompt_train = generate_embeddings(
        model, prompts, normalizer, train_dataset, generation, device, seed=seed
    )
    prompt_test = generate_embeddings(
        model, prompts, normalizer, test_dataset, generation, device, seed=seed + 1
    )
    metrics, per_class = evaluate_phenotypes(
        train_dataset,
        test_dataset,
        base_train,
        base_test,
        prompt_train,
        prompt_test,
        cfg["evaluation"],
    )
    pd.DataFrame(metrics).to_csv(run_dir / "phenotype_metrics.csv", index=False)
    pd.DataFrame(per_class).to_csv(run_dir / "per_class_f1.csv", index=False)
    latent = [
        {"arm": "flow", **latent_metrics(base_test, test_dataset.anchor_terra)},
        {
            "arm": "flow_softrepa",
            **latent_metrics(prompt_test, test_dataset.anchor_terra),
        },
    ]
    pd.DataFrame(latent).to_csv(run_dir / "latent_metrics.csv", index=False)
    gate = scale_up_gate(base_metrics, prompt_metrics, metrics)
    _write_json(run_dir / "scale_up_gate.json", gate)
    print(f"scale-up gate: {gate}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", choices=("diagnostic", "full", "all"), default="all")
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    cfg = load_config(args.config)
    _validate(cfg)
    seed = int(cfg["base_training"]["seed"] if args.seed is None else args.seed)
    _set_run_seed(cfg, seed, override_sampling=args.seed is not None)
    set_seed(seed)
    run_dir = Path(cfg["output_dir"]) / f"seed{seed}"
    ckpt_dir = Path(cfg["checkpoint_dir"]) / f"seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "resolved_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    _, train_dataset, test_dataset = build_pilot_datasets(cfg)
    device = _device(cfg)
    print(
        f"device={device}; train={len(train_dataset)}; test={len(test_dataset)}",
        flush=True,
    )
    model = normalizer = base_metrics = None
    if args.stage in {"diagnostic", "all"}:
        model, normalizer, base_metrics = _diagnostic_stage(
            cfg, train_dataset, test_dataset, device, run_dir, ckpt_dir
        )
        print(f"base diagnostic: {base_metrics}", flush=True)
    if args.stage in {"full", "all"}:
        _full_stage(
            cfg,
            train_dataset,
            test_dataset,
            device,
            run_dir,
            ckpt_dir,
            model,
            normalizer,
            base_metrics,
        )


if __name__ == "__main__":
    main()
