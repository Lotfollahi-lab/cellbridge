# Main model training and evaluation

Run commands from the repository root with `PYTHONPATH=src` (or the environment setup in the README). The existing lab `hest` environment supplies the training, image and widget dependencies; the lab `terra-dev` environment supplies the paper's CPU evaluation stack. The `scripts/inference/run_in_env.sh` wrapper selects these environments. Cluster scripts are optional LSF helpers; the Python commands can run directly elsewhere.

## Inputs and paper settings

`configs/train/lora/cellbridge.yaml` inherits explicit settings from `defaults.yaml`: last four UNI2-h blocks, LoRA rank 8, alpha 16, dropout 0.05, 224-pixel crops at 0.5 µm/pixel, and bidirectional contrastive training against 384-dimensional TERRA targets.

The input root is `${CELLBRIDGE_DATA_ROOT}/train/hest`. It contains `manifest.csv`, `splits/hest80_strict_v1.csv`, and sample directories containing `<sample>_terra.npz`. The manifest supplies slide locations and preprocessing metadata. Training uses the `distill_train` split with no per-sample cap; validation keeps the original cap of 5,000 cells per sample. "All cells" means all eligible cells in that training split, not validation or evaluation cells.

The paper run used 1,020,944 maximum steps, batch size 16 and seed 0; its selected checkpoint is step 950,000. Its training summary records 16,335,536 training cells and 30,000 validation cells. The adapted representation is 1,536-dimensional; the projected teacher space is not the main representation.

UNI2-h initialization is loaded through Trident's `uni_v2` encoder. To use a local backbone file, set `model.weights_path` through a configuration overlay. `CELLBRIDGE_CONFIG_OVERLAY=/path/to/local.yaml` overrides YAML settings without editing the shared configuration. W&B is disabled by default; enabling logging does not change the training objective.

## Train

```bash
python -m cellbridge.train.lora.train --config configs/train/lora/cellbridge.yaml
```

This writes metrics, `resolved_config.yaml` and `train_summary.json` under `outputs/reproducibility/train/uni_terra_four_layers_adapted_all_cells`, and weights under `checkpoints/uni_terra_four_layers_adapted_all_cells/best.pt`. Existing training artifacts are protected. To reproduce into fresh directories:

```bash
python -m cellbridge.train.lora.train --config configs/train/lora/cellbridge.yaml \
  --output-dir "$CELLBRIDGE_OUTPUTS_ROOT/reproducibility/train/terra4_reproduction" \
  --checkpoint-dir "$CELLBRIDGE_CHECKPOINTS_ROOT/terra4_reproduction"
```

On LSF, `bash scripts/train/submit_train.sh` submits the main configuration; pass a config path to select another run. Use a configuration overlay with fresh output/checkpoint directories for cluster reruns. Nothing is submitted merely by loading a configuration.

The preserved paper training directory contains its original config and logs unchanged. `relocation.json` records the old location and the current checkpoint checksum; the original paths inside historical logs/configs are not runnable defaults.

## Extract and evaluate

The method profile defaults to the main checkpoint, and CellBridge defaults to `lora_adapted`. Example for NuCLS:

```bash
python -m cellbridge.inference.run check --config configs/inference/nucls.yaml \
  --method cellbridge --output-dir "$CELLBRIDGE_OUTPUTS_ROOT/reproducibility/inference/cellbridge/uni_terra_four_layers_adapted_all_cells/nucls"
python -m cellbridge.inference.run run --config configs/inference/nucls.yaml \
  --method cellbridge --device cuda \
  --output-dir "$CELLBRIDGE_OUTPUTS_ROOT/reproducibility/inference/cellbridge/uni_terra_four_layers_adapted_all_cells/nucls"
```

Use a fresh output directory when rerunning published evaluations. `run` extracts embeddings, evaluates probes and performs the configured conformal analysis. The `extract`, `evaluate` and `conformal` commands can also run separately; cached evaluation needs no GPU. Use `--checkpoint` to select fresh training weights and `--arm lora_projected` only for an explicit projected-representation experiment.

For HEST, run one section at a time with `--sample-id <ID>` and an output ending in `hest/<ID>`. The `scripts/inference/hest/submit.sh` and `scripts/inference/ood/submit.sh` launch the selected CellBridge model and baseline methods. They no longer submit the retired 2L scaling runs.
