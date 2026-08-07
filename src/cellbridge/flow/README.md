# TERRA cell-flow feasibility pilot

This experiment predicts each TENX116 cell's fixed 384-D TERRA embedding from
its frozen UNI2 H&E crop embedding and 32 nearest H&E-cell neighbours. Training
uses STFlow's linear Gaussian path and clean-endpoint MSE. Neighbourhood pools
are kept on their respective sides of the established spatial 80/20 split.

The held-out diagnostic compares native endpoint error for all matched and
shuffled conditions within 32-way retrieval batches. The full stage runs only
when matched error is at least 5% lower and top-1 retrieval is at least twice
chance.

SoftREPA freezes the base model and trains four tokens at every predictor
cross-attention layer. Its logits exactly follow the local implementation:

```text
4 * exp(-endpoint_mse / 0.07)
```

The configurable false-negative mask excludes the eight closest cells in
frozen TERRA cosine space and condition centers within 50 µm.

From the repository root, run the diagnostic and inspect its gate:

```bash
PYTHONPATH=src ../../envs/terra-dev/bin/python -m cellbridge.flow.run \
  --config configs/flow/terra_cell_flow_tenx116.yaml --stage diagnostic
cat outputs/flow/terra_cell_flow_tenx116/seed0/base_diagnostic.json
```

If it passes, run SoftREPA and all four phenotype arms:

```bash
PYTHONPATH=src ../../envs/terra-dev/bin/python -m cellbridge.flow.run \
  --config configs/flow/terra_cell_flow_tenx116.yaml --stage full
```

To reproduce both gated stages in one process:

```bash
PYTHONPATH=src ../../envs/terra-dev/bin/python -m cellbridge.flow.run \
  --config configs/flow/terra_cell_flow_tenx116.yaml --stage all
```

Only after `scale_up_gate.json` reports `scale_up: true`, repeat with
`--seed 1` and `--seed 2`. Each seed has independent checkpoints and outputs.

The run writes base/prompt diagnostics, generated-latent cosine/MSE, phenotype
macro-F1 and balanced accuracy, per-class F1, resolved configuration, and the
two checkpoints under the configured output directories.
