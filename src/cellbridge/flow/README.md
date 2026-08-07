# TERRA cell-flow feasibility pilot

This experiment predicts each TENX116 cell's fixed 384-D TERRA embedding from
its frozen UNI2 H&E crop embedding and 32 nearest H&E-cell neighbours. Training
uses STFlow's linear Gaussian path and clean-endpoint MSE. Neighbourhood pools
are kept on their respective sides of the established spatial 80/20 split.

The held-out diagnostic uses up to 512 of the 4,096 sampled spatial-test cells.
It compares native endpoint error for matched and shuffled conditions in
32-way retrieval batches, reports a 95% interval for top-1, and stratifies the
metrics by flow time. The configured false-negative exclusions are applied to
both base and prompted diagnostics; unmasked metrics are also retained.
For scale-up, a prompted top-1 gain counts only when its interval is entirely
above the base interval, so a one-cell fluctuation cannot pass the gate.

SoftREPA freezes the base model and trains four tokens at every predictor
cross-attention layer. Its compatibility logits exactly follow the local code:

```text
4 * exp(-endpoint_mse / 0.07)
```

`softrepa.diagonal_weight` exposes the reference implementation's optional
matched-error anchor. It is explicitly zero in the pilot config, matching the
upstream default; a nonzero value is a separate scientific ablation. The
configurable false-negative mask symmetrically excludes the eight closest
cells in frozen TERRA cosine space and condition centers within 50 µm.

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
`--seed 1` and `--seed 2`. The CLI seed controls cell subsampling as well as
model initialization, flow noise, data order, generation, and evaluation.
Each seed has independent versioned checkpoints and output files.

The run writes base/prompt diagnostics, generated-latent cosine/MSE, phenotype
macro-F1 and balanced accuracy, per-class F1, and resolved configuration.
