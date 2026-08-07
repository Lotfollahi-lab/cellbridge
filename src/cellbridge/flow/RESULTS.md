# Corrected TENX116 feasibility result

## Decision

The corrected seed-0 run passed the base feasibility diagnostic but neither
SoftREPA variant passed the compatibility/retrieval scale-up gate. Seeds 1 and
2 and additional tissues were therefore not run, following the predeclared
rule to use three seeds only after a promising initial result.

The experiment used 12,000 spatial-training cells, 4,096 sampled spatial-test
cells, 32 H&E neighbours per condition, and 512 diagnostic states. The primary
diagnostic applies the configured symmetric false-negative mask; chance top-1
is 3.14%. The same frozen base was used for both prompt variants.

## Compatibility and retrieval

| Metric | Base flow | SoftREPA, dweight=0 | SoftREPA, dweight=10 |
|---|---:|---:|---:|
| Matched endpoint MSE | 0.2580 | 0.2805 | 0.2569 |
| Mismatched endpoint MSE | 0.3125 | 0.3663 | 0.3133 |
| Relative error separation | 17.43% | 23.41% | 17.98% |
| Top-1 retrieval | 62/512 (12.11%) | 68/512 (13.28%) | 75/512 (14.65%) |
| Top-1 95% interval | 9.56–15.22% | 10.61–16.50% | 11.85–17.97% |
| Top-5 retrieval | 47.85% | 45.70% | 47.66% |
| Mean matched probability | 3.35% | 3.49% | 3.40% |

The base condition signal is real but time-dependent: relative error
separation is 25.65% for `t < 0.25` and only 5.21% for `t >= 0.75`. Unanchored
SoftREPA mainly pushes mismatches away while increasing matched error. The
reference-style `dweight=10` anchor prevents that drift, but neither prompt
variant has a top-1 interval wholly above the base interval.

## Latent fidelity and phenotype prediction

| Arm | Latent cosine | Latent MSE | Macro-F1 | Balanced accuracy |
|---|---:|---:|---:|---:|
| Real TERRA | — | — | 0.7132 | 0.7378 |
| Direct UNI2 H&E | — | — | 0.2346 | 0.2512 |
| Base flow | 0.8705 | 0.1862 | 0.2176 | 0.2471 |
| Flow + SoftREPA, dweight=0 | 0.8191 | 0.2484 | 0.2210 | 0.2532 |
| Flow + SoftREPA, dweight=10 | 0.8602 | 0.1980 | 0.2180 | 0.2506 |

Unanchored SoftREPA improves macro-F1 by 0.0034 and balanced accuracy by
0.0061, but substantially worsens both latent metrics and remains below direct
H&E. Anchored SoftREPA preserves latent quality better, but its macro-F1 gain
is only 0.00034. Both scale-up gates are false.

All 16 per-class F1 values are in
`outputs/flow/terra_cell_flow_tenx116_corrected/seed0/softrepa_ablation_per_class_f1.csv`.

## Reproduction

From the repository root, set the data root and run the corrected seed:

```bash
export CELLBRIDGE_DATA_ROOT=/lustre/scratch126/cellgen/lotfollahi/so25/data
PYTHONPATH=src ../../envs/terra-dev/bin/python -m cellbridge.flow.run \
  --config configs/flow/terra_cell_flow_tenx116_corrected.yaml \
  --stage all --seed 0
```

The `dweight=10` ablation reused that frozen base and diagnostic:

```bash
mkdir -p outputs/flow/terra_cell_flow_tenx116_corrected_dweight10/seed0 \
  checkpoints/flow/terra_cell_flow_tenx116_corrected_dweight10/seed0
cp outputs/flow/terra_cell_flow_tenx116_corrected/seed0/base_diagnostic.json \
  outputs/flow/terra_cell_flow_tenx116_corrected_dweight10/seed0/
cp outputs/flow/terra_cell_flow_tenx116_corrected/seed0/base_training.csv \
  outputs/flow/terra_cell_flow_tenx116_corrected_dweight10/seed0/
cp checkpoints/flow/terra_cell_flow_tenx116_corrected/seed0/base.pt \
  checkpoints/flow/terra_cell_flow_tenx116_corrected_dweight10/seed0/
PYTHONPATH=src ../../envs/terra-dev/bin/python -m cellbridge.flow.run \
  --config configs/flow/terra_cell_flow_tenx116_corrected_dweight10.yaml \
  --stage full --seed 0
```

Outputs are under `outputs/flow/terra_cell_flow_tenx116_corrected*`; versioned
checkpoints are under the matching `checkpoints/flow/` directories.
