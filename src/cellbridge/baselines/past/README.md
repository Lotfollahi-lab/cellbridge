# PAST

This integration evaluates the released PAST image tower without fine-tuning.
It exports the raw 1024-dimensional output recommended by PAST's feature
extraction example, before the 256-dimensional multimodal training projection,
and uses CellBridge's shared frozen linear and cosine-kNN probes.

## Data contract

- checkpoint: `checkpoint-context300.pth` from `Boyoungc/PAST`;
- image tower: fine-tuned `vit_large_patch16_224` weights embedded in that
  checkpoint;
- input: a cell-centred 224x224 crop in native level-0 pixels, with black
  boundary padding and ImageNet normalization;
- HEST: all TERRA cells in the five fixed fast samples;
- PhenoCell: official patient-disjoint split, `all` and `immune` tasks;
- PanNuke: official train/validation/test split and five-class `all` task.

PAST's released crop code does not normalize physical resolution. Consequently,
this runner preserves 224 native source pixels on every dataset and records that
choice in each artifact's provenance.

## Weights

Download the 4,085,228,034-byte public checkpoint to
`repos/past_sc/checkpoints/checkpoint-context300.pth`:

```bash
hf download Boyoungc/PAST checkpoint-context300.pth \
  --revision 8c1abe3c955f80fd1b14dc74431bcb4cf2faede2 \
  --local-dir repos/past_sc/checkpoints
```

The runner verifies the released file size and SHA-256 before deserializing it.
A separate UNI checkpoint is not needed because the PAST checkpoint contains
the fine-tuned image tower.

## Run

```bash
scripts/baselines/past/run_in_env.sh check \
  --config configs/baselines/past.yaml
scripts/baselines/past/run_in_env.sh prepare \
  --config configs/baselines/past.yaml
scripts/baselines/past/submit_parallel.sh
```

For a small isolated smoke extraction, pass both `--max-cells-per-sample` and
an explicit `--output-dir` to an `extract-*` command. Features, metrics, and
logs are written below `outputs/baselines/past/`.
