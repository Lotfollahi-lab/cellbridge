# VOLTA

This integration evaluates the released VOLTA ovarian momentum encoder without
fine-tuning. It exports the raw 512-dimensional pre-projection features and
uses the repository's shared frozen linear and cosine-kNN probes.

## Data contract

- checkpoint: ovarian `preact_resnet18`, epoch 199;
- input: instance bounding box expanded to a square at 2× its longer side;
- resize: OpenCV cubic to 32×32;
- normalization: ovarian training mean and standard deviation;
- HEST: all TERRA cells in the five fixed fast samples, using aligned Xenium
  cell polygons;
- PhenoCell: official patient-disjoint split, `all` and `immune` tasks;
- PanNuke: official train/validation/test split and five-class `all` task.

The checkpoint must be at
`repos/VOLTA/pretrained_weights/ovarian/checkpoint.pth.tar`. Its expected
SHA-256 is recorded in `configs/baselines/volta.yaml` and checked before model
use.

## Run

```bash
scripts/baselines/volta/run_in_env.sh check \
  --config configs/baselines/volta.yaml
scripts/baselines/volta/run_in_env.sh prepare \
  --config configs/baselines/volta.yaml
scripts/baselines/volta/submit_parallel.sh
```

For a small isolated smoke extraction, pass both `--max-cells-per-sample` and
an explicit `--output-dir` to an `extract-*` command.

Features, per-sample predictions, metrics, logs, and combined comparison tables
are written below `outputs/baselines/volta/`.
