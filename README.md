# terra-multi

Multimodal preprocessing and embedding experiments for HEST Xenium samples.

## Layout

- `src/terra_multi/preprocess/`: numbered HEST preprocessing steps.
- `src/terra_multi/exploration/`: expression and cell-type probes.
- `src/terra_multi/distill/`: UNI-to-TERRA distillation.
- `src/terra_multi/evaluation/`: evaluation on non-ST datasets.
- `configs/`: preprocessing, exploration, and experiment settings.
- `scripts/`: LSF launchers matching those workflows.

The HEST split is stored with the data at
`data/train/hest/splits/hest80_strict_v1.csv`.

## Preprocessing

Run discovery and submit every sample:

```bash
scripts/preprocess/submit_all_samples.sh
```

Submit the numbered steps for one sample:

```bash
scripts/preprocess/run_sample_preprocess.sh TENX116
```

## Distillation

The source is separated by responsibility:

- `data.py`: discovery, alignment, sampling, and validated caches.
- `model.py`: projector and loss.
- `train.py`: loaders, optimization, and checkpointing.
- `inference.py`: post-training predictions and metrics.
- `evaluate_probes.py`: held-out linear probes and aggregation.

Launchers accept a config, so new variants only need a new YAML file:

```bash
scripts/distill/submit_train.sh configs/distill/distill_v0.yaml
scripts/distill/submit_probes.sh configs/distill/distill_v0.yaml
```

Submit one evaluation arm directly:

```bash
scripts/distill/submit_probe.sh configs/distill/distill_v0.yaml distilled
```

## OOD evaluation

PhenoCell is evaluated separately from HEST. The workflow reconstructs each
field from its LMDB tiles, extracts centred cell-context crops, and compares
frozen raw UNI with distilled embeddings on the official patient split. Stable
UNI features are kept under `data/eval/phenocell/embeddings/`; each experiment
output keeps checkpoint-specific projections under `embeddings/distilled/` and
probe results under `probes/`.

```bash
scripts/evaluation/phenocell/submit_all.sh configs/evaluation/phenocell.yaml
```

If the projector is still training, make embedding wait for that LSF job:

```bash
DEPENDENCY=<distillation-job-id> \
  scripts/evaluation/phenocell/submit_all.sh configs/evaluation/phenocell.yaml
```

The probe reports both the full 14-class benchmark and an immune-only task.
Change `crop_size`, `projection.checkpoint`, or `output_dir` in a copied config
to keep evaluation variants isolated.

After the UNI cache exists, new distillation variants only need projection and
probing. Keep `embedding.output_dir` unchanged, then change
`projection.checkpoint` and the top-level `output_dir` in a copied config:

```bash
scripts/evaluation/phenocell/submit_variant.sh configs/evaluation/my_variant.yaml
```
