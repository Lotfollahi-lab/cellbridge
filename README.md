# terra-multi

Multimodal preprocessing and embedding experiments for HEST Xenium samples.

## Layout

- `src/terra_multi/preprocess/`: numbered HEST preprocessing steps.
- `src/terra_multi/exploration/`: expression and cell-type probes.
- `src/terra_multi/distill/`: UNI-to-TERRA distillation.
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
