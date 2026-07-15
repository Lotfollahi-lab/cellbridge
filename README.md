# terra-multi

Multimodal preprocessing and embedding experiments for HEST Xenium samples.

## Layout

- `src/terra_multi/preprocess/`: numbered HEST preprocessing steps.
- `src/terra_multi/exploration/`: expression and cell-type probes.
- `src/terra_multi/distill/`: UNI-to-TERRA distillation.
- `src/terra_multi/lora/`: LoRA adaptation from H&E crops to TERRA features.
- `src/terra_multi/evaluation/`: shared linear/kNN evaluation and PhenoCell.
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

## Shared evaluation

Cell-type evaluation has one implementation under `src/terra_multi/evaluation/`:

- `linear_probe.py`: the existing standardized logistic-regression protocol.
- `zero_shot.py`: cosine kNN with no fitted model parameters.
- `classification.py`: shared fitting, metrics, and prediction artifacts.
- `hest.py` and `phenocell.py`: dataset-specific split and preparation rules.

kNN is non-parametric but not label-free: it uses labelled cells from the
training reference split. Checked-in configs run both `linear` and `knn`.
Historical linear result paths and schemas are unchanged; kNN artifacts live
under `knn/` and use separate `*_knn.csv` summaries.

Run every configured HEST feature arm and both evaluators locally with one
command:

```bash
PYTHONPATH=src python -m terra_multi.evaluation.run \
  --config configs/exploration/celltype_tenx116.yaml
```

## Distillation

The source is separated by responsibility:

- `data.py`: discovery, alignment, sampling, and validated caches.
- `model.py`: projector and loss.
- `train.py`: loaders, optimization, and checkpointing.
- `inference.py`: post-training predictions and metrics.
- `evaluate_probes.py`: feature-arm setup and shared evaluation aggregation.

Launch training and the four label-probe arms as one dependency-linked workflow:

```bash
scripts/distill/submit_experiment.sh configs/distill/distill_v0_vicreg_mild.yaml
```

The provided v0 configs are `distill_v0_cosine_only.yaml`,
`distill_v0_vicreg_mild.yaml`, and `distill_v0_vicreg_strong.yaml`. Each uses
the five-sample `label_test_fast` set by default; set `eval.label_test_mode` to
`full` to evaluate all annotated HEST samples. Training writes the best
checkpoint, resolved config, epoch metrics, and validation/fast-test predicted
embeddings before the probe jobs start.

Training and probes can also be submitted separately:

```bash
scripts/distill/submit_train.sh configs/distill/distill_v0_vicreg_mild.yaml
scripts/distill/submit_probes.sh configs/distill/distill_v0_vicreg_mild.yaml
```

Submit one evaluation arm directly:

```bash
scripts/distill/submit_probe.sh configs/distill/distill_v0_vicreg_mild.yaml distilled
```

Prototype distillation v1 fits 256 normalized TERRA prototypes on
`distill_train`, trains the UNI2 prototype classifier, exports its penultimate
and probability features, then evaluates both with the same cell-type probe:

```bash
scripts/distill/submit_experiment.sh configs/distill/distill_v1_prototype.yaml
```

The v1 config defaults to the five-sample fast label test. Its KMeans model,
assignments, exported features, metrics, and resolved prototype settings are
written under `outputs/distill_v1/`; the best student checkpoint is written to
`checkpoints/distill_v1/best.pt`.

## LoRA: H&E to TERRA

LoRA distillation trains UNI2-h directly from cell-centred H&E crops. The frozen
encoder gets rank-8 LoRA adapters in its last two or four attention blocks and
a 384-D TERRA projection head. `objective.mode` selects contrastive-only or the
optional 100-gene GEX head. Independently, `objective.targets` selects either
the original exact-pair InfoNCE targets or soft targets derived from pairwise
TERRA cosine similarity within each single-section batch. Because the HEST
assays have different gene panels, train-only HVGs are selected over the panel
union and SmoothL1 is masked wherever a gene was not assayed. Neither gene
selection nor any training decision reads `label_test_fast` or PhenoCell.

Submit training, best-checkpoint selection, feature export, and both HEST and
PhenoCell probes with one command:

```bash
scripts/lora/submit_experiment.sh configs/lora/contrastive_k0.yaml
```

Run the new soft-target experiment with the same training and evaluation
pipeline:

```bash
scripts/lora/submit_experiment.sh configs/lora/terra_similarity_last4.yaml
```

Its only alignment change is:

```yaml
objective:
  targets: terra_similarity
  teacher_temperature: 0.1
  exact_match_boost: 0.05
  bidirectional: true
```

The diagonal boost guarantees that the exact cell remains the strongest
positive. Similar TERRA cells receive softer positive mass, and the loss is
`KL(teacher || image-to-TERRA)`, optionally averaged with TERRA-to-image.

Submit the head-only baseline plus all three requested LoRA runs together:

```bash
scripts/lora/submit_suite.sh
```

The main configs are:

- `configs/lora/frozen_head_only.yaml`: frozen UNI plus a trained
  projection head.
- `configs/lora/contrastive_k0.yaml`: contrastive, last two blocks.
- `configs/lora/contrastive_last4.yaml`: contrastive, last four blocks.
- `configs/lora/contrastive_gex_last4.yaml`: contrastive plus 100-gene GEX,
  last four blocks.
- `configs/lora/terra_similarity_last4.yaml`: TERRA-similarity soft targets,
  last four blocks.

Each evaluation exports original UNI, adapter-disabled head, adapted UNI, and
adapted projected features; the GEX run also exports its 1024-D penultimate
feature. Results live below the configured `output_dir/evaluation/`, including
a compact `summary.csv` with HEST five-sample means and PhenoCell all/immune
macro-F1 and balanced accuracy. Set `max_cells_per_sample` lower for a quick
training smoke run. Training and validation use deterministic, full batches of
unique cells from one tissue section, so section identity cannot solve InfoNCE.

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

The evaluators report both the full 14-class benchmark and an immune-only task.
Change `crop_size`, `projection.checkpoint`, or `output_dir` in a copied config
to keep evaluation variants isolated.

After the UNI cache exists, new distillation variants only need projection and
probing. Keep `embedding.output_dir` unchanged, then change
`projection.checkpoint` and the top-level `output_dir` in a copied config:

```bash
scripts/evaluation/phenocell/submit_variant.sh configs/evaluation/my_variant.yaml
```
