# Frozen NOVAE hierarchy analysis

This workflow reproduces the frozen-TERRA phenotype-ordering analysis on the
same 12 annotated breast sections, sampled cell IDs, phenotype strata, random
pair draws, and 50,000 ordering triplets.

The configured model is the count-based `prism-oncology/novae-human-0`. Each
complete section is embedded before annotated cells are selected because NOVAE
representations depend on the spatial neighborhood. Missing embeddings and
invalid neighborhoods are audited and excluded from both embedding matrices;
TERRA is rescored on that exact shared-valid cohort before comparison.

## Environment

The cluster driver supports CUDA 12.4, so install a matching PyTorch build
before the editable NOVAE package:

```bash
module load cellgen/uv/0.11.8
uv venv ../../envs/novae --python 3.12
UV_CACHE_DIR=/tmp/novae-uv-cache uv pip install \
  --python ../../envs/novae/bin/python \
  --index https://download.pytorch.org/whl/cu124 torch==2.6.0
UV_CACHE_DIR=/tmp/novae-uv-cache uv pip install \
  --python ../../envs/novae/bin/python -e ../../repos/novae
```

The runtime wrapper stores Hugging Face weights under the workspace
`cache/huggingface` directory rather than the home filesystem.

## Run

Submit all extraction jobs followed by the dependent analysis job:

```bash
scripts/novae/submit_experiment.sh \
  configs/exploration/novae_phenotype_structure.yaml
```

Run only the final analysis after embeddings are present:

```bash
PYTHONPATH=src ../../envs/terra-dev/bin/python -m \
  cellbridge.exploration.analyze_novae_phenotypes \
  --config configs/exploration/novae_phenotype_structure.yaml
```

Results are written to
`outputs/exploration/novae_phenotype_structure/`, including the aligned sampled
cell manifest, alignment audit, raw pair/triplet tables, plots, and a compact
TERRA-versus-NOVAE comparison.
