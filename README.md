# CellBridge

**Fine-Grained and Reliable Cell Classification from Pathology Images**

CellBridge transfers molecular information from paired spatial transcriptomics
into cell-centered H&E representations. It adapts the final four attention blocks
of UNI2-h with LoRA against a frozen TERRA teacher. At inference, images alone
provide the features; a fitted probe and Cell-Ontology-Fallback (COF) return either
fine-grained labels or broader, biologically coherent prediction sets.

## Install

Use Python **3.11 or 3.12** for the full model workflow. From this repository:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[models,data]'
python -m pip install -r requirements/encoders.txt
```

The last command installs the **pinned Mahmood Lab TRIDENT** source used by UNI2-h.
The unrelated PyPI package named `trident` is not a substitute. Install a
PyTorch/torchvision build appropriate for your GPU before the model extra if needed.
UNI2-h also requires access to its pretrained weights; a CellBridge adapter does
not contain the frozen UNI2-h backbone. See [model prerequisites](docs/reproduction.md#models).

For CPU evaluation of saved OOD embeddings and conformal prediction, `pip install .`
is sufficient. Extras add workflows independently:

| Extra | Provides |
|---|---|
| `models` | PyTorch, torchvision, timm and model-loading dependencies |
| `data` | HEST/AnnData, whole-slide TIFF, geometry and LMDB readers |
| `preprocess` | `data` plus Scanpy and dataset preprocessing dependencies |
| `analysis` | `data` plus plotting, Jupyter, widgets and UMAP |
| `wandb` | Optional training logging |
| `dev` | Tests, linting and distribution build tools |

TERRA, NOVAE and SpatialWhisperer use their own installations; see the reproduction
guide. No PyPI publication is assumed here. You can build and install a wheel with
`python -m pip install '.[dev]'`, `python -m build`, then
`python -m pip install dist/cellbridge-0.1.0-py3-none-any.whl`.

## Data and paths

Set absolute paths before running commands or starting Jupyter:

```bash
export CELLBRIDGE_ROOT="$PWD"
export CELLBRIDGE_DATA_ROOT="/path/to/data"
export CELLBRIDGE_OUTPUTS_ROOT="/path/to/outputs"
export CELLBRIDGE_CHECKPOINTS_ROOT="/path/to/checkpoints"
export CELLBRIDGE_REPOS_ROOT="/path/to/external/model/repos"
```

A wheel includes the configs and ontology mappings. Without a checkout, run
`cellbridge-config ./cellbridge-project`, then set `CELLBRIDGE_ROOT` to that
project's absolute path. Existing config directories are never overwritten.
Unset data/output roots default to the research workspace for a checkout, or the
current working directory for an installed wheel. Explicit paths are recommended.

```text
data/
├── train/hest/
│   ├── manifest.csv
│   ├── splits/{hest80_strict_v1,hest_phenotype_split_v1}.csv
│   └── TENX.../
│       ├── TENX....h5ad
│       ├── TENX..._terra.npz
│       └── TENX..._granular_cell_labels.csv
└── eval/
    ├── nucls/{raw/rgb/,raw/csv/,nucls_info.csv,splits/}
    └── {pannuke,lizard,phenocell}/
        ├── <dataset>_lmdb/{data.mdb,lock.mdb}
        ├── <dataset>_info.csv
        └── splits/<split-file>.csv
```

HEST's manifest links each section to its H&E image and pixel size. Teacher NPZ
files pair cell IDs, H&E coordinates and embeddings; labels and split tables are
separate. OOD datasets pair cell metadata with fixed sample-level train/validation/
test assignments. **Do not regenerate random splits to reproduce the paper.**
[Dataset schemas and preparation](docs/data.md) specify required columns, sources
and preprocessing commands. Data, split/cohort CSVs, and model weights are not
bundled in the package.

## Train

Prepare the HEST artifacts and obtain UNI2-h access, then run:

```bash
cellbridge-train --config "$CELLBRIDGE_ROOT/configs/train/lora/cellbridge.yaml"
```

The paper config uses rank 8, alpha 16, batch size 16, symmetric InfoNCE with
`temperature=0.07`, 1,020,944 training steps, and validation every 50,000 steps.
Training writes `resolved_config.yaml` and logs under the configured output
directory, and the best adapter checkpoint to
`$CELLBRIDGE_CHECKPOINTS_ROOT/uni_terra_four_layers_adapted_all_cells/best.pt`.
Existing training artifacts are protected; use `--output-dir` and
`--checkpoint-dir` for a new run. Training targets CUDA; the paper used one H200.

## Evaluate

Start with one dataset and the paper representation:

```bash
CONFIG="$CELLBRIDGE_ROOT/configs/inference/nucls.yaml"
RUN="$CELLBRIDGE_OUTPUTS_ROOT/reproducibility/inference/cellbridge/uni_terra_four_layers_adapted_all_cells/nucls"

cellbridge-infer check --config "$CONFIG" --method cellbridge --arm lora_adapted --output-dir "$RUN"
cellbridge-infer run --config "$CONFIG" --method cellbridge --arm lora_adapted --output-dir "$RUN" --device cuda
```

`check` validates data/configuration and local checkpoint paths. `run` extracts
embeddings, evaluates linear and kNN probes, and runs conformal evaluation.
Use `extract`, `evaluate`, or `conformal` separately to reuse saved embeddings;
`--probe linear` limits a stage to the paper's conformal probe. `--checkpoint`
overrides the adapter path. For HEST, add `--sample-id TENX201` and use `hest.yaml`.

Results live under `embeddings/<arm>/` and
`evaluation/<arm>/<task>/<probe>/` (with an extra section directory for HEST).
`metrics.csv`, `predictions.npz`, `uncertainty.csv` and `bootstrap_metrics.csv`
record predictions and statistics. Conformal outputs are under
`conformal_prediction/`; **`branch_graph` is COF**. The paper uses
`lora_adapted` (1536-dimensional encoder features), not `lora_projected`.

## Reproduce the manuscript

Follow [the reproduction guide](docs/reproduction.md) for prerequisites, baseline
models, exact output layout, ablations and figure notebooks. With inputs ready:

```bash
bash scripts/reproduce.sh cellbridge check
bash scripts/reproduce.sh cellbridge run
# Repeat for uni, virchow2, volta, lemon, past, spatialwhisperer.
```

This runs four OOD datasets (both PhenoCell tasks) and the 14 paper HEST sections.
The source cohort has 15 entries; TENX143 is excluded as in Appendix A.1.
Notebooks 07, 08, 09 and 10 produce classification, conformal, qualitative and
ablation analyses. Install `.[analysis]` before opening them.

NuCLS granularity uses the saved paper-layout runs:

```bash
cellbridge-granularity --config configs/analysis/granularity.yaml
cellbridge-granularity-refit --config configs/analysis/granularity.yaml
```

The first sums raw-class probabilities into the native hierarchy; the second fits
coarse probes. Both save `summary.csv` and `bootstrap_metrics.csv` using paired
slide-bootstrap scoring. No plots or narrative reports are generated.
For a simple qualitative view in Jupyter:

```python
from cellbridge.qualitative.data import load_catalog
from cellbridge.qualitative.views import Explorer

Explorer(load_catalog(), metric="set_size").show()
```

Set size measures prediction breadth. The manuscript's reliability is
`1 - set_size / n_classes`; it is not a calibrated probability of correctness.

## Development and release status

```bash
python -m pip install -e '.[dev,preprocess,analysis]'
python -m pytest
python -m build
```

See [the audit](docs/AUDIT.md) for pre-release findings and their resolution.
CellBridge code is licensed under the [Apache License 2.0](LICENSE.md). Approved
download locations for adapter weights and exact split artifacts still require
author decisions. Third-party datasets and models retain their own terms. Package and CPU tests do not establish numerical
reproduction of the full manuscript.
