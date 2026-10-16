# 🎯 Reproducibility Guide

This guide walks you through how to replicate all the experimental results and ablation studies of the preprint <a href="TODO">Fine-Grained and Reliable Cell Classification from Pathology Images</a>.
It consists of the following sections:

- [Installation options](#installation-options)
- [Model access](#model-access)
- [Data preparation](#data-preparation)
- [Experimental results](#experimental-results)
- [Ablation studies](#ablation-studies)

## Installation options

 From the repository root, create an environment, with **Python 3.11**:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

For CPU evaluation of saved OOD embeddings and conformal prediction, the core package is sufficient.
However, HEST samples additionally need `.[data]`.

```bash
python -m pip install .
python -m pip install '.[data]'
```

For GPU use, install the matching PyTorch/torchvision build using the
[official PyTorch installation selector](https://pytorch.org/get-started/locally/)
**before** the commands below.
Then install CellBridge and the pinned encoder requirements.

```bash
python -m pip install '.[models,data]'
python -m pip install -r requirements/encoders.txt
```

Finally, add extras for the workflows you need:

| Extra | Provides |
|---|---|
| `models` | PyTorch, torchvision, timm and model-loading dependencies |
| `data` | HEST/AnnData, whole-slide TIFF, geometry and LMDB readers |
| `preprocess` | `data` plus Scanpy and dataset preprocessing dependencies |
| `analysis` | `data` plus plotting, Jupyter, widgets and UMAP |
| `wandb` | Optional training logging |
| `dev` | Tests, linting, and distribution build tools |

Install TERRA only when generating teacher embeddings, NOVAE for the teacher ablation, and SpatialWhisperer for that baseline.
These packages require separate installation as described below.

## Model access

After having installed the `models,data` extras and `requirements/encoders.txt` with Python 3.11, follow the below table regarding CellBridge and baselines models.

| Model | Access |
|---|---|
| CellBridge | Released `cellbridge.pt` adapter from [ogutsevda/cellbridge](https://huggingface.co/ogutsevda/cellbridge), plus the original UNI2-h weights loaded through TRIDENT. Obtain access to `MahmoodLab/UNI2-h` on Hugging Face and authenticate using the Hugging Face client. |
| UNI2-h | Same frozen backbone as above. |
| Virchow2 | Obtain access to `paige-ai/Virchow2`. |
| LEMON | Obtain access to `aliceblondel/LEMON` and use `arch: vits8`. |
| PAST | Download its `checkpoint-context300.pth` into `$CELLBRIDGE_REPOS_ROOT/past_sc/checkpoints/`. |
| VOLTA | Clone its repository at `$CELLBRIDGE_REPOS_ROOT/VOLTA` and place the ovarian `checkpoint.pth.tar` under `pretrained_weights/ovarian/`. The adapter imports the repository's backbone code. |
| SpatialWhisperer | Install the authors' `spatialwhisperer` package and its model dependencies in a compatible environment. The adapter calls `load_spatialwhisperer_model` using `Good-Lab/spatialwhisperer`. |
| TERRA teacher | Install the authors' TERRA package exposing `download_pretrained`, `harmonize_adata`, `tokenize_adata`, and `embed_dataset`. Preprocessing uses `Lotfollahi-lab/TERRA-112M`. |
| NOVAE teacher ablation | Install the authors' `novae` package in its supported environment. `configs/preprocess/novae.yaml` pins `prism-oncology/novae-human-0`. |

## Data preparation

### Paths

Set the followig path variables before running any commands.

| Variable | Meaning |
|---|---|
| `CELLBRIDGE_ROOT` | Project directory containing `configs/` |
| `CELLBRIDGE_DATA_ROOT` | Prepared training and evaluation data |
| `CELLBRIDGE_OUTPUTS_ROOT` | Training logs, extracted embeddings, evaluation results, and figures |
| `CELLBRIDGE_CHECKPOINTS_ROOT` | Checkpoints saved by your training runs |
| `CELLBRIDGE_REPOS_ROOT` | External baseline repositories and their local weights |

### Input folder structure

```text
data/
├── train/hest/
│   ├── manifest.csv
│   ├── splits/{hest80_strict_v1,hest_phenotype_split_v1}.csv
│   └── TENX.../
│       ├── TENX....h5ad
│       ├── TENX..._terra.npz
│       ├── TENX..._novae.npz
│       └── TENX..._granular_cell_labels.csv
└── eval/
    ├── nucls/{raw/rgb/,raw/csv/,nucls_info.csv,splits/}
    └── {pannuke,lizard,phenocell}/
        ├── <dataset>_lmdb/{data.mdb,lock.mdb}
        ├── <dataset>_info.csv
        └── splits/<split-file>.csv
```

### HEST

Use the human Xenium/Xenium 5k cohort of HEST-1k v1.3.0. The preprocessing config expects `HEST_ROOT` to contain `HEST_v1_3_0.csv`, `wsis/`, `transcripts/`, `xenium_seg/` and `cellvit_seg/`.

`manifest.csv` has one row per section:

| Column | Meaning |
|---|---|
| `id` | Section ID, e.g. `TENX201` |
| `wsi` | Path to registered H&E TIFF |
| `pixel_size_um` | Positive microns per level-0 H&E pixel |
| `output_dir` | Optional path to prepared artifacts |
| `xenium_cell_seg` | Registered cell geometry Parquet, required by HEST input checks and VOLTA crops |
| `transcripts`, `cellvit_seg` | Input Parquet paths needed for preprocessing |

Each `<id>/` directory contains:

- `<id>_terra.npz`: `cell_ids` (unique, length N), `cell_emb` (N × 384), `he_x`
  and `he_y` (N, H&E level-0 pixels).
- `<id>_novae.npz`: Same as above but with NOVAE and 64 dimensions.
- `<id>.h5ad`: Cell IDs in `obs_names`, native spatial coordinates in
  `obsm['spatial']` (N × 2), gene counts in `X`, and H&E coordinates in `obs`.
- `<id>_granular_cell_labels.csv`: `cell_id,group`. These are the fine-grained
  reference labels used by evaluation.

`hest80_strict_v1.csv` has unique `sample_id,split` rows (`distill_train`, `distill_val`, `label_test`).
`hest_phenotype_split_v1.csv` lists the 15 annotated `sample_id` values.
Evaluation validates that the entire cohort is held out, then selects the 14 paper sections.

After installing `.[preprocess]` and TERRA in its supported environment:

```bash
export HEST_ROOT=/path/to/HEST-v1.3.0
CFG="$CELLBRIDGE_ROOT/configs/preprocess/preprocess.yaml"
python -m cellbridge.preprocess.01_find_samples --config "$CFG"
python -m cellbridge.preprocess.02_quality_check_streaming --config "$CFG" --sample-id TENX201
python -m cellbridge.preprocess.04_terra_embed --config "$CFG" --sample-id TENX201
```

Repeat per section.
QC discards unassigned/control transcripts and calls below QV 20.
TERRA harmonization filters genes detected in fewer than 10 cells and cells with fewer than 10 genes.
Step 03 creates coarse CellViT-derived labels and step 05 caches original UNI embeddings.
Neither is required for the LoRA training loader once teacher artifacts exist.

### NuCLS

The supplied preparation command downloads the pinned pathologist-corrected single-rater archive and creates the fixed split:

```bash
python -m cellbridge.inference.prepare.nucls "$CELLBRIDGE_DATA_ROOT/eval/nucls"
# Or supply the existing archive with --archive /path/to/NuCLS_dataset.zip.
```

The generated `nucls_info.csv` includes `row_id`, `sample_id`, `cell_uid`, `label`, `image_name`, `slide_id`, `split`, and geometry `x,y,x0,y0,x1,y1`.
Image names refer to files in `raw/rgb/`.
`row_id` is a stable row identity, `cell_uid` is globally unique, centers and boxes are in source-image pixels, and boxes must have positive area.
`raw/csv/` supplies `raw_classification`, `main_classification`, and `super_classification` for the granularity analysis.
Unlabeled annotations are excluded from evaluation.

### PanNuke, Lizard, and PhenoCell

Use the prepared [PathoCellBench datasets](https://huggingface.co/datasets/FabianReith/phenobench) and their supplied splits.
Preserve the legacy `phenocell` directory name.
PanNuke may require reassembling `data.mdb.part_*` into `data.mdb` as described in that dataset's README.

Each `<dataset>_info.csv` requires `sample_id,tile_id,cell_id,label` but `row_id` is optional and otherwise assigned from CSV row order.

Split CSVs use `sample_name,train_test_val_split`, with unique samples assigned `train`, `val` (or `valid`), or `test`.
Filenames:

| Dataset | File under `splits/` | Bootstrap unit |
|---|---|---|
| NuCLS | `train_test_val_split.csv` | `slide_id` |
| PanNuke | `train_test_val_split.csv` | `sample_id` |
| Lizard | `lizard_default_train_test_val_split.csv` | `sample_id` |
| PhenoCell | `phenocell_dataset_split.csv` | Region prefix `^(reg\d+)` of `sample_id` |

### Output folder structure

Embedding manifests under `embeddings/<arm>/manifest.csv` describe each sample's NPZ (`embeddings`, `cell_ids`, `row_ids`).

Classification `predictions.npz` stores `ids`, `y_true`, `y_pred`, `classes`, and `probabilities`.

Conformal archives additionally store boolean `<method>_sets` matrices, including `branch_graph_sets` (COF), and graph anchors.

```text
$CELLBRIDGE_OUTPUTS_ROOT/reproducibility/inference/
├── cellbridge/uni_terra_four_layers_adapted_all_cells/
│   ├── nucls/{embeddings,evaluation}/...
│   ├── pannuke/{embeddings,evaluation}/...
│   ├── lizard/{embeddings,evaluation}/...
│   ├── phenocell/{embeddings,evaluation}/...
│   └── hest/TENX.../{embeddings,evaluation}/...
├── lemon/vits8/<dataset>/...
└── <other-method>/all/<dataset>/...
```

## Experimental results

### Run the experiments

After [data preparation](#data-preparation), set the [paths](#paths), and obtain [model access](#model-access).
Keep the supplied ontology snapshot and mappings.
Run these commands from the repository root:

```bash
export CELLBRIDGE_DEVICE=cuda
bash scripts/reproduce.sh cellbridge check
bash scripts/reproduce.sh cellbridge run
```

`run` extracts embeddings, evaluates linear and kNN classifiers, and calibrates/evaluates conformal prediction on all five datasets.
Repeat both commands with `cellbridge` replaced by `uni`, `virchow2`, `volta`, `lemon`, `past`, or `spatialwhisperer`, using each model's compatible environment.

To train CellBridge yourself, run `cellbridge-train --config configs/train/lora/cellbridge.yaml`, then set `CELLBRIDGE_CHECKPOINT` to its `best.pt` before evaluation.

To reuse saved embeddings, replace `run` with `evaluate` or `conformal`.

For the **NuCLS label granularity experiment**, whose results are in Figure 3 of the manuscript, run both commands after classification evaluation:

```bash
cellbridge-granularity --config configs/analysis/granularity.yaml
cellbridge-granularity-refit --config configs/analysis/granularity.yaml
```

The first aggregates fine-label probabilities and the second refits classifiers at each
label granularity.
Both save `summary.csv` and `bootstrap_metrics.csv` under
`$CELLBRIDGE_OUTPUTS_ROOT/reproducibility/analysis/nucls_granularity/{aggregated,refit}/`.

### Paper tables and figures

Install the `analysis` extra and open the notebooks below after the runs finish.
They read the [output folder structure](#output-folder-structure).

| Paper result | Code / output |
|---|---|
| Figure 2 and Appendix Figure 6 — ID evaluation | [Notebook 05](notebooks/05_downstream_analysis.ipynb), downstream ID analysis. |
| Table 2 and Appendix Tables 10–11 — OOD evaluation | [Notebook 05](notebooks/05_downstream_analysis.ipynb), downstream OOD analysis. |
| Figure 3 — NuCLS label granularity | See above explanation. |
| Table 3 and Appendix Table 12 — COF efficiency and coverage | [Notebook 06](notebooks/06_conformal_prediction.ipynb), conformal prediction. |
| Figure 4 and Appendix Figures 7–13 — applied case study | [Notebook 07](notebooks/07_qualitative_conformal.ipynb), interactive qualitative analysis. |

## Ablation studies

Notebook 08 expects both the main TERRA run and the NOVAE run below.

### CovGap — Table 4

The main conformal run already evaluates `split` (standard), `graph` (global graph), and `branch_graph` (COF).
Rerun with the below:

```bash
bash scripts/reproduce.sh cellbridge conformal
```

The **CovGap ablation** section of [notebook 08](notebooks/08_ablation_study.ipynb) generates Table 4 from these saved results.

### Teacher choice — Table 5

In the NOVAE environment, set `SAMPLE_ID` to a training or validation section ID and run the following command.
Repeat for every training and validation section:

```bash
python -m cellbridge.preprocess.novae_embed \
  --config configs/preprocess/novae.yaml --sample-id "$SAMPLE_ID"
```

Then return to the CellBridge environment, train, and evaluate the NOVAE-guided adapter:

```bash
cellbridge-train --config configs/train/lora/novae.yaml
CELLBRIDGE_RUN_VERSION=uni_novae_four_layers_adapted_all_cells \
CELLBRIDGE_CHECKPOINT="$CELLBRIDGE_CHECKPOINTS_ROOT/uni_novae_four_layers_adapted_all_cells/best.pt" \
  bash scripts/reproduce.sh cellbridge run
```

The **teacher ablation** section of [notebook 08](notebooks/08_ablation_study.ipynb) compares this run with the main TERRA run and generates Table 5.
