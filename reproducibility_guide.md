# Reproducibility guide

The configs preserve the manuscript's model/probe settings. Full numerical
reproduction additionally needs the exact data artifacts, split tables, pretrained
weights, software versions and saved adapter. These large inputs are not in this
repository.

- [Installation options](#installation-options)
- [Models and access](#models)
- [Dataset details](#dataset-details)
- [Main experiment](#main-experiment)
- [Protocol checks](#protocol-checks)
- [Analyses and ablations](#analyses-and-ablations)
- [Release validation](#release-validation)

## Installation options

Follow the [README installation steps](README.md#-installation) with Python
3.11 or 3.12. For GPU training or image inference, install a PyTorch/torchvision
build appropriate for your GPU before installing `.[models,data]`.

For CPU evaluation of saved OOD embeddings and conformal prediction, the core
package is sufficient:

```bash
python -m pip install .
```

Add extras for the workflows you need, for example
`python -m pip install '.[preprocess,analysis]'`. Use `-e` when developing locally.

| Extra | Provides |
|---|---|
| `models` | PyTorch, torchvision, timm and model-loading dependencies |
| `data` | HEST/AnnData, whole-slide TIFF, geometry and LMDB readers |
| `preprocess` | `data` plus Scanpy and dataset preprocessing dependencies |
| `analysis` | `data` plus plotting, Jupyter, widgets and UMAP |
| `wandb` | Optional training logging |
| `dev` | Tests, linting and distribution build tools |

TERRA, NOVAE and SpatialWhisperer require their own installations, described below.
For building and installing a wheel, see [release validation](#release-validation).

## Models

Install the `models,data` extras and `requirements/encoders.txt` in Python 3.11/3.12.
The latter pins Mahmood Lab TRIDENT to the commit recorded in the training
environment. Its [source installation](https://github.com/mahmoodlab/TRIDENT)
is separate from this package's PyPI-compatible dependencies. The unrelated PyPI
package named `trident` is not a substitute. The requirements
also include [pyvips' binary installation](https://libvips.github.io/pyvips/README.html)
so TRIDENT can import without a separately installed libvips shared library.
CellBridge's HEST crops use tifffile/zarr, independently of that binary's slide-reader support.

| Model | Inputs / installation |
|---|---|
| CellBridge | Released `cellbridge.pt` adapter from [ogutsevda/cellbridge](https://huggingface.co/ogutsevda/cellbridge), plus the original UNI2-h weights loaded through TRIDENT. Obtain access to `MahmoodLab/UNI2-h` on Hugging Face and authenticate using the Hugging Face client. A local backbone can be set as `model.weights_path` in the training config or `load.encoder_weights` in the CellBridge inference profile. |
| UNI2-h | Same frozen backbone, `configs/inference/methods/uni.yaml`. |
| Virchow2 | Access to `paige-ai/Virchow2`; the config pins its model revision. |
| LEMON | `aliceblondel/LEMON`, pinned revision, `arch: vits8`. |
| PAST | Download its `checkpoint-context300.pth` into `$CELLBRIDGE_REPOS_ROOT/past_sc/checkpoints/`, or edit the profile. |
| VOLTA | Clone its repository at `$CELLBRIDGE_REPOS_ROOT/VOLTA` and place the ovarian `checkpoint.pth.tar` under `pretrained_weights/ovarian/`, or edit the profile. The adapter imports the repository's backbone code. |
| SpatialWhisperer | Install the authors' `spatialwhisperer` package and its model dependencies in a compatible environment. The adapter calls `load_spatialwhisperer_model` using `Good-Lab/spatialwhisperer`. |
| TERRA teacher | Install the authors' TERRA package exposing `download_pretrained`, `harmonize_adata`, `tokenize_adata`, and `embed_dataset`; preprocessing uses `Lotfollahi-lab/TERRA-112M`. It is unnecessary when teacher NPZ files already exist. |
| NOVAE teacher ablation | Install the authors' `novae` package in its supported environment. `configs/preprocess/novae.yaml` pins `prism-oncology/novae-human-0`. |

Baseline profiles specify their own crop rules: HEST UNI2-h/CellBridge
uses 112 µm at 0.5 µm/px; external UNI2-h/CellBridge uses 224 source pixels.
LEMON uses 10 µm / 40 pixels, VOLTA uses instance boxes, and PAST pads with black.

Recorded main-training versions include PyTorch **2.6.0+cu124**, torchvision
**0.21.0+cu124**, timm **0.9.16**, and TRIDENT **0.3.0** (commit pinned in the
requirements file). These are provenance, not a claim that every baseline supports
one shared environment. Retain each run's environment and resolved configuration.

The CellBridge profile pins adapter revision
`1c334ee0783be5aaae93a5562522d73fc92ee6c1`. `check`, `extract`, and `run` use the
Hugging Face cache; set `HF_HOME` to choose its location. The adapter repository
is currently private, so authenticate with an authorized account (`HF_TOKEN` or
the Hugging Face client). After caching both the adapter and backbone, set
`HF_HUB_OFFLINE=1` for offline use. The adapter uses checkpoint format
`cellbridge.lora/2` and contains the LoRA parameters and alignment head, not the
UNI2-h backbone. A local `--checkpoint /path/to/best.pt` takes precedence over
the Hub configuration; missing local files raise an error without falling back
to the released model.

## Dataset details

Paths below are relative to `CELLBRIDGE_DATA_ROOT`; YAML files under `configs/`
are the authoritative filenames and settings. Images and cell coordinates must
already be registered. Cell segmentation is outside CellBridge's scope.

### Paths and folder layout

Set the path variables shown in the README before running commands or Jupyter:

| Variable | Points to |
|---|---|
| `CELLBRIDGE_ROOT` | Project directory containing `configs/` |
| `CELLBRIDGE_DATA_ROOT` | Prepared training and evaluation data |
| `CELLBRIDGE_OUTPUTS_ROOT` | Training logs, extracted embeddings, evaluation results, and figures |
| `CELLBRIDGE_CHECKPOINTS_ROOT` | Checkpoints saved by your training runs |
| `CELLBRIDGE_REPOS_ROOT` | External baseline repositories and their local weights |

The released CellBridge adapter uses the Hugging Face cache; it does not need to
be placed in `CELLBRIDGE_CHECKPOINTS_ROOT`. See [model setup](#models)
for backbone access, cache settings, and external baseline requirements.

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

HEST's manifest points to each section's H&E image and pixel size; the images can
live outside this directory. Teacher NPZ files contain cell IDs, H&E coordinates,
and embeddings. Labels and split tables are separate files. External benchmarks
pair cell metadata with fixed train/validation/test assignments. The sections below
specify the exact files and columns for each dataset.

Data, split/cohort tables, and model weights are not bundled in the Python package.
Preserve the paper's original splits rather than generating new random partitions.

#### Using a wheel without a repository checkout

The wheel includes configuration files and ontology mappings. Copy them to a
writable project directory, then point `CELLBRIDGE_ROOT` there:

```bash
cellbridge-config /absolute/path/to/cellbridge-project
export CELLBRIDGE_ROOT="/absolute/path/to/cellbridge-project"
```

Existing configuration files are never overwritten. Set the other path variables
as above. If omitted, data and output paths default to the research workspace for
a source checkout, or to subdirectories of the current working directory for an
installed wheel. Explicit paths avoid depending on these defaults.

### HEST

Use the human Xenium/Xenium 5k cohort of HEST-1k v1.3.0, as specified in Appendix
A.1. The preprocessing config expects `HEST_ROOT` to contain
`HEST_v1_3_0.csv`, `wsis/`, `transcripts/`, `xenium_seg/` and `cellvit_seg/`.

`train/hest/manifest.csv` has one row per section:

| Column | Meaning |
|---|---|
| `id` | Section ID, e.g. `TENX201` |
| `wsi` | Path to registered H&E TIFF |
| `pixel_size_um` | Positive microns per level-0 H&E pixel |
| `output_dir` | Optional path to prepared artifacts; otherwise `<data_root>/<id>` |
| `xenium_cell_seg` | Registered cell geometry Parquet, required by HEST input checks and VOLTA crops |
| `transcripts`, `cellvit_seg` | Input Parquet paths needed for preprocessing |

Use absolute manifest paths: relative paths are interpreted from the process's
working directory. Each `<id>/` directory contains:

- `<id>_terra.npz`: `cell_ids` (unique, length N), `cell_emb` (N × 384), `he_x`
  and `he_y` (N, H&E level-0 pixels). TERRA review additionally reads
  `neighborhood_emb`. Embedding vectors must be finite and nonzero.
- `<id>.h5ad`: cell IDs in `obs_names`, native spatial coordinates in
  `obsm['spatial']` (N × 2), gene counts in `X`, and H&E coordinates in `obs`.
  The spatial holdout uses these native coordinates, not the H&E crop coordinates.
- `<id>_granular_cell_labels.csv`: `cell_id,group`. These are the fine-grained
  reference labels used by evaluation, not the coarse CellViT labels from step 03.
  They must match the reviewed section mapping in `configs/preprocess/cell_ontology/hest/`.

`hest80_strict_v1.csv` has unique `sample_id,split` rows (`distill_train`,
`distill_val`, `label_test`). `hest_phenotype_split_v1.csv` lists the 15 annotated
`sample_id` values; `organ` is optional for qualitative display. Evaluation
validates that the entire cohort is held out, then selects the 14 paper sections.
The exact research split/cohort tables and granular labels must be supplied;
this repository does not silently recreate them.

After installing `.[preprocess]` and TERRA in its supported environment:

```bash
export HEST_ROOT=/path/to/HEST-v1.3.0
CFG="$CELLBRIDGE_ROOT/configs/preprocess/preprocess.yaml"
python -m cellbridge.preprocess.01_find_samples --config "$CFG"
python -m cellbridge.preprocess.02_quality_check_streaming --config "$CFG" --sample-id TENX201
python -m cellbridge.preprocess.04_terra_embed --config "$CFG" --sample-id TENX201
```

Repeat per section. QC discards unassigned/control transcripts and calls below
QV 20. TERRA harmonization filters genes detected in fewer than 10 cells and
cells with fewer than 10 genes. Step 03 creates coarse CellViT-derived labels;
step 05 caches original UNI embeddings. Neither is required for the LoRA training
loader once teacher artifacts exist. GPU embedding steps require their model
installations; the cluster scripts additionally require the site's LSF/modules.

### NuCLS

The supplied preparation command downloads the pinned pathologist-corrected
single-rater archive, creates the fixed split, and records provenance:

```bash
python -m cellbridge.inference.prepare.nucls "$CELLBRIDGE_DATA_ROOT/eval/nucls"
# Or supply the existing archive with --archive /path/to/NuCLS_dataset.zip.
```

The generated `nucls_info.csv` includes `row_id`, `sample_id`, `cell_uid`,
`label`, `image_name`, `slide_id`, `split`, and geometry
`x,y,x0,y0,x1,y1`. Image names refer to files in `raw/rgb/`. `row_id` is a
stable row identity, `cell_uid` is globally unique, centers and boxes are in
source-image pixels, and boxes must have positive area. `raw/csv/` supplies
`raw_classification`, `main_classification`, and `super_classification` for the
granularity analysis. Unlabeled annotations are excluded from evaluation.

### PanNuke, Lizard and PhenoCell

Use the prepared [PathoCellBench datasets](https://huggingface.co/datasets/FabianReith/phenobench)
and their supplied splits. Preserve the legacy `phenocell` directory name.
PanNuke may require reassembling `data.mdb.part_*` into `data.mdb` as described in
that dataset's README. Do not substitute a new random split or different tiling.

Each `<dataset>_info.csv` requires `sample_id,tile_id,cell_id,label`; `row_id` is
optional and otherwise assigned from CSV row order. Any recorded `split` or
`split_id` must agree with the split file. Reordering metadata after extraction
invalidates automatically assigned row IDs.

Split CSVs use `sample_name,train_test_val_split`, with unique samples assigned
`train`, `val` (or `valid`), or `test`. Filenames:

| Dataset | File under `splits/` | Bootstrap unit |
|---|---|---|
| NuCLS | `train_test_val_split.csv` | `slide_id` |
| PanNuke | `train_test_val_split.csv` | `sample_id` |
| Lizard | `lizard_default_train_test_val_split.csv` | `sample_id` |
| PhenoCell | `phenocell_dataset_split.csv` | Region prefix `^(reg\d+)` of `sample_id` |

LMDB keys are zero-padded integers (e.g. `00000000`); each trusted pickle entry
contains `sample_name`, `tile_name` (ending `_TILE_<index>`), `image` (CHW), and
`instance_mask` (HW integer instance IDs). A sample's tiles must be consecutive.
The configured tile size is 224; `value_scale` is 1 for PanNuke/Lizard and 250
for PhenoCell. Geometry is reconstructed from instance masks and tile layout.

PanNuke uses 256 × 256 source fields; PhenoCell uses 1440 × 1920. Lizard's layout
is inferred from tile content. Existing PanNuke missing-instance exclusions are
preserved by `skip_missing_cells`. See the audit before changing these settings.

### Saved results

Embedding manifests under `embeddings/<arm>/manifest.csv` describe each sample's
NPZ (`embeddings`, `cell_ids`, `row_ids`). These IDs, and sample IDs in the
manifest, must continue to agree with the dataset metadata.

Classification `predictions.npz` stores `ids`, `y_true`, `y_pred`, `classes`, and
`probabilities`. Conformal archives additionally store boolean `<method>_sets`
matrices, including `branch_graph_sets` (COF), and graph anchors. All matrices
use the saved `classes` column order. Never align predictions by incidental row
order; the loaders validate and align IDs explicitly.

## Main experiment

1. Prepare [data and exact split artifacts](#dataset-details). Set the five path variables
   in the README. For paper runs, keep the provided ontology snapshot and mappings.
2. Use the released adapter (the inference default), or train
   `configs/train/lora/cellbridge.yaml`. For your own training run, set
   `CELLBRIDGE_CHECKPOINT` to its `best.pt` before running the commands below.
   Training uses 59 sections; six separate
   sections provide up to 5,000 validation cells each. The sampler uses full
   within-section batches and drops each section's incomplete final batch.
3. Run `bash scripts/reproduce.sh cellbridge check`, then
   `bash scripts/reproduce.sh cellbridge run`. Run the same stages for
   `uni virchow2 volta lemon past spatialwhisperer` in their compatible environments.
   The script defaults to `check`; it does not submit cluster jobs. It stops at the
   first failure. A `run` stage may take substantial time and writes to the selected
   result directory; use a new output root for an independent reproduction.
4. Open notebooks 05 and 06 from the repository to generate the classification and
   conformal tables. Notebook 08 collects the teacher and conformal-construction
   ablations. Notebook 07 provides qualitative review.

`CELLBRIDGE_DEVICE=cuda` selects the extraction device. `CELLBRIDGE_CHECKPOINT`
overrides the adapter for the CellBridge runner. For a different experiment,
`CELLBRIDGE_RUN_VERSION` selects a distinct results directory. `evaluate` and
`conformal` reuse existing embeddings and do not need image encoders on a GPU.
HEST evaluation still needs its labels and AnnData coordinates (`data` extra).

The result layout consumed by notebooks and granularity is:

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

For HEST, a result path ends in
`evaluation/<arm>/all/<probe>/<sample>/metrics.csv`. OOD paths end in
`evaluation/<arm>/<task>/<probe>/metrics.csv`. Conformal artifacts sit in the
sibling `conformal_prediction/` directory. The general inference CLI also supports
a shared HEST output directory; use the per-section layout above for notebooks.

## Protocol checks

| Stage | Protocol |
|---|---|
| Representation | Paper CellBridge uses `lora_adapted`; discard the projection head for probing. |
| Dimensionality | Training-only PCA to 256 dimensions for UNI2-h, CellBridge and Virchow2; baseline native dimensions otherwise. |
| HEST classification | Per-section spatial test stripe at the largest 20% of y coordinates; C selected by three-fold stratified CV on the fit partition. |
| OOD classification | Fixed train/val/test; C selected on val, then fit on train+val. |
| Linear probe | StandardScaler + multinomial logistic regression; C ∈ {0.1, 1, 10}, max_iter=2000. |
| kNN | k=20, cosine distance, uniform voting; same reference partitions as classification. |
| HEST conformal | Approximately 64% fit, 16% calibration, 20% test. The split is formed before per-split support filtering. |
| OOD conformal | Train/val/test become fit/calibration/test; C is selected within fit only. |
| Ontology | Pinned 2026-06-08 snapshot; reviewed dataset mappings and branch roots. |
| Conformal eligibility | Exclude unmapped classes and classes with fewer than 20 cells in any required split. |
| COF | `branch_graph`, α=0.1; candidate-wise branch thresholds followed by enlarging ontology closure. |
| Uncertainty | 1,000 grouped resamples, seed 0; fixed fitted model/calibration. Group units are in [the dataset split table](#pannuke-lizard-and-phenocell). |

HEST reports section-level variation, not cell-wise bootstrap uncertainty.
Notebook 05's classification summary uses section SD; conformal/ablation summaries
use section SE. OOD uncertainty files include macro AUROC when enough finite
replicates exist. Metrics are fractions except `coverage_gap`, which is already
in percentage points. Match table captions before comparing formatted values.

## Analyses and ablations

- **NuCLS hierarchy:** run `cellbridge-granularity`, then
  `cellbridge-granularity-refit` with `configs/analysis/granularity.yaml`. Each
  accepts an explicit output directory and saves `summary.csv` and
  `bootstrap_metrics.csv`; refitting also saves `audit.csv`.
  Both protocols use identical test cells and paired slide resamples. Use fresh
  directories when inputs change: `refit --resume` checks IDs but does not prove
  complete input/configuration equivalence. `common.py` shares scoring/bootstrap
  between the two commands. Neither command generates plots or narrative reports.
- **Teacher:** create NOVAE artifacts using
  `python -m cellbridge.preprocess.novae_embed --config configs/preprocess/novae.yaml --sample-id TENX201`
  (repeat for each training/validation section), then train
  `configs/train/lora/novae.yaml`. Evaluate with
  `CELLBRIDGE_RUN_VERSION=uni_novae_four_layers_adapted_all_cells` and
  `CELLBRIDGE_CHECKPOINT` pointing to that run's `best.pt`. Notebook 08 compares
  this directory to the main TERRA run.
- **Conformal constructions:** a conformal run saves the three paper methods:
  `split` (standard), `graph` (global graph), and `branch_graph` (COF).
  Notebook 08 reports their CovGap comparison; no new embeddings are needed.
- **Qualitative:** `data.py` loads/aligns predictions, `figures.py` renders them,
  and `views.py` provides the three notebook controls. `Explorer` shows patches,
  `TissueMap` locates regions, and `TerraView` adds transcriptomic context. Optional
  `metric="depth"` is a ground-truth-dependent diagnostic; the paper-facing
  `metric="set_size"` reads the saved final sets. A missing training config is
  recorded as missing provenance rather than blocking review. Save buttons export
  figures, tables and provenance to a new timestamped directory.

Notebooks are available in the repository/source archive. Some publication plots
in notebook 05 use LaTeX; install a TeX distribution or disable that notebook's
`text.usetex` setting. Notebook 07's existing interactive controls are preserved.

## Release validation

These commands build and validate a local distribution; they do not publish it to
PyPI. From the repository root:

```bash
python -m pip install '.[dev,preprocess,analysis]'
python -m pytest
python -m build
python -m twine check dist/*
```

In a separate clean environment, install the built wheel:

```bash
python -m pip install dist/cellbridge-0.1.0-py3-none-any.whl
```

Change out of the checkout, run
`cellbridge-infer --help`, and copy configs with `cellbridge-config`.
This checks distribution completeness; it does not substitute for a full
GPU/data reproduction. Public access to the hosted adapter and download
destinations for the exact data/split artifacts must be finalized before
describing this as a completed public release.
