# Dataset details

Paths below are relative to `CELLBRIDGE_DATA_ROOT`; YAML files under `configs/`
are the authoritative filenames and settings. Images and cell coordinates must
already be registered. Cell segmentation is outside CellBridge's scope.

## Paths and folder layout

Set the path variables shown in the README before running commands or Jupyter:

| Variable | Points to |
|---|---|
| `CELLBRIDGE_ROOT` | Project directory containing `configs/` |
| `CELLBRIDGE_DATA_ROOT` | Prepared training and evaluation data |
| `CELLBRIDGE_OUTPUTS_ROOT` | Training logs, extracted embeddings, evaluation results, and figures |
| `CELLBRIDGE_CHECKPOINTS_ROOT` | Checkpoints saved by your training runs |
| `CELLBRIDGE_REPOS_ROOT` | External baseline repositories and their local weights |

The released CellBridge adapter uses the Hugging Face cache; it does not need to
be placed in `CELLBRIDGE_CHECKPOINTS_ROOT`. See [model setup](reproduction.md#models)
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

### Using a wheel without a repository checkout

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

## HEST

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

## NuCLS

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

## PanNuke, Lizard and PhenoCell

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

## Saved results

Embedding manifests under `embeddings/<arm>/manifest.csv` describe each sample's
NPZ (`embeddings`, `cell_ids`, `row_ids`). These IDs, and sample IDs in the
manifest, must continue to agree with the dataset metadata.

Classification `predictions.npz` stores `ids`, `y_true`, `y_pred`, `classes`, and
`probabilities`. Conformal archives additionally store boolean `<method>_sets`
matrices, including `branch_graph_sets` (COF), and graph anchors. All matrices
use the saved `classes` column order. Never align predictions by incidental row
order; the loaders validate and align IDs explicitly.
