# Reproducing the manuscript

The configs preserve the manuscript's model/probe settings. Full numerical
reproduction additionally needs the exact data artifacts, split tables, pretrained
weights, software versions and saved adapter. These large inputs are not in this
repository. Read [AUDIT.md](AUDIT.md) for unresolved provenance questions.

## Models

Install the `models,data` extras and `requirements/encoders.txt` in Python 3.11/3.12.
The latter pins Mahmood Lab TRIDENT to the commit recorded in the training
environment. Its [source installation](https://github.com/mahmoodlab/TRIDENT)
is separate from this package's PyPI-compatible dependencies. The requirements
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

Hibou-L remains an optional implementation but is excluded from the paper's six
baselines. Baseline profiles specify their own crop rules: HEST UNI2-h/CellBridge
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

## Main experiment

1. Prepare [data and exact split artifacts](data.md). Set the five path variables
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
4. Open notebooks 07 and 08 from the repository to generate the classification and
   conformal tables. Notebook 10 collects the teacher and conformal-construction
   ablations. Notebook 09 provides qualitative review.

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
| Uncertainty | 1,000 grouped resamples, seed 0; fixed fitted model/calibration. Group units are in [data.md](data.md). |

HEST reports section-level variation, not cell-wise bootstrap uncertainty.
Notebook 07's classification summary uses section SD; conformal/ablation summaries
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
  `CELLBRIDGE_CHECKPOINT` pointing to that run's `best.pt`. Notebook 10 compares
  this directory to the main TERRA run.
- **Conformal constructions:** a conformal run saves global, class, branch and
  hybrid split/graph methods. Notebook 10 selects methods via `COVGAP_METHODS`;
  no new embeddings are needed.
- **Supervised UNI control:** `cellbridge-finetune train --config
  configs/train/finetune/uni.yaml --dataset-config configs/inference/nucls.yaml`.
  This is a separate supervised control, not the main CellBridge training route.
- **Qualitative:** `data.py` loads/aligns predictions, `figures.py` renders them,
  and `views.py` provides the three notebook controls. `Explorer` shows patches,
  `TissueMap` locates regions, and `TerraView` adds transcriptomic context. Optional
  `metric="depth"` is a ground-truth-dependent diagnostic; the paper-facing
  `metric="set_size"` reads the saved final sets. A missing training config is
  recorded as missing provenance rather than blocking review. Save buttons export
  figures, tables and provenance to a new timestamped directory.

Notebooks are available in the repository/source archive. Some publication plots
in notebook 07 use LaTeX; install a TeX distribution or disable that notebook's
`text.usetex` setting. Notebook 09's existing interactive controls are preserved.

## Release validation

Run `python -m pytest`, `python -m build`, and `python -m twine check dist/*`.
Install the wheel into a clean environment, change out of the checkout, run
`cellbridge-infer --help`, and copy configs with `cellbridge-config`.
This checks distribution completeness; it does not substitute for a full
GPU/data reproduction. Public access to the hosted adapter and download
destinations for the exact data/split artifacts must be finalized before
describing this as a completed public release.
