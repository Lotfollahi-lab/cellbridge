# SpatialWhisperer baseline

This integration extracts SpatialWhisperer's **normalized, projected 2048-D
image embedding** and evaluates exactly that representation with the existing
cell-phenotyping protocol:

- HEST `label_test_fast_v1`: a within-sample spatial split, linear probe, and
  cosine 20-nearest-neighbour classifier;
- PhenoCell OOD: the official patient split for both `all` and `immune` tasks,
  using the same two classifiers.

No PCA is applied to SpatialWhisperer. Both classifiers receive the same saved
feature matrix, so extraction is done once and probing is cheap and repeatable.

> [!NOTE]
> The model card describes a 512-D shared space, but the pinned released
> checkpoint, its projection-head tensors, and the authors' seed-training
> configuration all specify 2048 dimensions. This integration follows and
> validates the executable checkpoint: the second normalized image output is
> 2048-D. A future genuine 512-D checkpoint must use a separate revision and
> output directory.

> [!CAUTION]
> **The fast HEST set is held out from TERRA training, not necessarily from
> SpatialWhisperer pretraining. SpatialWhisperer was trained with HEST-1K
> image--transcriptome pairs. Treat its HEST score as pretraining-exposed and
> do not describe it as a clean zero-shot or external-held-out result unless
> the SpatialWhisperer training manifest proves that these five samples (and
> related slides) were absent. Report PhenoCell separately as the OOD result.**

## 1. Prepare the external repository

The expected checkout is:

```text
/lustre/scratch126/cellgen/lotfollahi/so25/
├── projects/terra-multi/
└── repos/spatialwhisperer/
```

Keep the authors' repository as `upstream` and your fork as `origin`:

```bash
cd /lustre/scratch126/cellgen/lotfollahi/so25/repos/spatialwhisperer
git remote -v
git rev-parse --short HEAD
```

The current local checkout was inspected at Git revision `32991ab`. Before a
run, record `git rev-parse HEAD` in your experiment notes; if you intentionally
move to another revision, keep that exact commit with the results.

Install Pixi once if it is not already available, then build the locked model
environment (the first install is about 5 GB):

```bash
curl -fsSL https://pixi.sh/install.sh | bash
source ~/.bashrc
cd /lustre/scratch126/cellgen/lotfollahi/so25/repos/spatialwhisperer
pixi install
```

PhenoCell is stored in LMDB, while the upstream Pixi environment does not
currently declare the Python `lmdb` package. Make this small dependency change
on a branch in **your fork**, and commit both lock files so every cluster job
gets the same environment:

```bash
cd /lustre/scratch126/cellgen/lotfollahi/so25/repos/spatialwhisperer
git switch -c terra-multi-baseline  # omit if this branch already exists
pixi add --pypi lmdb
git add pixi.toml pixi.lock
git commit -m "Add LMDB for terra-multi PhenoCell evaluation"
git push -u origin terra-multi-baseline
```

If you already made that change on a fork branch, switch to it instead of
creating another branch. Record the resulting local repository commit with
the run provenance.

## 2. Authorize the model downloads

SpatialWhisperer's own checkpoint is public, but its loader also needs gated
UNI2 and Geneformer components. Accept the terms for
[`MahmoodLab/UNI2-h`](https://huggingface.co/MahmoodLab/UNI2-h), create a
read-only Hugging Face token, and either log in inside the Pixi environment or
export the token before running/submitting jobs:

```bash
cd /lustre/scratch126/cellgen/lotfollahi/so25/repos/spatialwhisperer
pixi run huggingface-cli login

# Alternatively, especially for LSF jobs:
export HF_TOKEN=hf_your_read_token
```

Never add a token to YAML, a shell script, Git, or an LSF log. `HF_HOME` may be
set to a shared, persistent path if compute nodes do not share the default
cache.

## 3. Check paths and prepare weights

All commands are run from `projects/terra-multi`. The wrapper selects the
external Pixi environment and adds `terra-multi/src` to `PYTHONPATH`:

```bash
cd /lustre/scratch126/cellgen/lotfollahi/so25/projects/terra-multi
CONFIG=configs/baselines/spatialwhisperer.yaml
RUNNER=scripts/baselines/spatialwhisperer/run_in_env.sh

$RUNNER check --config "$CONFIG"
$RUNNER prepare --config "$CONFIG"
```

`check` is read-only and validates the repository, data paths, split, and
configuration. `prepare` downloads/caches the model components and verifies
that the image encoder can produce a finite, unit-normalized 2048-D embedding.
The config pins the Hugging Face checkpoint to immutable revision
`23107503e5679924e33a118aa90d70d5c168ceeb`. Generated provenance records the
resolved revision and checkpoint checksum. The first `prepare` run can take
several minutes; later runs use the cache.

The CLI always expects common flags **after** its subcommand, as shown above.

## 4. Run a small smoke test

Use a separate output directory so partial artifacts cannot be mistaken for a
complete benchmark:

```bash
SMOKE=outputs/baselines/spatialwhisperer_smoke

$RUNNER extract-hest \
  --config "$CONFIG" \
  --output-dir "$SMOKE" \
  --device cuda \
  --samples TENX116 \
  --max-cells-per-sample 128

$RUNNER extract-phenocell \
  --config "$CONFIG" \
  --output-dir "$SMOKE" \
  --device cuda \
  --samples reg026_B.hdf \
  --max-cells-per-sample 128
```

This checks WSI access, LMDB reconstruction, SpatialWhisperer's released image
path, GPU inference, and artifact writing. A deliberately truncated smoke run
is not valid for probing or reporting.

## 5. Run the complete baseline

For a sequential run on one allocated GPU:

```bash
$RUNNER run --config "$CONFIG" --device cuda
```

The explicit equivalent is useful when resuming or debugging:

```bash
$RUNNER extract-hest --config "$CONFIG" --device cuda
$RUNNER extract-phenocell --config "$CONFIG" --device cuda
$RUNNER probe --config "$CONFIG" --dataset all
```

Extraction is signature-aware: rerunning a completed command should validate
and reuse current artifacts. Add `--overwrite` only when you intentionally want
to replace them. To run just one dataset's classifiers:

```bash
$RUNNER probe --config "$CONFIG" --dataset hest
$RUNNER probe --config "$CONFIG" --dataset phenocell
```

### LSF (recommended for the full fast benchmark)

The launcher submits one preparation job, five dependent HEST GPU extraction
jobs (one per fast sample), one dependent PhenoCell GPU extraction job, and a
final CPU probe job that waits for all six exports:

```bash
scripts/baselines/spatialwhisperer/submit_parallel.sh "$CONFIG"
```

Defaults can be overridden without editing the script, for example:

```bash
SPATIALWHISPERER_GPU_QUEUE=training-normal \
SPATIALWHISPERER_GPU_MEM_GB=96 \
SPATIALWHISPERER_PROBE_MEM_GB=128 \
scripts/baselines/spatialwhisperer/submit_parallel.sh "$CONFIG"
```

The launcher also reads the Git-ignored `scripts/cluster_env.sh` when present
for `BGROUP`, the CPU queue, and the `terra-dev` environment path. Logs are
written below `outputs/baselines/spatialwhisperer/logs/`.

## 6. Understand the image inputs

The released SpatialWhisperer checkpoint is context-only. For HEST, each cell
is represented at the scale expected by its UNI2 image path:

- 224 x 224 pixels at 0.5 micrometres/pixel (112 micrometres FoV).

For PhenoCell, physical pixel size is not supplied. The integration therefore
preserves the existing benchmark's 96-pixel context field of view, resizes it
to 224, and sends that single context view to the model. This convention must
remain fixed when comparing models.

The extractor saves the second return value of
`get_image_features(..., normalize_embeds=True)`: the shared projected
embedding, not UNI2's unprojected feature vector.

## 7. Results and reporting checklist

The default run is rooted at:

```text
outputs/baselines/spatialwhisperer/
├── hest_label_test_fast/
│   ├── features/         # one embedding artifact per HEST sample
│   ├── probes/           # per-sample/per-method probe products
│   └── probe_metrics*.csv
├── phenocell/
│   ├── features/         # one embedding artifact per PhenoCell sample
│   ├── probes/           # all/immune, linear/kNN probe products
│   └── probe_metrics*.csv
├── summary_linear.csv    # combined HEST + PhenoCell linear summary
├── summary_knn.csv       # combined HEST + PhenoCell kNN summary
└── logs/                 # LSF stdout/stderr
```

Before reporting a table, verify that:

1. all five HEST samples are present: `TENX116`, `TENX191`, `TENX195`,
   `TENX196`, and `TENX197`;
2. both `linear` and `knn` results exist, with cosine `k=20` for kNN;
3. PhenoCell contains both `all` and `immune` tasks using the official patient
   split;
4. the feature metadata says 2048 dimensions, normalized, and `pca=false`;
5. the exact SpatialWhisperer Git revision and config are retained alongside
   the scores;
6. the HEST table is clearly labelled **pretraining-exposed**.

For reproducible comparisons, do not select a different crop size, choose `k`
from the test data, fit PCA on all cells, or mix embeddings from different
model revisions in one output directory.
