# CellBridge

CellBridge learns cell representations from H&E images by distilling a spatial transcriptomic teacher into UNI2-h. Downstream probes predict cell labels, and ontology-aware conformal prediction returns sets of plausible labels.

The main paper model uses **TERRA, LoRA on the last four transformer blocks, all eligible training cells, and the adapted 1,536-dimensional image representation**. NOVAE is an alternative teacher using the same training pipeline.

## Workspace

Code lives in this repository; data, weights and generated results live outside Git. Configure another machine with:

```bash
export CELLBRIDGE_ROOT="$PWD"
export CELLBRIDGE_DATA_ROOT=/path/to/data
export CELLBRIDGE_CHECKPOINTS_ROOT=/path/to/checkpoints
export CELLBRIDGE_OUTPUTS_ROOT=/path/to/outputs
export PYTHONPATH="$CELLBRIDGE_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
```

For the lab layout `<workspace>/projects/cellbridge`, these roots default to sibling workspace directories. Use an environment with the dependencies for the selected workflow; packaging and installation specifications are still being consolidated.

```text
data/train/hest/<sample>/             # counts, coordinates and teacher embeddings
checkpoints/<model>/best.pt           # LoRA adapters and projection heads
outputs/reproducibility/
  train/<model>/                     # resolved config, metrics, summary and logs
  inference/cellbridge/<model>/       # per-dataset embeddings and evaluations
  analysis/                          # calibration, granularity and supporting analyses
outputs/visualizations/res_figures/   # notebook figure exports
```

Model directory names are `uni_terra_four_layers_adapted_all_cells` and `uni_novae_four_layers_adapted_all_cells`. The adapter checkpoints also require the original UNI2-h backbone weights.

## Workflows

- [Train and evaluate the main model](docs/training.md)
- [Reproduce the NOVAE teacher comparison](docs/novae.md)
- [Reproduce the NuCLS granularity experiment](docs/granularity.md)

Notebooks 07–10 cover downstream performance, conformal prediction, tissue inspection by prediction-set size, and teacher/conformal ablations. Notebook 07 compares the selected model with the baselines; retired 2L scaling and projected comparisons are omitted.

The code license is undecided. Dataset, backbone and teacher-model terms are separate.
