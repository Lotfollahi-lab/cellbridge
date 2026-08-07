# TENX116 feasibility result

The pilot used 12,000 spatial-training cells and 4,096 spatial-test cells from
TENX116, with 32 neighbouring H&E cells per condition and three seeds. The
TERRA targets and all normalization statistics were fit on training cells only.

## Three-seed results

| Metric | Base flow | Flow + SoftREPA |
|---|---:|---:|
| Matched native endpoint MSE | 0.2577 ± 0.0074 | 0.2766 ± 0.0087 |
| Mismatched native endpoint MSE | 0.3095 ± 0.0107 | 0.3618 ± 0.0110 |
| Relative matched-error separation | 16.74% ± 0.69% | 23.56% ± 0.26% |
| 32-way retrieval top-1 | 12.89% ± 2.40% | 13.48% ± 2.03% |
| Generated-latent cosine | 0.8726 ± 0.0016 | 0.8343 ± 0.0067 |
| Generated-latent MSE | 0.1831 ± 0.0036 | 0.2292 ± 0.0099 |
| Phenotype macro-F1 | 0.2298 ± 0.0024 | 0.2289 ± 0.0010 |
| Phenotype balanced accuracy | 0.2586 ± 0.0010 | 0.2596 ± 0.0015 |

The non-generative comparison arms were TERRA macro-F1 0.7132 / balanced
accuracy 0.7378 and direct UNI2 H&E macro-F1 0.2335 ± 0.0018 / balanced
accuracy 0.2526 ± 0.0025.

The ordinary flow's matched-versus-mismatched diagnostic passed in all seeds.
SoftREPA increased relative error separation and top-1 retrieval in all seeds,
but it increased absolute matched error, worsened both latent-fidelity metrics
in all seeds, and reduced mean macro-F1. It improved both phenotype metrics in
only one of three seeds.

**Decision: do not scale to additional tissues.** The aggregate gate is false
because SoftREPA did not improve both mean macro-F1 and mean balanced accuracy.

Per-class values are written to `three_seed_per_class_f1.csv` by:

```bash
PYTHONPATH=src ../../envs/terra-dev/bin/python -m cellbridge.flow.summary \
  --output-dir outputs/flow/terra_cell_flow_tenx116 --seeds 0 1 2
```
