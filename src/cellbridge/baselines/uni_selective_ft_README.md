# Selective UNI fine-tuning baseline

This is a separate supervised control for Cellbridge. Every classifier starts
from the original UNI2-h weights and trains only the final transformer block,
the encoder's final normalization, and a dataset-specific classification head.
It never loads TERRA targets, an ST teacher, or a Cellbridge checkpoint.

The registered comparison uses Cellbridge seed `0` only. HEST uses the existing
five-sample fast cohort, 0.5 um/px cell crops, sample-specific source classes,
and spatial 80/20 split. A stratified subset of the spatial training region is
used for epoch selection; the selected schedule is then refit on the complete
80% training region. PhenoCell and PanNuke use their official train/validation/
test field splits and the established 96px crops resized to 224px. After epoch
selection, their final models are refit on official train+validation.

Run the read-only contract check with:

```bash
scripts/baselines/uni_selective_ft/check.sh
```

Submit crop preparation, five HEST models, two PhenoCell tasks, one PanNuke
model, and final aggregation with:

```bash
scripts/baselines/uni_selective_ft/submit_parallel.sh
```

Results are written under `outputs/baselines/uni_selective_ft`. Each result
directory contains metrics, per-class F1, test predictions/probabilities,
training history, the resolved config, and a compact checkpoint containing only
the trainable UNI block/norm/head tensors. The main `all` tasks retain all 14
PhenoCell classes and all 5 PanNuke classes; `phenocell/immune` is the existing
7-class auxiliary evaluation.
