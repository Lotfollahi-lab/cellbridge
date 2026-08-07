# Hyperbolic CellBridge

Install the optional geometry dependency in the training environment:

```bash
python -m pip install -r requirements-hyperbolic.txt
```

The reference run is configured in
`configs/hyperbolic/cellbridge.yaml`. Curvature, curvature bounds, tangent-norm
clamping, tangent dimension, cone `K`, and cone loss weight are explicit there.

Generate the six requested ablation configs:

```bash
python scripts/hyperbolic/run_ablations.py
```

Submit them through the existing train-then-evaluate LSF workflow:

```bash
python scripts/hyperbolic/run_ablations.py --action submit
```

`--action local` instead runs training and evaluation sequentially in the
current environment. Evaluation keeps the existing HEST and OOD linear/kNN
probes, and additionally writes `hierarchy_metrics.csv`,
`hierarchy_metrics.json`, exact sampled triplets, and sampled cell IDs under
the normal evaluation directory.
