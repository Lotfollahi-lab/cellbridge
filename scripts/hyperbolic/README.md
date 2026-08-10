# Hyperbolic CellBridge

Install the optional geometry dependency in the training environment:

```bash
python -m pip install -r requirements-hyperbolic.txt
```

The reference run is configured in
`configs/hyperbolic/cellbridge.yaml`. Curvature, curvature bounds, tangent-norm
clamping, tangent dimension, cone `K`, cone loss weight, and the minimum
Poincare radius for ontology apices are explicit there. The latter is enforced
as a curvature-aware Lorentz radius after every optimizer step.

Generate the six requested ablation configs:

```bash
python scripts/hyperbolic/run_ablations.py
```

Submit them through the existing train-then-evaluate LSF workflow:

```bash
python scripts/hyperbolic/run_ablations.py --action submit
```

Use `--only` to run a subset. `--output-root`, `--checkpoint-root`, and
`--wandb-group` keep replacement suites isolated from prior artifacts while
using the same script.

`--action local` instead runs training and evaluation sequentially in the
current environment. Evaluation keeps the existing HEST and OOD linear/kNN
probes, and additionally writes `hierarchy_metrics.csv`,
`hierarchy_metrics.json`, exact sampled triplets, and sampled cell IDs under
the normal evaluation directory.

## Breast-only geometry screen

The patient-disjoint screening suite trains only on annotated breast samples
TENX191–196, validates on TENX197–199, and evaluates only the held-out
TENX200–202 cohort. It compares matched d64 Euclidean, hyperbolic, and
hyperbolic-with-cones arms without running the broader OOD suite:

```bash
python scripts/hyperbolic/run_breast_only.py --action submit
```

Set `LORA_QUEUE` and `LORA_EVAL_QUEUE` to override the shared launcher queues.
