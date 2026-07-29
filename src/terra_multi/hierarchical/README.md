# HEST hierarchical labels

This package turns every exact label in the 15-sample HEST `label_test` split into
a reviewed, auditable hierarchy:

```text
cell → compartment → lineage → canonical Cell Ontology identity
```

Numeric prefixes such as `0_` and `3_` are retained as `source_cluster`, but do
not define biological identities. State, malignancy, and tissue context are
separate columns. A `partial` label is usable only to `hierarchy_depth`; an
`excluded` label has no defensible identity and must not form a training pair.

Run locally from the repository root:

```bash
PYTHONPATH=src ../../envs/terra-dev/bin/python \
  -m terra_multi.hierarchical.build_labels \
  --config configs/hierarchical/hest_label_test.yaml
```

Use `--mapping-only` to generate the audit and summaries without the per-cell
Parquet files. On LSF, run:

```bash
bash scripts/hierarchical/submit_labels.sh
```

The output contains:

- `source_label_mapping.csv`: one row for each of the 166 exact source labels.
- `sample_label_counts.csv`: source-label counts within each sample.
- `canonical_usage.csv`: how source labels and cells collapse into identities.
- `hierarchy_nodes.csv`: the task-specific tree and edges.
- `hierarchy_tree.{dot,png}`: counted hierarchy visualization.
- `cells/<sample_id>.parquet`: per-cell hierarchy, state, and audit columns.
- `REPORT.md`: readable coverage and all 166 mapping decisions.

Regenerate the tree after rebuilding the label summaries with:

```bash
PYTHONPATH=src ../../envs/terra-dev/bin/python \
  -m terra_multi.hierarchical.plot_tree \
  --artifact-dir outputs/hierarchical/hest_label_test
```

The mapping is pinned to Cell Ontology release `2026-03-26`. It is intentionally
reviewed code rather than a live fuzzy matcher: a new source label causes a hard
failure until someone makes an explicit biological decision.

## Phenotype-aware LoRA protocol

The fixed development split is
`${TERRA_MULTI_DATA_ROOT}/train/hest/splits/hest_phenotype_split_v1.csv`.
It preserves all 59 original `distill_train` and six original `distill_val`
sections, then
partitions the 12 annotated breast sections by patient:

| Role | Samples | Patients | Purpose |
| --- | --- | --- | --- |
| breast train | TENX191–TENX196 | patients 1–2 | paired loss plus optional hierarchy loss |
| breast validation | TENX197–TENX199 | patient 3 | checkpoint and hyperparameter selection |
| breast test | TENX200–TENX202 | patient 4 | untouched in-domain evaluation |
| cross-organ test | TENX116, TENX143, TENX187 | pancreas, lymph node, ovary | held-out transfer evaluation |

The two training patients contain all full leaf identities observed in either
the validation or breast-test patient; broader partial labels remain usable
only through their known lineage.

The breast training sections use the same 5,000-cell per-section cap as the
base 59. No validation or test section enters either training loss. The
checkpoint metric is `breast_val/loss_total`; the original six-section
validation loss is still logged separately as `base_val/*`.

All four ablations use identical seeds, initialization, data, balanced
sample-level batching, and configuration values except for the two loss
switches:

| Config | Negative downweighting | Hierarchy ranking |
| --- | --- | --- |
| `lora_baseline.yaml` | off | off |
| `lora_negative_downweighting.yaml` | on | off |
| `lora_hierarchy_ranking.yaml` | off | on |
| `lora_combined.yaml` | on | on |

The exact H&E–TERRA diagonal remains the only positive in `L_pair`.
Downweighting changes only known off-diagonal denominator terms. The ranking
loss orders configurable relations and is added as
`L_total = L_pair + lambda_hier * L_hier`. Partial labels contribute only
through `hierarchy_depth`; excluded and unknown labels contribute neither
ranking relations nor softened negatives. Cross-sample comparisons are
preferred whenever the batch provides them.

Build the per-cell label artifacts once, then launch one run:

```bash
scripts/hierarchical/submit_labels.sh
scripts/hierarchical/submit_experiment.sh \
  configs/hierarchical/lora_combined.yaml
```

Launch all four matched ablations with:

```bash
scripts/hierarchical/submit_ablations.sh
```

The unchanged legacy command remains valid:

```bash
scripts/lora/submit_experiment.sh configs/lora/contrastive_k0.yaml
```

For hierarchy-enabled runs, evaluation is written separately under
`evaluation/hest_breast_val`, `evaluation/hest_breast_test`,
`evaluation/hest_cross_organ`, and `evaluation/phenocell`. The first three
reuse the existing HEST spatial train/test probe; PhenoCell remains the
independent OOD evaluation.
