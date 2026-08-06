# Frozen UNI versus CellBridge conformal evaluation

This experiment fits the same fixed-C probabilistic linear classifier to frozen
UNI and CellBridge features. It does not update either encoder.

PhenoCell's official field split is used as follows:

- `train`: classifier fitting and representation-specific PCA fitting;
- `val`: conformal calibration only;
- `test`: final reporting only.

`Other cells` is excluded because it has no defensible biological placement.
The remaining benchmark classes are leaves of the explicit task hierarchy in
the YAML config. All paths have the same depth.

The standard method uses the finite-sample split-conformal quantile. The
hierarchical method implements the paper's full graph construction, including
closest-node tie breaking and the conformal-risk correction. Calibration checks
all probability breakpoints rather than an arbitrary lambda grid.

Run from the repository root:

```bash
module load cellgen/uv
source ../../envs/terra-dev/bin/activate
PYTHONPATH=src python -m cellbridge.hierarchical_conformal.run \
  --config configs/hierarchical_conformal/phenocell.yaml
```

Results are written as three compact tables: `summary.csv`,
`per_class_coverage.csv`, and `calibration.csv`.

## Annotated HEST breast cohort

The HEST analysis uses the reviewed Cell Ontology release `2026-03-26` labels
under `outputs/hierarchical/hest_label_test`. Only `full`, depth-three canonical
identities are treated as classifier leaves; partial and excluded annotations
are not promoted to fine-grained truth labels.

The patient-disjoint split is:

- `TENX191`–`TENX196` (patients 1–2): PCA and classifier fitting;
- `TENX197`–`TENX199` (patient 3): conformal calibration only;
- `TENX200`–`TENX202` (patient 4): final reporting only.

Export the otherwise uncached breast-training feature arms from the same
`lora_baseline` checkpoint used in the PhenoCell comparison:

```bash
PYTHONPATH=src python -m cellbridge.hierarchical_conformal.export_hest \
  --config configs/hierarchical/lora_baseline.yaml
```

Then run:

```bash
PYTHONPATH=src python -m cellbridge.hierarchical_conformal.run_hest \
  --config configs/hierarchical_conformal/hest_breast.yaml
```

The three cross-organ sections are not folded into this closed-class analysis:
they contain 28 depth-three ontology identities absent from breast training.
In addition to the PhenoCell tables, HEST writes `per_sample_coverage.csv` and
records train-only PCA explained variance in `calibration.csv`.
