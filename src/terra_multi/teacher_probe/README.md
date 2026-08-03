# TERRA contextual gene-token teacher probe

This experiment asks whether contextual TERRA gene-token embeddings improve
phenotype classification beyond the pooled cell embedding and the same genes'
expression. It uses only TENX116 and never loads or trains UNI/CellBridge.

The installed `terra-st==0.1.7` implementation was inspected directly. Its
public `get_gene_embed(..., include_spatial_cell_emb=True)` call returns
`cell_emb_gene<ENSEMBL_ID>` and `spatial_cell_emb_gene<ENSEMBL_ID>`. The latter
is the selected cell-gene token after attending to the spatial neighborhood;
this is the contextual feature used here. Extraction checks the returned keys
instead of silently assuming them.

The registered config pins `Lotfollahi-lab/TERRA-112M` at revision
`00d1c977e49364ba10c28692abcbd102e115bf91`, matching the original TENX116
pooled-embedding run. It repeats the same raw-count harmonization thresholds
(`min_genes_per_cell=10`, `min_cells_per_gene=10`) and model-bundle tokenizer.

Run a quick end-to-end extraction and probe:

```bash
scripts/teacher_probe/submit_extract.sh \
  configs/teacher_probe/tenx116_gene_tokens.yaml smoke

# Submit after extraction finishes, or pass its job ID through the generic
# submitter's dependency mechanism when orchestrating both jobs.
scripts/teacher_probe/submit_probe.sh \
  configs/teacher_probe/tenx116_gene_tokens.yaml smoke
```

Use `full` instead of `smoke` for the registered 25,000-cell run. Selection is
a stable, label-blind hash of `cell_id`; tokenization and neighborhoods are
computed on the complete section first. The sparse HDF5 cache records explicit
token presence and stores only present contextual embeddings.

Evaluation uses five grouped 500-um spatial-block folds. Each gene's four-PC
PCA is fit only to token-present training cells. Absent rows receive zero PCs
plus an explicit presence bit. The repository's standardized
`LogisticRegressionCV` convention is then fit independently on each training
fold for all three arms.

`REPORT.md` and `summary.csv` contain aggregate results. The output also keeps
fold metrics, the paired key delta, per-class F1, long-form confusion matrices,
predictions, gene PCA diagnostics, coefficient-based gene importance, plots,
the resolved config, and a detailed alignment/presence report.

