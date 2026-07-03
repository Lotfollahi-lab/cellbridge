import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from terra_multi.helpers.io import (
    align_by_cell_id,
    load_coords,
    load_embeddings,
    load_targets,
    select_hvgs,
)
from terra_multi.helpers.metrics import per_gene_pcc, summarize
from terra_multi.helpers.probe import fit_probe, predict
from terra_multi.helpers.splits import spatial_split

MODELS = ("terra", "uni", "concat")


def main():
    parser = argparse.ArgumentParser(description="Run one linear probe.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True, choices=MODELS)
    args = parser.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    model = args.model

    terra_emb, terra_ids = load_embeddings(
        cfg["terra_embeddings"], key=cfg.get("terra_key")
    )
    uni_emb, uni_ids = load_embeddings(cfg["uni_embeddings"])
    coords_all, coord_ids = load_coords(cfg["adata_path"])

    (terra_al, uni_al, coords), common_ids = align_by_cell_id(
        [(terra_emb, terra_ids), (uni_emb, uni_ids), (coords_all, coord_ids)]
    )
    print(f"[{model}] common cells across terra/uni/target: {len(common_ids):,}")

    if model == "terra":
        X = terra_al
    elif model == "uni":
        X = uni_al
    else:  # concat
        X = np.concatenate([terra_al, uni_al], axis=1)
    print(f"[{model}] feature matrix: {X.shape}")

    # --- targets + spatial split (shared across models) ------------------------
    gene_list = select_hvgs(cfg["adata_path"], cfg["n_genes"])
    Y = load_targets(cfg["adata_path"], common_ids, gene_list)
    train_idx, test_idx = spatial_split(
        common_ids, coords, cfg["test_frac"], cfg["split_axis"]
    )
    print(
        f"[{model}] genes={len(gene_list)} train={len(train_idx)} test={len(test_idx)} "
        f"(held out top {cfg['test_frac']:.0%} along {cfg['split_axis']})"
    )

    # --- fit / predict / score -------------------------------------------------
    model_obj = fit_probe(X[train_idx], Y[train_idx], cfg["alphas"])
    Y_pred = predict(model_obj, X[test_idx])
    Y_test = Y[test_idx]

    summary = {
        "model": model,
        "n_features": int(X.shape[1]),
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
        **summarize(Y_test, Y_pred),
    }
    pcc = per_gene_pcc(Y_test, Y_pred)

    out_dir = Path(cfg["output_dir"]) / model
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([summary]).to_csv(out_dir / "metrics.csv", index=False)
    np.savez(
        out_dir / "per_gene_pcc.npz",
        per_gene_pcc=pcc,
        gene_names=np.asarray(gene_list),
    )

    print(f"\n[{model}] summary")
    for k, v in summary.items():
        print(f"  {k:22s} {v}")
    print(
        f"[{model}] wrote {out_dir / 'metrics.csv'} and {out_dir / 'per_gene_pcc.npz'}"
    )


if __name__ == "__main__":
    main()
