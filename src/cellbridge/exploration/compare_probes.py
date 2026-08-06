import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

MODELS = ("terra", "uni", "concat")


def _load(results_dir):
    """Return ``(metrics_by_model, pcc_by_model, gene_names)``."""
    metrics = {}
    pcc = {}
    gene_names = None
    for m in MODELS:
        mdir = results_dir / m
        metrics[m] = pd.read_csv(mdir / "metrics.csv").iloc[0]
        z = np.load(mdir / "per_gene_pcc.npz", allow_pickle=False)
        pcc[m] = np.asarray(z["per_gene_pcc"], dtype=float)
        names = [str(g) for g in z["gene_names"]]
        if gene_names is None:
            gene_names = names
        elif names != gene_names:
            raise ValueError(f"gene order differs between models (at '{m}')")
    return metrics, pcc, np.asarray(gene_names)


def _print_table(metrics):
    header = (
        f"{'Model':12s} | {'Mean PCC':>8s} | {'Median PCC':>10s} | "
        f"{'% genes PCC>0.3':>15s} | {'RMSE':>8s}"
    )
    print(header)
    print("-" * len(header))
    for m in MODELS:
        r = metrics[m]
        print(
            f"{m:12s} | {r['mean_pcc']:8.4f} | {r['median_pcc']:10.4f} | "
            f"{r['pct_genes_pcc_gt_0.3']:15.1f} | {r['rmse']:8.4f}"
        )


def _bar_figure(pcc, gene_names, out_path):
    order = np.argsort(-np.nan_to_num(pcc["concat"], nan=-np.inf))
    labels = gene_names[order]
    x = np.arange(len(labels))
    width = 0.27
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.32), 4.5))
    for i, m in enumerate(MODELS):
        ax.bar(x + (i - 1) * width, pcc[m][order], width=width, label=m)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_ylabel("per-gene PCC (test)")
    ax.set_title("Per-gene expression-prediction PCC by model (sorted by concat)")
    ax.axhline(0, color="k", linewidth=0.6)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _scatter_figure(pcc, best_single, out_path):
    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.scatter(best_single, pcc["concat"], s=18, alpha=0.7)
    both = np.concatenate([best_single, pcc["concat"]])
    lo, hi = float(np.nanmin(both)), float(np.nanmax(both))
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="y = x")
    ax.set_xlabel("best single-modality per-gene PCC (max of TERRA, UNI)")
    ax.set_ylabel("concat per-gene PCC")
    ax.set_title("Does concatenation beat the best single modality?")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Compare the three probe runs.")
    parser.add_argument("--results_dir", required=True)
    args = parser.parse_args()
    results_dir = Path(args.results_dir)

    metrics, pcc, gene_names = _load(results_dir)
    _print_table(metrics)

    best_single = np.maximum(pcc["terra"], pcc["uni"])
    bar_path = results_dir / "per_gene_pcc_bar.png"
    scatter_path = results_dir / "concat_vs_best_single.png"
    _bar_figure(pcc, gene_names, bar_path)
    _scatter_figure(pcc, best_single, scatter_path)

    # Where does concatenation add the most over the better single modality?
    delta = pcc["concat"] - best_single
    rank = np.argsort(np.where(np.isnan(delta), -np.inf, delta))[::-1]
    print("\nGenes where concat helps most (concat PCC - best single PCC):")
    for j in rank[:10]:
        print(
            f"  {gene_names[j]:16s} concat={pcc['concat'][j]:+.3f} "
            f"terra={pcc['terra'][j]:+.3f} uni={pcc['uni'][j]:+.3f} "
            f"delta={delta[j]:+.3f}"
        )

    tm = float(metrics["terra"]["mean_pcc"])
    um = float(metrics["uni"]["mean_pcc"])
    cm = float(metrics["concat"]["mean_pcc"])
    beats_both = cm > tm and cm > um
    print(f"\nMean PCC -- terra={tm:.4f}  uni={um:.4f}  concat={cm:.4f}")
    print(
        f"Concat beats BOTH single modalities on mean PCC: {'YES' if beats_both else 'NO'}"
    )
    print(f"\nSaved figures:\n  {bar_path}\n  {scatter_path}")


if __name__ == "__main__":
    main()
