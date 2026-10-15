"""Compare regrouped and directly fitted NuCLS granularity curves."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from cellbridge.config import load_config
from cellbridge.inference.granularity.run import DEFAULT_CONFIG, LEVELS


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--reuse-dir", type=Path)
    parser.add_argument("--refit-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--figure-dir", type=Path)
    args = parser.parse_args()
    settings = load_config(args.config)
    args.reuse_dir = args.reuse_dir or Path(settings["output_dir"])
    args.refit_dir = args.refit_dir or Path(settings["refit_output_dir"])
    output = args.output_dir or Path(settings["comparison_output_dir"])
    figure_dir = args.figure_dir or Path(settings["comparison_figure_dir"])
    protocols = {
        "Regroup existing probabilities": args.reuse_dir,
        "Fit probes to each label level": args.refit_dir,
    }
    output.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    tables = []
    contrasts = []
    colors = ["#bd2435", "#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b", "#7f7f7f"]
    for protocol, source in protocols.items():
        tables.append(pd.read_csv(source / "summary.csv").assign(protocol=protocol))
        contrasts.append(pd.read_csv(source / "paired_differences.csv").assign(protocol=protocol))
    summary = pd.concat(tables, ignore_index=True)
    methods = list(summary["method"].drop_duplicates())
    summary.to_csv(output / "summary.csv", index=False)
    pd.concat(contrasts, ignore_index=True).to_csv(output / "paired_differences.csv", index=False)
    for probe in ("linear", "knn"):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.8), sharey=True)
        for ax, protocol in zip(axes, protocols):
            selected = summary.query("probe == @probe and protocol == @protocol and metric == 'macro_f1'")
            for method, color in zip(methods, colors):
                rows = selected.loc[selected.method == method].set_index("level").loc[list(LEVELS)]
                ax.plot(range(3), rows.estimate * 100, marker="o", color=color, label=method,
                        linewidth=2.8 if method == "CellBridge" else 1.4)
                ax.fill_between(range(3), rows.ci_low * 100, rows.ci_high * 100, color=color, alpha=.07)
            ax.set_title(protocol, fontsize=11)
            ax.set_xticks(range(3), ["Super (5)", "Main (8)", "Raw (12)"])
            ax.set_xlabel("Annotation granularity")
            ax.grid(alpha=.2)
        selected = summary.query("probe == @probe and metric == 'macro_f1'")
        axes[0].set_ylim(np.floor(selected.ci_low.min() * 20) * 5,
                         np.ceil(selected.ci_high.max() * 20) * 5)
        axes[0].set_ylabel("Macro F1 (%)")
        fig.suptitle(f"NuCLS · {'Linear' if probe == 'linear' else 'kNN'} probes", fontsize=14)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
        fig.tight_layout(rect=(0, .14, 1, .94))
        for extension in ("png", "pdf"):
            fig.savefig(figure_dir / f"granularity_comparison_{probe}.{extension}", dpi=220, bbox_inches="tight")
        plt.close(fig)
    print(output)


if __name__ == "__main__":
    main()
