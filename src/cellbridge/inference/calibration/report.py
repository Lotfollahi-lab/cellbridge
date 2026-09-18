"""Aggregate the complete audit, check alignment and write comparison artifacts."""

import json
import hashlib
import subprocess
import importlib.metadata
from pathlib import Path

import numpy as np
import pandas as pd

from cellbridge.inference.calibration.metrics import METRICS


def interval(values):
    return {"bootstrap_se": float(np.std(values, ddof=1)),
            "ci_2_5": float(np.percentile(values, 2.5)),
            "ci_97_5": float(np.percentile(values, 97.5))}


def report(output):
    cases = json.loads((output / "cases.json").read_text())
    missing = [c["key"] for c in cases if not (output / "runs" / c["key"] / "audit.json").exists()]
    if missing:
        raise ValueError(f"{len(missing)} cases incomplete: {missing[:5]}")
    audits, points, draws, bins = [], [], [], []
    for case in cases:
        directory = output / "runs" / case["key"]
        context = {key: case[key] for key in ("dataset", "task", "method", "sample_id")}
        audits.append(json.loads((directory / "audit.json").read_text()))
        points.append(pd.read_csv(directory / "metrics.csv").assign(**context))
        draws.append(pd.read_csv(directory / "bootstrap.csv").assign(**context))
        bins.append(pd.read_csv(directory / "reliability.csv").assign(**context))
    audit = pd.DataFrame(audits)
    for key, group in audit.groupby(["dataset", "task", "sample_id"], dropna=False):
        if len(group) != 7:
            raise ValueError(f"expected seven methods for {key}")
        for column in ("train_ids_sha256", "validation_ids_sha256", "test_identity_sha256", "classes_sha256"):
            if group[column].nunique() != 1:
                raise ValueError(f"methods do not have identical {column}: {key}")
    audit.to_csv(output / "audit.csv", index=False)
    from cellbridge.inference.calibration.run import ROOT
    source_files = list((ROOT / 'src/cellbridge/inference/calibration').glob('*.py'))
    source_files += [ROOT / 'src/cellbridge/inference/evaluate' / f'{name}.py' for name in ('linear', 'ood', 'hest')]
    provenance = {
        'git_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'git_branch': subprocess.check_output(['git', 'branch', '--show-current'], cwd=ROOT, text=True).strip(),
        'versions': {name: importlib.metadata.version(name) for name in ('numpy', 'scipy', 'scikit-learn', 'pandas')},
        'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
        'cases_sha256': hashlib.sha256((output / 'cases.json').read_bytes()).hexdigest(),
    }
    (output / 'provenance.json').write_text(json.dumps(provenance, indent=2))
    points = pd.concat(points, ignore_index=True)
    draws = pd.concat(draws, ignore_index=True)
    bins = pd.concat(bins, ignore_index=True)
    points.to_csv(output / "per_section_metrics.csv", index=False)
    summary, all_draws = [], []
    for key, group in points.groupby(["dataset", "task", "method", "variant"], sort=False):
        ds, task, method, variant = key
        context = dict(zip(("dataset", "task", "method", "variant"), key))
        if ds == "hest":
            group = group.sort_values("sample_id")
            if len(group) != 14:
                raise ValueError("HEST must contain 14 sections")
            n = len(group)
            weights = np.random.default_rng(0).multinomial(n, np.full(n, 1/n), size=1000)
            replicate = pd.DataFrame(weights @ group[list(METRICS)].to_numpy() / n, columns=METRICS)
        else:
            replicate = draws.loc[(draws.dataset == ds) & (draws.task == task) &
                                  (draws.method == method) & (draws.variant == variant)].sort_values("bootstrap")
            replicate = replicate[list(METRICS)].reset_index(drop=True)
        if len(replicate) != 1000:
            raise ValueError(f"expected 1000 bootstrap draws: {key}")
        all_draws.append(replicate.assign(**context, bootstrap=np.arange(1000)))
        for metric in METRICS:
            summary.append({**context, "metric": metric, "estimate": float(group[metric].mean()),
                            "section_sd": float(group[metric].std(ddof=1)) if ds == "hest" else np.nan,
                            **interval(replicate[metric])})
    summary = pd.DataFrame(summary)
    all_draws = pd.concat(all_draws, ignore_index=True)
    summary.to_csv(output / "summary.csv", index=False)
    all_draws.to_csv(output / "bootstrap_metrics.csv", index=False)
    comparisons = []
    for (ds, task, variant), group in all_draws.groupby(["dataset", "task", "variant"]):
        ours = group.loc[group.method == "CellBridge"].sort_values("bootstrap")
        for method, other in group.groupby("method"):
            if method == "CellBridge":
                continue
            other = other.sort_values("bootstrap")
            for metric in METRICS:
                estimates = summary.loc[(summary.dataset == ds) & (summary.task == task) &
                                        (summary.variant == variant) & (summary.metric == metric)].set_index("method").estimate
                comparisons.append({"dataset": ds, "task": task, "variant": variant, "comparison": f"CellBridge - {method}",
                                    "metric": metric, "estimate": estimates['CellBridge'] - estimates[method],
                                    **interval(ours[metric].to_numpy() - other[metric].to_numpy())})
    for (ds, task, method), group in all_draws.groupby(["dataset", "task", "method"]):
        raw = group.loc[group.variant == 'raw'].sort_values('bootstrap')
        scaled = group.loc[group.variant == 'temperature_scaled'].sort_values('bootstrap')
        for metric in METRICS:
            estimates = summary.loc[(summary.dataset == ds) & (summary.task == task) &
                                    (summary.method == method) & (summary.metric == metric)].set_index('variant').estimate
            comparisons.append({"dataset": ds, "task": task, "variant": method, "comparison": 'temperature_scaled - raw',
                                "metric": metric, "estimate": estimates['temperature_scaled'] - estimates['raw'],
                                **interval(scaled[metric].to_numpy() - raw[metric].to_numpy())})
    pd.DataFrame(comparisons).to_csv(output / "paired_differences.csv", index=False)
    _write_markdown(output, summary, audit)
    _plot(output, bins)
    print(f"Wrote complete 7-method, 6-task comparison to {output}")


def _write_markdown(output, summary, audit):
    from cellbridge.inference.calibration.run import METHODS
    methods = list(METHODS)
    lines = ["# Single-model calibration audit", "",
             "CellBridge 4L / all / adapted and six baselines; linear probes on the existing conformal label tasks.", "",
             "Each cell is raw → temperature-scaled. ECE is top-label ECE with 15 equal-width bins, in percentage points; lower is better.",
             "Temperatures minimize held-out validation NLL. No test labels enter temperature fitting. Probe C is reused from the saved train-only selection.",
             "HEST averages 14 section metrics equally; external datasets use cell-weighted fixed-test metrics.", ""]
    order = [('hest', 'all'), ('nucls', 'all'), ('pannuke', 'all'), ('lizard', 'all'), ('phenocell', 'all'), ('phenocell', 'immune')]
    for metric, title, factor in [('ece_15', 'ECE (percentage points)', 100), ('nll', 'Negative log likelihood', 1),
                                   ('brier', 'Brier score (sum over classes)', 1), ('accuracy', 'Accuracy (%)', 100),
                                   ('macro_f1', 'Macro-F1 (%)', 100)]:
        lines += [f"## {title}", '', '| Task | ' + ' | '.join(methods) + ' |', '|---|' + '---:|' * len(methods)]
        selected = summary.loc[summary.metric == metric].set_index(['dataset', 'task', 'method', 'variant']).estimate
        for ds, task in order:
            values = [f"{factor * selected[ds, task, method, 'raw']:.3f} → {factor * selected[ds, task, method, 'temperature_scaled']:.3f}" for method in methods]
            lines.append(f"| {ds} {task} | " + ' | '.join(values) + ' |')
        lines.append('')
    lines += ["## Validation and interpretation", '',
              f"All {len(audit)} refitted probes matched the original test IDs, labels and class ordering. Maximum absolute probability difference: {audit.max_probability_error.max():.3g}.",
              "Train/validation/test identities match across all seven methods. Training and temperature-fitting rows are disjoint.",
              f"Refitted versus saved argmax differences: {int(audit.changed_argmax_predictions.sum())} cells across all cases. Probabilities are checked within absolute tolerance 0.001.",
              f"Cases with logistic-regression convergence warnings: {int(audit.get('convergence_warning', pd.Series(False, index=audit.index)).fillna(False).sum())}. These retain the existing iteration budget; see audit.csv.",
              f"Temperature fits at numerical bounds: {int(audit.temperature_at_bound.sum())}.",
              "`summary.csv` includes 95% percentile intervals from 1,000 grouped bootstrap draws; `paired_differences.csv` contains paired model and scaling differences.",
              "OOD groups follow each existing dataset config; HEST resamples sections. Intervals condition on the fitted probes and temperatures and do not include training or temperature-fitting uncertainty.",
              "All external tasks fit their probe and temperature on that dataset's own designated train/validation splits; this is not source-only calibration under distribution shift.",
              "The inherited conformal protocol selects supported classes using all three splits. This audit preserves that predefined evaluation population; it is not a newly selected test cohort.",
              "ECE bin sensitivity (10, 15, 20 bins) is included in the CSVs. Reliability plots for HEST pool cells for visualization, while headline HEST metrics weight sections equally.",
              "No representation training, ensembling or graph-set calibration is performed."]
    (output / 'results.md').write_text('\n'.join(lines) + '\n')


def _plot(output, bins):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from cellbridge.inference.calibration.run import METHODS
    plot_dir = output / 'figures'
    plot_dir.mkdir(exist_ok=True)
    selected = bins.loc[bins.n_bins == 15].copy()
    for (ds, task), group in selected.groupby(['dataset', 'task']):
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharex=True, sharey=True)
        for ax, variant in zip(axes, ['raw', 'temperature_scaled']):
            ax.plot([0, 1], [0, 1], '--', color='0.6', linewidth=1)
            for method in METHODS:
                rows = group.loc[(group.variant == variant) & (group.method == method)].copy()
                rows['correct_sum'] = rows.accuracy.fillna(0) * rows.n
                rows['confidence_sum'] = rows.confidence.fillna(0) * rows.n
                combined = rows.groupby('bin')[['n', 'correct_sum', 'confidence_sum']].sum()
                combined = combined.loc[combined.n > 0]
                ax.plot(combined.confidence_sum / combined.n, combined.correct_sum / combined.n, marker='.', label=method)
            ax.set(title=variant.replace('_', ' ').title(), xlabel='Mean confidence', xlim=(0, 1), ylim=(0, 1))
            ax.grid(alpha=.2)
        axes[0].set_ylabel('Observed accuracy')
        axes[1].legend(fontsize=7, loc='lower right')
        fig.suptitle(f"{ds.upper()} {task}" + (' — pooled cells' if ds == 'hest' else ''))
        fig.tight_layout()
        fig.savefig(plot_dir / f'{ds}_{task}.pdf')
        fig.savefig(plot_dir / f'{ds}_{task}.png', dpi=180)
        plt.close(fig)
