"""Combined PAST probes and comparison tables."""

from __future__ import annotations

from cellbridge.baselines.evaluation import probe_hest
from cellbridge.baselines.volta_ood_probe import probe_ood
from cellbridge.evaluation.artifacts import write_table


def probe_all(cfg, root, dataset, samples=None):
    results = {}
    if dataset in {"all", "hest"}:
        results["HEST"] = probe_hest(
            cfg, root / "hest_label_test_fast", samples=samples
        )
    for name, display in (("phenocell", "PhenoCell"), ("pannuke", "PanNuke")):
        if dataset in {"all", name}:
            results[display] = probe_ood(cfg, root / name, name)
    if dataset == "all":
        for method in ("linear", "knn"):
            hest = results["HEST"][method]
            rows = [
                {
                    "baseline": cfg["baseline"]["name"],
                    "dataset": "HEST",
                    "task": "label_test_fast_5_sample_mean",
                    "method": method,
                    "macro_f1": hest["macro_f1"].mean(),
                    "balanced_accuracy": hest["balanced_accuracy"].mean(),
                }
            ]
            for display in ("PhenoCell", "PanNuke"):
                rows.extend(
                    {
                        "baseline": cfg["baseline"]["name"],
                        "dataset": display,
                        "task": row.task,
                        "method": method,
                        "macro_f1": row.test_macro_f1,
                        "balanced_accuracy": row.test_balanced_accuracy,
                    }
                    for row in results[display][method].itertuples(index=False)
                )
            write_table(root / f"summary_{method}.csv", rows)
    return results
