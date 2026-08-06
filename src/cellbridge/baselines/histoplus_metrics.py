"""Pure aggregation helpers for the HistoPLUS end-to-end baseline."""

from __future__ import annotations

import numpy as np


def _sum_class_rows(sample_metrics, key):
    totals = {}
    for metrics in sample_metrics:
        for row in metrics[key]:
            target = totals.setdefault(
                row["class"],
                {"class": row["class"], "tp": 0, "fp": 0, "fn": 0},
            )
            for name in ("tp", "fp", "fn"):
                target[name] += int(row[name])

    rows = []
    for target in totals.values():
        tp, fp, fn = target["tp"], target["fp"], target["fn"]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        )
        rows.append({**target, "precision": precision, "recall": recall, "f1": f1})
    return sorted(rows, key=lambda row: row["class"])


def aggregate_metrics(sample_metrics, dataset):
    """Aggregate counts and confusion matrices, never a mean of field scores."""
    if not sample_metrics:
        raise ValueError("cannot aggregate an empty metric sequence")
    summed = {
        key: int(sum(item[key] for item in sample_metrics))
        for key in (
            "n_reference",
            "n_scorable_reference",
            "n_predictions",
            "n_geometric_matches",
            "n_scorable_matches",
        )
    }
    precision = (
        summed["n_geometric_matches"] / summed["n_predictions"]
        if summed["n_predictions"]
        else 0.0
    )
    recall = (
        summed["n_geometric_matches"] / summed["n_reference"]
        if summed["n_reference"]
        else 0.0
    )
    coverage = (
        summed["n_scorable_matches"] / summed["n_scorable_reference"]
        if summed["n_scorable_reference"]
        else 0.0
    )
    matched_rows = _sum_class_rows(sample_metrics, "per_class_matched")
    end_rows = _sum_class_rows(sample_metrics, "per_class_end_to_end")
    matched_macro = (
        float(np.mean([row["f1"] for row in matched_rows])) if matched_rows else 0.0
    )
    result = {
        "dataset": dataset,
        "n_samples": len(sample_metrics),
        **summed,
        "coverage": coverage,
        "detection_precision": precision,
        "detection_recall": recall,
        "detection_f1": (
            2 * precision * recall / (precision + recall) if precision + recall else 0.0
        ),
        "matched_macro_f1": matched_macro,
        "coverage_adjusted_macro_f1": matched_macro * coverage,
        "end_to_end_macro_f1": (
            float(np.mean([row["f1"] for row in end_rows])) if end_rows else 0.0
        ),
        "classes": [row["class"] for row in end_rows],
        "per_class_matched": matched_rows,
        "per_class_end_to_end": end_rows,
    }
    if dataset == "pannuke":
        for key in ("panoptic_tp", "panoptic_fp", "panoptic_fn"):
            result[key] = int(sum(item[key] for item in sample_metrics))
        result["panoptic_iou_sum"] = float(
            sum(item["panoptic_iou_sum"] for item in sample_metrics)
        )
        tp, fp, fn = (
            result["panoptic_tp"],
            result["panoptic_fp"],
            result["panoptic_fn"],
        )
        result["dq"] = tp / (tp + 0.5 * fp + 0.5 * fn) if tp + fp + fn else 0.0
        result["sq"] = result["panoptic_iou_sum"] / tp if tp else 0.0
        result["pq"] = result["dq"] * result["sq"]
    return result
