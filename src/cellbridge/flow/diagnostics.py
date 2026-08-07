"""Summaries for matched-versus-mismatched flow endpoint errors."""

from __future__ import annotations

import numpy as np


def _summarize_rows(
    matrices: list[np.ndarray],
    masks: list[np.ndarray],
    selectors: list[np.ndarray],
    *,
    scale: float | None,
    temperature: float | None,
) -> dict[str, float | int]:
    matched: list[float] = []
    mismatched: list[float] = []
    top1: list[bool] = []
    top5: list[bool] = []
    better: list[bool] = []
    chance: list[float] = []
    candidates: list[int] = []
    matched_logits: list[float] = []
    mismatched_logits: list[float] = []
    probabilities: list[float] = []
    for errors, valid, selected in zip(matrices, masks, selectors, strict=True):
        for row in np.flatnonzero(selected):
            row_valid = valid[row].copy()
            row_valid[row] = True
            negatives = row_valid.copy()
            negatives[row] = False
            if not negatives.any():
                raise ValueError("each diagnostic row needs at least one valid negative")
            diagonal = float(errors[row, row])
            negative_errors = errors[row, negatives]
            ranking = np.argsort(np.where(row_valid, errors[row], np.inf))
            candidate_count = int(row_valid.sum())
            matched.append(diagonal)
            mismatched.extend(negative_errors.tolist())
            better.append(diagonal < float(negative_errors.mean()))
            top1.append(ranking[0] == row)
            top5.append(row in ranking[: min(5, candidate_count)])
            chance.append(1.0 / candidate_count)
            candidates.append(candidate_count)
            if scale is not None and temperature is not None:
                logits = scale * np.exp(-errors[row] / temperature)
                valid_logits = logits[row_valid]
                shifted = valid_logits - valid_logits.max()
                numerator = np.exp(logits[row] - valid_logits.max())
                probabilities.append(float(numerator / np.exp(shifted).sum()))
                matched_logits.append(float(logits[row]))
                mismatched_logits.extend(logits[negatives].tolist())
    if not matched:
        raise ValueError("diagnostic selection contains no states")
    match = np.asarray(matched)
    mismatch = np.asarray(mismatched)
    successes = int(np.sum(top1))
    count = len(top1)
    rate = successes / count
    z = 1.96
    denominator = 1 + z * z / count
    center = (rate + z * z / (2 * count)) / denominator
    half_width = z * np.sqrt(
        rate * (1 - rate) / count + z * z / (4 * count**2)
    ) / denominator
    result: dict[str, float | int] = {
        "matched_error_mean": float(match.mean()),
        "mismatched_error_mean": float(mismatch.mean()),
        "relative_error_reduction": float(
            (mismatch.mean() - match.mean()) / mismatch.mean()
        ),
        "matched_better_fraction": float(np.mean(better)),
        "retrieval_top1": float(rate),
        "retrieval_top1_successes": successes,
        "retrieval_top1_ci95_low": float(center - half_width),
        "retrieval_top1_ci95_high": float(center + half_width),
        "retrieval_top5": float(np.mean(top5)),
        "retrieval_candidates_mean": float(np.mean(candidates)),
        "retrieval_candidates_min": int(np.min(candidates)),
        "retrieval_candidates_max": int(np.max(candidates)),
        "chance_top1": float(np.mean(chance)),
        "n_states": count,
    }
    if probabilities:
        result.update(
            matched_logit_mean=float(np.mean(matched_logits)),
            mismatched_logit_mean=float(np.mean(mismatched_logits)),
            matched_probability_mean=float(np.mean(probabilities)),
        )
    return result


def summarize_error_matrices(
    matrices: list[np.ndarray],
    *,
    valid_masks: list[np.ndarray] | None = None,
    times: list[np.ndarray] | None = None,
    scale: float | None = None,
    temperature: float | None = None,
) -> dict:
    """Summarize error/retrieval overall and, when supplied, by flow time."""

    if not matrices:
        raise ValueError("at least one error matrix is required")
    for errors in matrices:
        if errors.ndim != 2 or errors.shape[0] != errors.shape[1]:
            raise ValueError("diagnostic matrices must be square")
    masks = valid_masks or [np.ones_like(errors, dtype=bool) for errors in matrices]
    if len(masks) != len(matrices):
        raise ValueError("valid_masks must align with diagnostic matrices")
    for errors, valid in zip(matrices, masks, strict=True):
        if valid.shape != errors.shape or not np.issubdtype(valid.dtype, np.bool_):
            raise ValueError("each valid mask must be boolean and match its matrix")
    selectors = [np.ones(len(errors), dtype=bool) for errors in matrices]
    result = _summarize_rows(
        matrices, masks, selectors, scale=scale, temperature=temperature
    )
    if times is None:
        return result
    if len(times) != len(matrices) or any(
        values.shape != (len(errors),)
        for values, errors in zip(times, matrices, strict=True)
    ):
        raise ValueError("times must align with diagnostic matrix rows")
    result["time_strata"] = {}
    edges = (0.0, 0.25, 0.5, 0.75, 1.0)
    for low, high in zip(edges[:-1], edges[1:], strict=True):
        selected = [
            (values >= low) & (values < high if high < 1.0 else values <= high)
            for values in times
        ]
        if any(values.any() for values in selected):
            key = f"t_{low:.2f}_{high:.2f}"
            result["time_strata"][key] = _summarize_rows(
                matrices,
                masks,
                selected,
                scale=scale,
                temperature=temperature,
            )
    return result
