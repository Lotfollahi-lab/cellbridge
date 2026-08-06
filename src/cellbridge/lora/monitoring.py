"""Windowed training metrics and multi-criterion checkpoint selection."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

IMPROVEMENT_MARGIN = 1e-6


class MetricWindow:
    """Average step metrics over a logging window instead of sampling one step.

    Logging a single step aliases badly against a deterministic batch schedule:
    with an alternating labelled/unlabelled interleave and an even ``log_every``,
    every logged row comes from the same source. Averaging the whole window
    removes that.

    Metrics that only apply on gated steps are averaged over those steps alone,
    so a rare labelled batch is not diluted by the zeros around it. The gate is
    reported as the rate of gated steps in the window.
    """

    def __init__(self, gate_key: str | None = None, gated_prefixes: Iterable[str] = ()):
        self.gate_key = gate_key
        self.gated_prefixes = tuple(gated_prefixes)
        self.reset()

    def reset(self) -> None:
        self._sums: dict[str, float] = {}
        self._steps = 0
        self._gated_steps = 0

    def _is_gated(self, name: str) -> bool:
        return name != self.gate_key and name.startswith(self.gated_prefixes)

    def add(self, metrics: Mapping[str, float]) -> None:
        gate_open = self.gate_key is None or bool(float(metrics.get(self.gate_key, 0)))
        self._steps += 1
        self._gated_steps += int(gate_open)
        for name, value in metrics.items():
            # Register every key even when the gate is shut, so a window with no
            # labelled step reports NaN rather than dropping the column.
            total = self._sums.setdefault(name, 0.0)
            if self._is_gated(name) and not gate_open:
                continue
            self._sums[name] = total + float(value)

    @property
    def steps(self) -> int:
        return self._steps

    def mean(self) -> dict[str, float]:
        if self._steps == 0:
            raise ValueError("metric window is empty")
        averaged = {}
        for name, total in self._sums.items():
            count = self._gated_steps if self._is_gated(name) else self._steps
            averaged[name] = total / count if count else float("nan")
        if self.gate_key is not None:
            averaged[self.gate_key] = self._gated_steps / self._steps
        averaged["window_steps"] = float(self._steps)
        return averaged


@dataclass(frozen=True)
class Criterion:
    """One validation metric with its optimisation direction and checkpoint."""

    name: str
    mode: str
    path: Path

    def __post_init__(self):
        if self.mode not in {"min", "max"}:
            raise ValueError(f"criterion mode must be 'min' or 'max', got {self.mode!r}")

    @property
    def initial(self) -> float:
        return -math.inf if self.mode == "max" else math.inf

    def improves(self, value: float, best: float) -> bool:
        if self.mode == "max":
            return value > best + IMPROVEMENT_MARGIN
        return value < best - IMPROVEMENT_MARGIN


class CheckpointSelector:
    """Keep a separate best checkpoint per criterion, from one training run.

    Selecting on the paired contrastive loss and on a labelled validation probe
    answers different questions, and the two disagree. Tracking both keeps new
    runs comparable with runs selected the old way. Training stops only once
    every criterion has gone stale, so neither checkpoint is truncated early.
    """

    def __init__(self, criteria: Iterable[Criterion], patience: int):
        self.criteria = tuple(criteria)
        if not self.criteria:
            raise ValueError("at least one selection criterion is required")
        if len({criterion.name for criterion in self.criteria}) != len(self.criteria):
            raise ValueError("selection criteria must have unique metric names")
        if int(patience) < 1:
            raise ValueError("patience must be positive")
        self.patience = int(patience)
        self._best = {criterion.name: criterion.initial for criterion in self.criteria}
        self._best_step = {criterion.name: 0 for criterion in self.criteria}
        self._stale = {criterion.name: 0 for criterion in self.criteria}

    @property
    def primary(self) -> Criterion:
        return self.criteria[0]

    def update(self, val_metrics: Mapping[str, float], step: int) -> list[Criterion]:
        """Record one validation round and return the criteria that improved."""

        improved = []
        for criterion in self.criteria:
            if criterion.name not in val_metrics:
                raise KeyError(f"validation metric {criterion.name!r} is unavailable")
            value = float(val_metrics[criterion.name])
            if math.isfinite(value) and criterion.improves(value, self._best[criterion.name]):
                self._best[criterion.name] = value
                self._best_step[criterion.name] = step
                self._stale[criterion.name] = 0
                improved.append(criterion)
            else:
                self._stale[criterion.name] += 1
        return improved

    @property
    def should_stop(self) -> bool:
        return all(stale >= self.patience for stale in self._stale.values())

    def summary(self) -> dict[str, dict]:
        return {
            criterion.name: {
                "mode": criterion.mode,
                "best_value": self._best[criterion.name],
                "best_step": self._best_step[criterion.name],
                "checkpoint": str(criterion.path),
            }
            for criterion in self.criteria
        }
