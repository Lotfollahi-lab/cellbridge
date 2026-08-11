"""Windowed training metrics and multi-criterion checkpoint selection."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

IMPROVEMENT_MARGIN = 1e-6


class MetricWindow:
    """Average step metrics over a logging window."""

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        self._sums: dict[str, float] = {}
        self._steps = 0

    def add(self, metrics: Mapping[str, float]) -> None:
        self._steps += 1
        for name, value in metrics.items():
            self._sums[name] = self._sums.get(name, 0.0) + float(value)

    @property
    def steps(self) -> int:
        return self._steps

    def mean(self) -> dict[str, float]:
        if self._steps == 0:
            raise ValueError("metric window is empty")
        averaged = {name: total / self._steps for name, total in self._sums.items()}
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
