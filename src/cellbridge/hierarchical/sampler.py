"""Deterministic cross-sample batches for annotated hierarchy supervision."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from torch.utils.data import Sampler

from .losses import RELATION_ORDER


class AnnotatedBatchSampler(Sampler[list[int]]):
    """Use every full-batch cell once, with optional paper-style relation draws."""

    def __init__(
        self,
        dataset,
        batch_size: int,
        *,
        seed: int = 0,
        shuffle: bool = True,
        strategy: str = "balanced",
        relations: Sequence[str] = ("identity", "lineage", "unrelated"),
        cross_sample: str = "prefer",
    ):
        if batch_size < 2:
            raise ValueError("annotated batch_size must be at least 2")
        if len(dataset.sample_indices) < 2:
            raise ValueError("annotated batches require at least two samples")
        if strategy not in {"balanced", "hierarchy"}:
            raise ValueError(
                "annotated sampling strategy must be balanced or hierarchy"
            )
        if cross_sample not in {"any", "prefer", "only"}:
            raise ValueError("sampling cross_sample must be any, prefer, or only")
        if any(name not in RELATION_ORDER[1:] for name in relations):
            raise ValueError("annotated sampler received an unknown relation")
        self.dataset = dataset
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.strategy = strategy
        self.relations = tuple(relations)
        self.cross_sample = cross_sample
        self.epoch = 0
        self.n_batches = len(dataset) // self.batch_size
        self.samples = np.asarray(dataset.sample_ids_by_index, dtype=str)
        self.levels = dataset.levels.numpy()
        self.depth = dataset.depth.numpy()
        self.usable = dataset.usable.numpy()
        self._pools = self._node_pools()

    def _node_pools(self):
        pools = {}
        for name, level, minimum in (
            ("identity", 3, 3),
            ("lineage", 2, 2),
            ("compartment", 1, 1),
        ):
            by_node = {}
            for index in np.flatnonzero(self.usable & (self.depth >= minimum)):
                by_node.setdefault(int(self.levels[index, level]), set()).add(
                    int(index)
                )
            pools[name] = by_node
        pools["usable"] = set(map(int, np.flatnonzero(self.usable)))
        return pools

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __len__(self):
        return self.n_batches

    def _balanced(self, rng):
        pools = {
            sample: (
                rng.permutation(indices).tolist() if self.shuffle else indices.tolist()
            )
            for sample, indices in self.dataset.sample_indices.items()
        }
        positions = {sample: 0 for sample in pools}
        while (
            sum(len(pool) - positions[sample] for sample, pool in pools.items())
            >= self.batch_size
        ):
            batch = []
            while len(batch) < self.batch_size:
                active = [
                    sample
                    for sample, pool in pools.items()
                    if positions[sample] < len(pool)
                ]
                order = rng.permutation(active) if self.shuffle else active
                for sample in order:
                    batch.append(pools[sample][positions[sample]])
                    positions[sample] += 1
                    if len(batch) == self.batch_size:
                        break
            yield batch

    def _options(self, anchor, relation, available):
        if not self.usable[anchor]:
            return set()
        identity = self._pools["identity"].get(int(self.levels[anchor, 3]), set())
        lineage = self._pools["lineage"].get(int(self.levels[anchor, 2]), set())
        compartment = self._pools["compartment"].get(int(self.levels[anchor, 1]), set())
        if relation == "identity":
            options = identity if self.depth[anchor] >= 3 else set()
        elif relation == "lineage":
            options = lineage - identity
        elif relation == "compartment":
            options = compartment - lineage
        else:
            options = self._pools["usable"] - compartment
        options = options & available
        cross = {
            index for index in options if self.samples[index] != self.samples[anchor]
        }
        if self.cross_sample == "only":
            return cross
        if self.cross_sample == "prefer" and cross:
            return cross
        return options

    def _hierarchy(self, rng):
        available = set(range(len(self.dataset)))
        order = iter(rng.permutation(len(self.dataset)).tolist())
        while len(available) >= self.batch_size:
            batch = []
            while len(batch) < self.batch_size:
                anchor = next((index for index in order if index in available), None)
                if anchor is None:
                    break
                available.remove(anchor)
                batch.append(anchor)
                if len(batch) == self.batch_size:
                    break
                for relation in self.relations:
                    options = self._options(anchor, relation, available)
                    if options:
                        candidate = int(rng.choice(sorted(options)))
                        available.remove(candidate)
                        batch.append(candidate)
                    if len(batch) == self.batch_size:
                        break
            if len(batch) != self.batch_size:
                return
            yield batch

    def __iter__(self):
        rng = np.random.default_rng([self.seed, self.epoch])
        iterator = (
            self._balanced(rng) if self.strategy == "balanced" else self._hierarchy(rng)
        )
        yield from iterator


class InterleavedBatchSampler(Sampler[list[int]]):
    """Interleave base and annotated batches with an optional fixed source share."""

    def __init__(
        self,
        base,
        annotated,
        annotated_offset: int,
        seed: int = 0,
        annotated_fraction: float | None = None,
    ):
        self.base = base
        self.annotated = annotated
        self.offset = int(annotated_offset)
        self.seed = int(seed)
        if annotated_fraction is not None and not 0 < float(annotated_fraction) < 1:
            raise ValueError("annotated_fraction must be strictly between 0 and 1")
        self.annotated_fraction = (
            None if annotated_fraction is None else float(annotated_fraction)
        )
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)
        for sampler in (self.base, self.annotated):
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)

    def __len__(self):
        return len(self.base) + len(self.annotated)

    @property
    def source_counts(self) -> tuple[int, int]:
        total = len(self)
        if self.annotated_fraction is None:
            return len(self.base), len(self.annotated)
        annotated = round(total * self.annotated_fraction)
        annotated = min(total - 1, max(1, annotated))
        return total - annotated, annotated

    def _fixed_fraction_schedule(self):
        """Spread the requested annotated count as evenly as integer steps allow."""

        base_count, annotated_count = self.source_counts
        total = base_count + annotated_count
        steps = np.arange(total, dtype=np.int64)
        return (
            ((steps + 1) * annotated_count) // total
            - (steps * annotated_count) // total
        ).astype(np.int8)

    def __iter__(self):
        rng = np.random.default_rng([self.seed, self.epoch])
        if self.annotated_fraction is None:
            schedule = np.concatenate(
                (
                    np.zeros(len(self.base), dtype=np.int8),
                    np.ones(len(self.annotated), dtype=np.int8),
                )
            )
            schedule = rng.permutation(schedule)
        else:
            schedule = self._fixed_fraction_schedule()

        samplers = self.base, self.annotated
        epoch_base = self.epoch
        if self.annotated_fraction is not None:
            epoch_base = self.epoch * (len(self) + 1)
            for sampler in samplers:
                if hasattr(sampler, "set_epoch"):
                    sampler.set_epoch(epoch_base)
        iterators = [iter(sampler) for sampler in samplers]
        cycles = [0, 0]
        for source in schedule:
            source = int(source)
            try:
                batch = next(iterators[source])
            except StopIteration:
                cycles[source] += 1
                sampler = samplers[source]
                if hasattr(sampler, "set_epoch"):
                    sampler.set_epoch(epoch_base + cycles[source])
                iterators[source] = iter(sampler)
                try:
                    batch = next(iterators[source])
                except StopIteration as error:
                    raise RuntimeError(
                        "an interleaved source sampler yielded fewer batches than "
                        "its reported nonzero length"
                    ) from error
            if source:
                batch = [self.offset + index for index in batch]
            yield batch
