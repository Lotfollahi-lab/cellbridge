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
            raise ValueError("annotated sampling strategy must be balanced or hierarchy")
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
                by_node.setdefault(int(self.levels[index, level]), set()).add(int(index))
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
                    sample for sample, pool in pools.items() if positions[sample] < len(pool)
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
        compartment = self._pools["compartment"].get(
            int(self.levels[anchor, 1]), set()
        )
        if relation == "identity":
            options = identity if self.depth[anchor] >= 3 else set()
        elif relation == "lineage":
            options = lineage - identity
        elif relation == "compartment":
            options = compartment - lineage
        else:
            options = self._pools["usable"] - compartment
        options = options & available
        cross = {index for index in options if self.samples[index] != self.samples[anchor]}
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
        iterator = self._balanced(rng) if self.strategy == "balanced" else self._hierarchy(rng)
        yield from iterator


class InterleavedBatchSampler(Sampler[list[int]]):
    """Interleave base and annotated batches without changing either sampler."""

    def __init__(self, base, annotated, annotated_offset: int, seed: int = 0):
        self.base = base
        self.annotated = annotated
        self.offset = int(annotated_offset)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)
        for sampler in (self.base, self.annotated):
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)

    def __len__(self):
        return len(self.base) + len(self.annotated)

    def __iter__(self):
        rng = np.random.default_rng([self.seed, self.epoch])
        schedule = np.concatenate(
            (np.zeros(len(self.base), dtype=np.int8), np.ones(len(self.annotated), dtype=np.int8))
        )
        schedule = rng.permutation(schedule)
        iterators = iter(self.base), iter(self.annotated)
        for source in schedule:
            batch = next(iterators[int(source)])
            if source:
                batch = [self.offset + index for index in batch]
            yield batch
