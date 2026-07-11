"""Embedding loading, alignment, sampling, and caching."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

UNI_SUFFIX = "_uni2.npz"
TERRA_SUFFIX = "_terra.npz"
LABEL_SUFFIX = "_granular_cell_labels.csv"


@dataclass(frozen=True)
class SplitCache:
    uni: np.ndarray
    terra: np.ndarray
    sample_ids: np.ndarray
    cell_ids: np.ndarray

    def __len__(self):
        return len(self.uni)


def _sample_path(data_root, sample_id, suffix):
    return Path(data_root) / sample_id / f"{sample_id}{suffix}"


def all_samples(data_root):
    """Samples with both UNI and TERRA embeddings."""
    root = Path(data_root)
    return sorted(
        p.name
        for p in root.iterdir()
        if p.is_dir()
        and _sample_path(root, p.name, UNI_SUFFIX).exists()
        and _sample_path(root, p.name, TERRA_SUFFIX).exists()
    )


def annotated_samples(data_root):
    """Samples with granular cell labels."""
    root = Path(data_root)
    return sorted(
        p.name
        for p in root.iterdir()
        if p.is_dir() and _sample_path(root, p.name, LABEL_SUFFIX).exists()
    )


def _load_npz(path, key):
    with np.load(path, allow_pickle=False) as data:
        values = np.asarray(data[key])
        cell_ids = np.asarray(data["cell_ids"]).astype(str)
    if len(values) != len(cell_ids):
        raise ValueError(f"{path}: {len(values)} rows but {len(cell_ids)} cell IDs")
    return values, cell_ids


def load_sample_embeddings(
    data_root, sample_id, uni_key="uni_emb", terra_key="cell_emb"
):
    """Load one sample with TERRA rows aligned to UNI."""
    uni, uni_ids = _load_npz(_sample_path(data_root, sample_id, UNI_SUFFIX), uni_key)
    terra, terra_ids = _load_npz(
        _sample_path(data_root, sample_id, TERRA_SUFFIX), terra_key
    )

    if not np.array_equal(uni_ids, terra_ids):
        terra_row = {cell_id: i for i, cell_id in enumerate(terra_ids)}
        keep = np.fromiter(
            (cell_id in terra_row for cell_id in uni_ids),
            dtype=bool,
            count=len(uni_ids),
        )
        if not keep.any():
            raise ValueError(f"{sample_id}: UNI and TERRA have no shared cells")
        rows = np.fromiter(
            (terra_row[cell_id] for cell_id in uni_ids[keep]),
            dtype=int,
            count=int(keep.sum()),
        )
        uni, uni_ids, terra = uni[keep], uni_ids[keep], terra[rows]

    return (
        uni.astype(np.float32, copy=False),
        terra.astype(np.float32, copy=False),
        uni_ids,
    )


def _sample_rng(seed, sample_id):
    sample_hash = int.from_bytes(hashlib.sha1(sample_id.encode()).digest()[:8], "big")
    return np.random.default_rng([int(seed), sample_hash])


def select_indices(n, cap, base_seed, sample_id):
    """Choose at most cap rows deterministically."""
    if cap is None or n <= cap:
        return np.arange(n)
    return np.sort(_sample_rng(base_seed, sample_id).choice(n, cap, replace=False))


def _cache_signature(data_root, sample_ids, cap, seed, uni_key, terra_key):
    root = Path(data_root).resolve()
    files = []
    for sample_id in sample_ids:
        for suffix in (UNI_SUFFIX, TERRA_SUFFIX):
            path = _sample_path(root, sample_id, suffix)
            stat = path.stat()
            files.append((str(path), stat.st_size, stat.st_mtime_ns))
    payload = {
        "samples": list(sample_ids),
        "cap": cap,
        "seed": seed,
        "uni_key": uni_key,
        "terra_key": terra_key,
        "files": files,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha1(encoded).hexdigest()


def build_split_cache(
    data_root,
    sample_ids,
    max_cells_per_sample,
    sampling_seed,
    uni_key="uni_emb",
    terra_key="cell_emb",
    cache_path=None,
):
    """Load, cap, and shuffle a sample split."""
    sample_ids = [str(sample_id) for sample_id in sample_ids]
    if not sample_ids:
        raise ValueError("cannot build a cache from an empty sample split")

    signature = _cache_signature(
        data_root,
        sample_ids,
        max_cells_per_sample,
        sampling_seed,
        uni_key,
        terra_key,
    )
    if cache_path is not None and Path(cache_path).exists():
        cached = load_cache(cache_path, signature)
        if cached is not None:
            return cached

    uni_parts, terra_parts, sample_parts, cell_parts = [], [], [], []
    for sample_id in sample_ids:
        uni, terra, cell_ids = load_sample_embeddings(
            data_root, sample_id, uni_key, terra_key
        )
        rows = select_indices(
            len(cell_ids), max_cells_per_sample, sampling_seed, sample_id
        )
        uni_parts.append(uni[rows].astype(np.float16))
        terra_parts.append(terra[rows].astype(np.float16))
        sample_parts.append(np.full(len(rows), sample_id))
        cell_parts.append(cell_ids[rows])

    cache = SplitCache(
        uni=np.concatenate(uni_parts),
        terra=np.concatenate(terra_parts),
        sample_ids=np.concatenate(sample_parts),
        cell_ids=np.concatenate(cell_parts),
    )
    order = np.random.default_rng(sampling_seed).permutation(len(cache))
    cache = SplitCache(
        cache.uni[order],
        cache.terra[order],
        cache.sample_ids[order],
        cache.cell_ids[order],
    )
    if cache_path is not None:
        save_cache(cache_path, cache, signature)
    return cache


def save_cache(path, cache, signature):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp.npz")
    np.savez(
        tmp_path,
        uni=cache.uni,
        terra=cache.terra,
        sample_ids=cache.sample_ids,
        cell_ids=cache.cell_ids,
        signature=np.asarray(signature),
    )
    tmp_path.replace(path)


def load_cache(path, expected_signature):
    with np.load(path, allow_pickle=False) as data:
        if "signature" not in data.files:
            return None
        if str(data["signature"].item()) != expected_signature:
            return None
        return SplitCache(
            data["uni"],
            data["terra"],
            data["sample_ids"],
            data["cell_ids"],
        )


class CellEmbeddingDataset(Dataset):
    def __init__(self, uni, terra):
        if len(uni) != len(terra):
            raise ValueError(f"UNI has {len(uni)} rows; TERRA has {len(terra)}")
        self.uni = uni
        self.terra = terra

    def __len__(self):
        return len(self.uni)

    def __getitem__(self, index):
        uni = torch.from_numpy(np.asarray(self.uni[index], dtype=np.float32))
        terra = torch.from_numpy(np.asarray(self.terra[index], dtype=np.float32))
        return uni, terra
