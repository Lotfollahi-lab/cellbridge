"""H&E cell crops and TERRA/GEX targets for LoRA distillation."""

from __future__ import annotations

import bisect
import hashlib
import json
import re
from collections import OrderedDict
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler

from cellbridge.helpers.cohorts import cohort_in_split, samples_in_split

TARGET_PATCH_SIZE = 224
DST_PIXEL_SIZE_UM = 0.5
TERRA_DIM = 384
VALID_SPLITS = {"distill_train", "distill_val", "label_test", "label_test_fast"}
HVG_METHOD = "raw_overdispersion_union_rank_sum"


def canonical_id(value) -> str:
    """Represent string, byte, and numeric identifiers consistently."""
    if isinstance(value, (bytes, np.bytes_)):
        value = bytes(value).decode("utf-8")
    if isinstance(value, (str, np.str_)):
        text = str(value)
        if re.fullmatch(r"[+-]?(?:0|[1-9]\d*)\.0+", text):
            return str(int(text.split(".", 1)[0]))
        return text
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)) and np.isfinite(value):
        return str(int(value)) if float(value).is_integer() else str(value)
    return str(value)


def _canonical_ids(values) -> np.ndarray:
    return np.asarray([canonical_id(value) for value in values], dtype=str)


def select_indices(n, cap, base_seed, sample_id):
    """Choose at most ``cap`` rows deterministically for one sample."""
    if cap is None or n <= cap:
        return np.arange(n)
    sample_hash = int.from_bytes(
        hashlib.sha1(str(sample_id).encode()).digest()[:8], "big"
    )
    rng = np.random.default_rng([int(base_seed), sample_hash])
    return np.sort(rng.choice(n, cap, replace=False))


def split_samples(
    splits_path,
    split: str,
    label_test_fast_path=None,
) -> list[str]:
    """Load one fixed HEST split, deriving ``label_test_fast`` from label_test."""
    if split not in VALID_SPLITS:
        raise ValueError(f"split must be one of {sorted(VALID_SPLITS)}, got {split!r}")

    if split != "label_test_fast":
        return samples_in_split(
            splits_path,
            split,
            normalize_id=canonical_id,
        )
    if label_test_fast_path is None:
        raise ValueError("label_test_fast_path is required for label_test_fast")
    return cohort_in_split(
        label_test_fast_path,
        splits_path,
        "label_test",
        normalize_id=canonical_id,
    )


def source_patch_size(
    pixel_size_um: float,
    target_size: int = TARGET_PATCH_SIZE,
    target_pixel_size_um: float = DST_PIXEL_SIZE_UM,
) -> int:
    """Level-0 crop side for a requested physical field of view."""
    if not np.isfinite(pixel_size_um) or pixel_size_um <= 0:
        raise ValueError(f"pixel_size_um must be positive, got {pixel_size_um}")
    if target_size < 1 or target_pixel_size_um <= 0:
        raise ValueError("target crop size and pixel size must be positive")
    return max(1, round(target_size * target_pixel_size_um / pixel_size_um))


def crop_cell(
    wsi,
    cx: float,
    cy: float,
    pixel_size_um: float,
    target_size: int = TARGET_PATCH_SIZE,
    target_pixel_size_um: float = DST_PIXEL_SIZE_UM,
) -> Image.Image:
    """Crop, white-pad, and resize one level-0 cell patch for UNI2-h."""
    patch_size = source_patch_size(
        pixel_size_um,
        target_size=target_size,
        target_pixel_size_um=target_pixel_size_um,
    )
    height, width = wsi.shape[:2]
    half = patch_size // 2
    x0, y0 = int(round(cx)) - half, int(round(cy)) - half
    x1, y1 = x0 + patch_size, y0 + patch_size
    vx0, vy0 = max(x0, 0), max(y0, 0)
    vx1, vy1 = min(x1, width), min(y1, height)
    canvas = np.full((patch_size, patch_size, 3), 255, dtype=np.uint8)
    if vx1 > vx0 and vy1 > vy0:
        region = np.asarray(wsi[vy0:vy1, vx0:vx1])
        if region.ndim == 2:
            region = np.repeat(region[..., None], 3, axis=-1)
        canvas[vy0 - y0 : vy1 - y0, vx0 - x0 : vx1 - x0] = region[..., :3]
    return Image.fromarray(canvas).resize(
        (target_size, target_size), Image.Resampling.BILINEAR
    )


def open_wsi(path):
    """Open the level-0 TIFF as a lazy zarr array."""
    import tifffile
    import zarr

    try:
        store = tifffile.imread(path, aszarr=True, level=0)
    except Exception:
        store = tifffile.imread(path, aszarr=True, key=0)
    return zarr.open(store, mode="r")


def _manifest_rows(manifest_path) -> dict[str, dict]:
    table = pd.read_csv(manifest_path)
    required = {"id", "wsi", "pixel_size_um"}
    if not required <= set(table.columns):
        raise ValueError(f"{manifest_path} must contain columns {sorted(required)}")
    rows = {}
    for row in table.to_dict("records"):
        sample_id = canonical_id(row["id"])
        if sample_id in rows:
            raise ValueError(f"{manifest_path} contains duplicate sample {sample_id!r}")
        rows[sample_id] = row
    return rows


def _output_dir(data_root, sample_id: str, manifest_row: dict) -> Path:
    value = manifest_row.get("output_dir")
    if value is not None and not pd.isna(value):
        return Path(value)
    return Path(data_root) / sample_id


def _terra_path(data_root, sample_id: str, manifest_row: dict) -> Path:
    return _output_dir(data_root, sample_id, manifest_row) / f"{sample_id}_terra.npz"


def _adata_path(data_root, sample_id: str, manifest_row: dict) -> Path:
    return _output_dir(data_root, sample_id, manifest_row) / f"{sample_id}.h5ad"


def _load_terra_cells(path, terra_key, cap, seed, sample_id):
    with np.load(path, allow_pickle=True) as data:
        required = {"cell_ids", terra_key, "he_x", "he_y"}
        if not required <= set(data.files):
            raise ValueError(
                f"{path} is missing keys {sorted(required - set(data.files))}"
            )
        cell_ids = _canonical_ids(data["cell_ids"])
        terra = np.asarray(data[terra_key], dtype=np.float32)
        centers = np.column_stack((data["he_x"], data["he_y"])).astype(np.float64)

    if terra.shape != (len(cell_ids), TERRA_DIM):
        raise ValueError(
            f"{path}: expected TERRA shape ({len(cell_ids)}, {TERRA_DIM}), "
            f"got {terra.shape}"
        )
    if centers.shape != (len(cell_ids), 2) or not np.isfinite(centers).all():
        raise ValueError(f"{path}: invalid H&E centers")
    if len(set(cell_ids.tolist())) != len(cell_ids):
        raise ValueError(f"{path}: duplicate cell IDs after canonicalization")

    rows = select_indices(len(cell_ids), cap, seed, sample_id)
    cell_ids, terra, centers = cell_ids[rows], terra[rows], centers[rows]
    norms = np.linalg.norm(terra, axis=1, keepdims=True)
    if not np.isfinite(terra).all() or np.any(norms <= 0):
        raise ValueError(f"{path}: TERRA embeddings must be finite and nonzero")
    return cell_ids, terra / norms, centers


def _aligned_rows(obs_names, cell_ids, path) -> np.ndarray:
    obs_ids = _canonical_ids(obs_names)
    if len(set(obs_ids.tolist())) != len(obs_ids):
        raise ValueError(f"{path}: duplicate expression cell IDs")
    row_for_id = {cell_id: row for row, cell_id in enumerate(obs_ids)}
    missing = [cell_id for cell_id in cell_ids if cell_id not in row_for_id]
    if missing:
        raise KeyError(f"{path}: expression is missing cell IDs {missing[:5]}")
    return np.asarray([row_for_id[cell_id] for cell_id in cell_ids], dtype=int)


def _gex_targets(path, cell_ids, genes: Sequence[str]):
    """Normalize measured genes and mask genes absent from this assay panel."""
    import anndata as ad

    adata = ad.read_h5ad(path, backed="r")
    rows = _aligned_rows(adata.obs_names, cell_ids, path)
    var_names = _canonical_ids(adata.var_names)
    col_for_gene = {gene: col for col, gene in enumerate(var_names)}
    mask = np.asarray([gene in col_for_gene for gene in genes], dtype=bool)
    if not mask.any():
        raise ValueError(f"{path}: none of the selected genes are assayed")
    positions = np.flatnonzero(mask)
    cols = np.asarray([col_for_gene[genes[position]] for position in positions])

    order = np.argsort(rows)
    inverse = np.argsort(order)
    matrix = adata.X[rows[order]][inverse]
    adata.file.close()
    totals = np.asarray(matrix.sum(axis=1)).ravel()
    measured = matrix[:, cols]
    measured = measured.toarray() if sp.issparse(measured) else np.asarray(measured)
    scale = np.divide(
        1e4, totals, out=np.zeros_like(totals, dtype=float), where=totals > 0
    )
    targets = np.zeros((len(rows), len(genes)), dtype=np.float32)
    targets[:, positions] = np.log1p(measured * scale[:, None]).astype(np.float32)
    return targets, mask


class HESTCellDataset(Dataset):
    """Lazy H&E crops paired with normalized TERRA and optional GEX targets."""

    def __init__(
        self,
        data_root,
        manifest_path,
        splits_path,
        split: str,
        *,
        label_test_fast_path=None,
        transform: Callable | None = None,
        max_cells_per_sample: int | None = None,
        seed: int = 0,
        genes: Sequence[str] | None = None,
        terra_key: str = "cell_emb",
        wsi_opener: Callable = open_wsi,
        max_open_wsis: int = 64,
        target_size: int = TARGET_PATCH_SIZE,
        target_pixel_size_um: float = DST_PIXEL_SIZE_UM,
        sample_ids: Sequence[str] | None = None,
    ):
        if max_cells_per_sample is not None and max_cells_per_sample < 1:
            raise ValueError("max_cells_per_sample must be positive or None")
        if max_open_wsis < 1:
            raise ValueError("max_open_wsis must be positive")

        self.split = split
        self.transform = transform
        self.genes = tuple(map(str, genes)) if genes is not None else None
        self.wsi_opener = wsi_opener
        self.max_open_wsis = int(max_open_wsis)
        self.target_size = int(target_size)
        self.target_pixel_size_um = float(target_pixel_size_um)
        source_patch_size(1.0, self.target_size, self.target_pixel_size_um)
        self._wsi_cache = OrderedDict()

        allowed_samples = split_samples(splits_path, split, label_test_fast_path)
        if sample_ids is None:
            samples = allowed_samples
        else:
            samples = [canonical_id(sample) for sample in sample_ids]
            invalid = sorted(set(samples) - set(allowed_samples))
            if invalid:
                raise ValueError(
                    f"requested samples are outside {split}: {invalid[:5]}"
                )
            if not samples or len(samples) != len(set(samples)):
                raise ValueError("sample_ids must be nonempty and unique")
        manifest = _manifest_rows(manifest_path)
        missing = [sample for sample in samples if sample not in manifest]
        if missing:
            raise KeyError(f"manifest is missing split samples {missing[:5]}")

        self.sample_ids = samples
        self._samples = []
        self._ends = []
        self.sample_indices = {}
        start = 0
        for sample_id in samples:
            row = manifest[sample_id]
            cell_ids, terra, centers = _load_terra_cells(
                _terra_path(data_root, sample_id, row),
                terra_key,
                max_cells_per_sample,
                seed,
                sample_id,
            )
            gex_result = (
                _gex_targets(
                    _adata_path(data_root, sample_id, row), cell_ids, self.genes
                )
                if self.genes is not None
                else None
            )
            gex, gex_mask = gex_result if gex_result is not None else (None, None)
            stop = start + len(cell_ids)
            self.sample_indices[sample_id] = np.arange(start, stop, dtype=int)
            self._samples.append(
                {
                    "sample_id": sample_id,
                    "wsi": str(row["wsi"]),
                    "pixel_size_um": float(row["pixel_size_um"]),
                    "cell_ids": cell_ids,
                    "terra": terra,
                    "centers": centers,
                    "gex": gex,
                    "gex_mask": gex_mask,
                }
            )
            self._ends.append(stop)
            start = stop
        self._length = start

    def __len__(self):
        return self._length

    @property
    def sample_cell_ids(self) -> dict[str, np.ndarray]:
        """Expose capped IDs in dataset order for metadata alignment."""

        return {sample["sample_id"]: sample["cell_ids"] for sample in self._samples}

    @property
    def sample_centers(self) -> dict[str, np.ndarray]:
        """Expose H&E centers in dataset order for spatial splits."""

        return {sample["sample_id"]: sample["centers"] for sample in self._samples}

    @property
    def sample_terra(self) -> dict[str, np.ndarray]:
        """Expose TERRA targets in dataset order; writable for target shrinkage."""

        return {sample["sample_id"]: sample["terra"] for sample in self._samples}

    def _get_wsi(self, sample):
        sample_id = sample["sample_id"]
        if sample_id in self._wsi_cache:
            return self._wsi_cache.pop(sample_id)
        wsi = self.wsi_opener(sample["wsi"])
        if len(self._wsi_cache) == self.max_open_wsis:
            _, old = self._wsi_cache.popitem(last=False)
            _close_wsi(old)
        return wsi

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        sample_number = bisect.bisect_right(self._ends, index)
        start = 0 if sample_number == 0 else self._ends[sample_number - 1]
        row = index - start
        sample = self._samples[sample_number]
        wsi = self._get_wsi(sample)
        self._wsi_cache[sample["sample_id"]] = wsi
        cx, cy = sample["centers"][row]
        image = crop_cell(
            wsi,
            cx,
            cy,
            sample["pixel_size_um"],
            self.target_size,
            self.target_pixel_size_um,
        )
        if self.transform is not None:
            image = self.transform(image)

        item = {
            "image": image,
            "terra": torch.from_numpy(sample["terra"][row]),
            "sample_id": sample["sample_id"],
            "cell_id": sample["cell_ids"][row],
        }
        if sample["gex"] is not None:
            item["gex"] = torch.from_numpy(sample["gex"][row])
            item["gex_mask"] = torch.from_numpy(sample["gex_mask"])
        return item

    def close(self):
        for wsi in self._wsi_cache.values():
            _close_wsi(wsi)
        self._wsi_cache.clear()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_wsi_cache"] = OrderedDict()
        return state

    def __del__(self):
        self.close()


def _close_wsi(wsi):
    store = getattr(wsi, "store", None)
    close = getattr(store, "close", None)
    if close is not None:
        close()


class WithinSampleBatchSampler(Sampler[list[int]]):
    """Yield full single-sample batches in locality-preserving visits."""

    def __init__(
        self,
        dataset: HESTCellDataset,
        batch_size: int,
        seed=0,
        shuffle=True,
        batches_per_sample_visit=32,
    ):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if batches_per_sample_visit < 1:
            raise ValueError("batches_per_sample_visit must be positive")
        if not dataset.sample_indices:
            raise ValueError("dataset has no samples")

        self.sample_indices = tuple(dataset.sample_indices.values())
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.batches_per_sample_visit = int(batches_per_sample_visit)
        self.epoch = 0
        too_small = [
            len(indices)
            for indices in self.sample_indices
            if len(indices) < batch_size
        ]
        if too_small:
            raise ValueError(
                "every sample must contain at least batch_size cells; "
                f"smallest sample has {min(too_small)}, batch_size={batch_size}"
            )
        if any(
            len(np.unique(indices)) != len(indices)
            for indices in self.sample_indices
        ):
            raise ValueError("sample index pools must not contain duplicates")
        self.n_batches = sum(
            len(indices) // self.batch_size for indices in self.sample_indices
        )

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        rng = np.random.default_rng([self.seed, self.epoch])
        pools = [
            rng.permutation(indices) if self.shuffle else indices
            for indices in self.sample_indices
        ]
        visits = []
        for sample_number, indices in enumerate(pools):
            starts = list(
                range(0, len(indices) - self.batch_size + 1, self.batch_size)
            )
            visit_size = self.batches_per_sample_visit
            visits.extend(
                [
                    (sample_number, start)
                    for start in starts[offset : offset + visit_size]
                ]
                for offset in range(0, len(starts), visit_size)
            )
        if self.shuffle:
            visits = [visits[index] for index in rng.permutation(len(visits))]

        for visit in visits:
            for sample_number, start in visit:
                batch = pools[sample_number][start : start + self.batch_size]
                if len(np.unique(batch)) != self.batch_size:
                    raise RuntimeError("contrastive batches must contain unique cells")
                yield batch.tolist()


def fit_train_hvgs(
    data_root,
    manifest_path,
    splits_path,
    output_path,
    n_genes: int,
    *,
    max_cells_total: int,
    seed: int = 0,
) -> list[str]:
    """Fit and persist HVGs using capped ``distill_train`` raw counts only."""
    import anndata as ad

    if n_genes < 1:
        raise ValueError("n_genes must be positive")
    if max_cells_total < 1:
        raise ValueError("max_cells_total must be positive")
    train_samples = split_samples(splits_path, "distill_train")
    manifest = _manifest_rows(manifest_path)
    sources = _gene_sources(data_root, manifest, train_samples)
    caps = _balanced_caps(train_samples, max_cells_total)
    rank_sums = {}
    coverage = {}
    sample_gene_counts = {}
    sample_cell_counts = {}
    for sample_id in train_samples:
        if sample_id not in manifest:
            raise KeyError(f"manifest is missing train sample {sample_id!r}")
        row = manifest[sample_id]
        terra_path = _terra_path(data_root, sample_id, row)
        with np.load(terra_path, allow_pickle=True) as data:
            cell_ids = _canonical_ids(data["cell_ids"])
        chosen = select_indices(len(cell_ids), caps[sample_id], seed, sample_id)
        cell_ids = cell_ids[chosen]
        adata_path = _adata_path(data_root, sample_id, row)
        sample = ad.read_h5ad(adata_path, backed="r")
        rows = _aligned_rows(sample.obs_names, cell_ids, adata_path)
        sample_memory = sample[np.sort(rows)].to_memory()
        sample.file.close()
        sample = sample_memory
        sample_gene_counts[sample_id] = int(sample.n_vars)
        sample_cell_counts[sample_id] = int(sample.n_obs)
        overdispersion = pd.Series(
            _raw_overdispersion(sample.X),
            index=sample.var_names.astype(str),
        )
        percentile = overdispersion.rank(method="average", pct=True)
        for gene, score in percentile.items():
            gene = str(gene)
            coverage[gene] = coverage.get(gene, 0) + 1
            if np.isfinite(score):
                rank_sums[gene] = rank_sums.get(gene, 0.0) + float(score)

    if n_genes > len(coverage):
        raise ValueError(
            f"requested {n_genes} genes, but train samples contain only "
            f"{len(coverage)} unique genes"
        )
    genes = sorted(
        coverage,
        key=lambda gene: (-rank_sums.get(gene, 0.0), -coverage[gene], gene),
    )[:n_genes]
    if len(genes) != n_genes:
        raise RuntimeError(
            f"union HVG aggregation selected {len(genes)} genes; expected {n_genes}"
        )

    selected_coverage = {gene: coverage[gene] for gene in genes}
    selected_rank_sums = {gene: round(rank_sums.get(gene, 0.0), 8) for gene in genes}
    payload = {
        "genes": genes,
        "method": HVG_METHOD,
        "aggregation": "sum of within-sample raw-overdispersion percentiles",
        "n_genes": int(n_genes),
        "split": "distill_train",
        "sample_ids": train_samples,
        "sample_cell_counts": sample_cell_counts,
        "sample_gene_counts": sample_gene_counts,
        "gene_sample_coverage": selected_coverage,
        "gene_rank_sums": selected_rank_sums,
        "max_cells_total": int(max_cells_total),
        "seed": int(seed),
        "sources": sources,
        "signature": _gene_signature(
            train_samples, n_genes, max_cells_total, seed, sources
        ),
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(output_path)
    return genes


def _raw_overdispersion(matrix):
    """Method-of-moments excess-Poisson dispersion for each raw-count gene."""
    n_cells, n_genes = matrix.shape
    if n_cells < 2:
        raise ValueError("at least two cells per sample are required for HVG fitting")
    if sp.issparse(matrix):
        if matrix.data.size and (
            not np.isfinite(matrix.data).all() or np.any(matrix.data < 0)
        ):
            raise ValueError("HVG input must contain finite, nonnegative raw counts")
        sums = np.asarray(matrix.sum(axis=0)).ravel().astype(np.float64)
        squared_sums = np.asarray(matrix.power(2).sum(axis=0)).ravel()
    else:
        matrix = np.asarray(matrix)
        if not np.isfinite(matrix).all() or np.any(matrix < 0):
            raise ValueError("HVG input must contain finite, nonnegative raw counts")
        sums = matrix.sum(axis=0, dtype=np.float64)
        squared_sums = np.square(matrix, dtype=np.float64).sum(axis=0)
    if sums.shape != (n_genes,):
        raise RuntimeError("unexpected gene-statistic shape")
    mean = sums / n_cells
    variance = (squared_sums - n_cells * mean**2) / (n_cells - 1)
    score = np.full(n_genes, np.nan, dtype=np.float64)
    measured = mean > 0
    score[measured] = (variance[measured] - mean[measured]) / mean[measured] ** 2
    return score


def _balanced_caps(samples, total):
    base, remainder = divmod(int(total), len(samples))
    if base == 0:
        raise ValueError(
            f"max_cells_total={total} is smaller than {len(samples)} train samples"
        )
    return {
        sample: base + (position < remainder) for position, sample in enumerate(samples)
    }


def _file_identity(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {
        "path": str(path),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _gene_sources(data_root, manifest, samples):
    return [
        {
            "sample_id": sample,
            "terra": _file_identity(_terra_path(data_root, sample, manifest[sample])),
            "h5ad": _file_identity(_adata_path(data_root, sample, manifest[sample])),
        }
        for sample in samples
    ]


def _gene_signature(samples, n_genes, cap, seed, sources):
    payload = json.dumps(
        {
            "method": HVG_METHOD,
            "samples": samples,
            "n_genes": n_genes,
            "cap": cap,
            "seed": seed,
            "sources": sources,
        },
        sort_keys=True,
    ).encode()
    return hashlib.sha1(payload).hexdigest()


def load_selected_genes(path) -> list[str]:
    payload = json.loads(Path(path).read_text())
    genes = payload.get("genes")
    if (
        not isinstance(genes, list)
        or not genes
        or not all(isinstance(g, str) for g in genes)
        or len(genes) != len(set(genes))
        or payload.get("n_genes") != len(genes)
        or payload.get("split") != "distill_train"
        or payload.get("method") != HVG_METHOD
    ):
        raise ValueError(f"{path}: invalid selected-gene artifact")
    return genes


def dataset_kwargs(cfg, transform, genes=None):
    """Translate shared LoRA configuration into HEST dataset arguments."""

    data_root = cfg["data_root"]
    crop = cfg.get("crop", {})
    settings = {
        "data_root": data_root,
        "manifest_path": cfg.get("manifest_path", str(Path(data_root) / "manifest.csv")),
        "splits_path": cfg["splits_path"],
        "transform": transform,
        "max_cells_per_sample": cfg.get("max_cells_per_sample"),
        "seed": int(cfg.get("sampling_seed", 0)),
        "terra_key": cfg.get("terra_key", "cell_emb"),
        "target_size": int(crop.get("target_size", TARGET_PATCH_SIZE)),
        "target_pixel_size_um": float(crop.get("target_pixel_size_um", DST_PIXEL_SIZE_UM)),
        "max_open_wsis": int(cfg.get("max_open_wsis", 64)),
    }
    if genes is not None:
        settings["genes"] = genes
    return settings


def build_datasets(cfg, transform, selected_genes_path=None):
    """Build train/validation datasets and fit train-only genes when requested."""
    data_root = cfg["data_root"]
    manifest_path = cfg.get("manifest_path", str(Path(data_root) / "manifest.csv"))
    splits_path = cfg["splits_path"]
    seed = int(cfg.get("sampling_seed", 0))
    common = dataset_kwargs(cfg, transform)

    mode = cfg.get("objective", {}).get("mode", "contrastive")
    if mode not in {"contrastive", "contrastive_gex"}:
        raise ValueError(f"unknown objective.mode {mode!r}")
    genes = None
    if mode == "contrastive_gex":
        gex = cfg.get("gex", {})
        n_genes = int(gex.get("num_selected_genes", 100))
        max_selection = int(gex.get("max_cells_for_selection", 200_000))
        if selected_genes_path is None:
            selected_genes_path = Path(cfg["output_dir"]) / "selected_genes.json"
        genes = _load_or_fit_genes(
            data_root,
            manifest_path,
            splits_path,
            selected_genes_path,
            n_genes,
            max_selection,
            seed,
        )
        common["genes"] = genes

    train = HESTCellDataset(split="distill_train", **common)
    validation = HESTCellDataset(split="distill_val", **common)
    return train, validation, genes


def _load_or_fit_genes(
    data_root,
    manifest_path,
    splits_path,
    path,
    n_genes,
    max_cells_total,
    seed,
):
    path = Path(path)
    train_samples = split_samples(splits_path, "distill_train")
    manifest = _manifest_rows(manifest_path)
    sources = _gene_sources(data_root, manifest, train_samples)
    signature = _gene_signature(train_samples, n_genes, max_cells_total, seed, sources)
    if path.exists():
        payload = json.loads(path.read_text())
        if payload.get("signature") == signature:
            return load_selected_genes(path)
    return fit_train_hvgs(
        data_root,
        manifest_path,
        splits_path,
        path,
        n_genes,
        max_cells_total=max_cells_total,
        seed=seed,
    )


def build_label_test_dataset(
    cfg,
    transform,
    max_cells_per_sample=None,
    sample_ids=None,
):
    """Build the five-sample held-out HEST dataset without expression targets."""
    data_root = cfg["data_root"]
    crop = cfg.get("crop", {})
    evaluation = cfg.get("eval", {})
    return HESTCellDataset(
        data_root=data_root,
        manifest_path=cfg.get("manifest_path", str(Path(data_root) / "manifest.csv")),
        splits_path=cfg["splits_path"],
        split="label_test_fast",
        label_test_fast_path=evaluation["label_test_fast_path"],
        transform=transform,
        max_cells_per_sample=max_cells_per_sample,
        seed=int(cfg.get("sampling_seed", 0)),
        terra_key=cfg.get("terra_key", "cell_emb"),
        target_size=int(crop.get("target_size", TARGET_PATCH_SIZE)),
        target_pixel_size_um=float(crop.get("target_pixel_size_um", DST_PIXEL_SIZE_UM)),
        max_open_wsis=int(cfg.get("max_open_wsis", 64)),
        sample_ids=sample_ids,
    )
