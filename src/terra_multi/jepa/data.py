"""Spatial windows from aligned UNI and TERRA section embeddings.

Each window contains image embeddings from context cells and TERRA targets from
a nearby, masked block of cells in the same tissue section. Spatial parameters
are expressed in microns; H&E pixel coordinates are converted with the
``pixel_size_um`` recorded in the HEST manifest.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.spatial import cKDTree
from torch.utils.data import Dataset

from terra_multi.cohorts import samples_in_split
from terra_multi.distill.data import select_indices


def _validate_context_arrays(sample_id, cell_ids, image, coordinates_um):
    n_cells = len(cell_ids)
    if not sample_id:
        raise ValueError("sample_id must be non-empty")
    if image.ndim != 2 or image.shape[0] != n_cells:
        raise ValueError("image must have shape [cells, image_features]")
    if coordinates_um.shape != (n_cells, 2):
        raise ValueError("coordinates_um must have shape [cells, 2]")
    if n_cells == 0:
        raise ValueError("a section must contain at least one aligned cell")
    if len(np.unique(np.asarray(cell_ids).astype(str))) != n_cells:
        raise ValueError("cell_ids must be unique within a section")
    if not np.isfinite(image).all():
        raise ValueError("image embeddings must be finite")
    if not np.isfinite(coordinates_um).all():
        raise ValueError("coordinates_um must be finite")


@dataclass(frozen=True)
class WindowSpec:
    """Geometry and fixed tensor sizes for one spatial JEPA window."""

    target_cells: int = 8
    context_cells: int = 64
    context_radius_um: float = 300.0
    target_guard_um: float = 80.0
    max_attempts: int = 64

    def __post_init__(self):
        if self.target_cells < 1 or self.context_cells < 1:
            raise ValueError("target_cells and context_cells must be positive")
        if not np.isfinite(self.context_radius_um) or self.context_radius_um <= 0:
            raise ValueError("context_radius_um must be positive")
        if not np.isfinite(self.target_guard_um) or self.target_guard_um < 0:
            raise ValueError("target_guard_um must be non-negative")
        if self.target_guard_um >= self.context_radius_um:
            raise ValueError("target_guard_um must be smaller than context_radius_um")
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be positive")


@dataclass(frozen=True)
class ContextSection:
    """Row-aligned UNI embeddings and physical cell centers for one section."""

    sample_id: str
    cell_ids: np.ndarray
    image: np.ndarray
    coordinates_um: np.ndarray

    def __post_init__(self):
        _validate_context_arrays(
            self.sample_id,
            self.cell_ids,
            self.image,
            self.coordinates_um,
        )


@dataclass(frozen=True)
class SectionEmbeddings:
    """Row-aligned UNI/TERRA embeddings and cell centers for one section."""

    sample_id: str
    cell_ids: np.ndarray
    image: np.ndarray
    terra: np.ndarray
    coordinates_um: np.ndarray

    def __post_init__(self):
        _validate_context_arrays(
            self.sample_id,
            self.cell_ids,
            self.image,
            self.coordinates_um,
        )
        if self.terra.ndim != 2 or self.terra.shape[0] != len(self.cell_ids):
            raise ValueError("terra must have shape [cells, terra_features]")
        if not np.isfinite(self.terra).all():
            raise ValueError("TERRA embeddings must be finite")


@dataclass(frozen=True)
class TargetStatistics:
    """Train-split TERRA centering and scaling statistics."""

    mean: np.ndarray
    std: np.ndarray
    count: int


def sample_context_rows(
    rows,
    distances,
    *,
    target_row: int,
    n_context: int,
    guard_radius_um: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Uniformly sample one target's visible annular context."""

    rows = np.asarray(rows, dtype=int)
    distances = np.asarray(distances, dtype=np.float64)
    if rows.ndim != 1 or distances.shape != rows.shape:
        raise ValueError("rows and distances must be matching one-dimensional arrays")
    if int(n_context) < 1 or float(guard_radius_um) < 0:
        raise ValueError("context count must be positive and guard radius non-negative")

    eligible = (rows != int(target_row)) & (distances >= float(guard_radius_um))
    rows, distances = rows[eligible], distances[eligible]
    if len(rows) > int(n_context):
        selected = rng.choice(len(rows), int(n_context), replace=False)
        rows, distances = rows[selected], distances[selected]
    order = np.lexsort((rows, distances))
    return rows[order]


def split_sample_ids(splits_path, split: str) -> list[str]:
    """Return one section-disjoint split from a complete assignment table."""

    return samples_in_split(splits_path, split)


def _manifest_rows(manifest_path) -> dict[str, dict]:
    table = pd.read_csv(manifest_path)
    required = {"id", "pixel_size_um"}
    if not required <= set(table.columns):
        raise ValueError(f"{manifest_path} must contain columns {sorted(required)}")
    table = table.copy()
    table["id"] = table["id"].astype(str)
    if table["id"].duplicated().any():
        duplicates = table.loc[table["id"].duplicated(keep=False), "id"].unique()
        raise ValueError(f"{manifest_path} contains duplicate samples {duplicates[:5]}")
    rows = {}
    for row in table.to_dict("records"):
        pixel_size_um = float(row["pixel_size_um"])
        if not np.isfinite(pixel_size_um) or pixel_size_um <= 0:
            raise ValueError(f"{row['id']}: pixel_size_um must be positive")
        rows[str(row["id"])] = row
    return rows


def _sample_directory(data_root, sample_id, manifest_row):
    output_dir = manifest_row.get("output_dir")
    if output_dir is not None and not pd.isna(output_dir):
        return Path(output_dir)
    return Path(data_root) / sample_id


def _aligned_rows(image_path, terra_path, sample_id):
    with np.load(image_path, allow_pickle=False) as data:
        if "cell_ids" not in data.files:
            raise ValueError(f"{image_path} is missing key 'cell_ids'")
        image_ids = np.asarray(data["cell_ids"]).astype(str)
    with np.load(terra_path, allow_pickle=False) as data:
        if "cell_ids" not in data.files:
            raise ValueError(f"{terra_path} is missing key 'cell_ids'")
        terra_ids = np.asarray(data["cell_ids"]).astype(str)
    if len(np.unique(image_ids)) != len(image_ids):
        raise ValueError(f"{image_path}: duplicate cell IDs")
    if len(np.unique(terra_ids)) != len(terra_ids):
        raise ValueError(f"{terra_path}: duplicate cell IDs")

    if np.array_equal(image_ids, terra_ids):
        image_rows = terra_rows = np.arange(len(image_ids), dtype=int)
    else:
        terra_row = {cell_id: row for row, cell_id in enumerate(terra_ids)}
        image_rows = np.flatnonzero([cell_id in terra_row for cell_id in image_ids])
        if len(image_rows) == 0:
            raise ValueError(f"{sample_id}: UNI and TERRA have no shared cells")
        terra_rows = np.asarray([terra_row[image_ids[row]] for row in image_rows])

    return image_ids[image_rows], image_rows, terra_rows, len(image_ids), len(terra_ids)


def _selected_rows(n_rows, max_rows, seed, sample_id):
    if max_rows is not None and int(max_rows) < 1:
        raise ValueError("max_cells_per_section must be positive or None")
    return np.asarray(
        select_indices(
            n_rows,
            None if max_rows is None else int(max_rows),
            int(seed),
            str(sample_id),
        ),
        dtype=int,
    )


def _load_context_components(
    data_root,
    sample_id,
    manifest_row,
    image_key,
    max_cells_per_section=None,
    seed=0,
):
    directory = _sample_directory(data_root, sample_id, manifest_row)
    image_path = directory / f"{sample_id}_uni2.npz"
    terra_path = directory / f"{sample_id}_terra.npz"

    cell_ids, image_rows, terra_rows, n_image, n_terra = _aligned_rows(
        image_path, terra_path, sample_id
    )
    selected = _selected_rows(len(cell_ids), max_cells_per_section, seed, sample_id)
    cell_ids = cell_ids[selected]
    image_rows = image_rows[selected]
    terra_rows = terra_rows[selected]

    # Keep cached sections compact while converting only the aligned rows.
    with np.load(image_path, allow_pickle=False) as data:
        if image_key not in data.files:
            raise ValueError(f"{image_path} is missing key {image_key!r}")
        values = np.asarray(data[image_key])
        if values.ndim != 2 or len(values) != n_image:
            raise ValueError(f"{image_path}: embeddings and cell IDs are misaligned")
        aligned_image = np.asarray(values[image_rows], dtype=np.float16)
    with np.load(terra_path, allow_pickle=False) as data:
        required = {"he_x", "he_y"}
        if not required <= set(data.files):
            missing = sorted(required - set(data.files))
            raise ValueError(f"{terra_path} is missing keys {missing}")
        he_x = np.asarray(data["he_x"])
        he_y = np.asarray(data["he_y"])
        if he_x.shape != (n_terra,) or he_y.shape != (n_terra,):
            raise ValueError(f"{terra_path}: H&E coordinates must have shape [cells]")
        aligned_coordinates = np.column_stack((he_x[terra_rows], he_y[terra_rows]))

    pixel_size_um = float(manifest_row["pixel_size_um"])
    coordinates_um = np.asarray(aligned_coordinates, dtype=np.float64) * pixel_size_um
    context = ContextSection(
        sample_id=str(sample_id),
        cell_ids=cell_ids,
        image=aligned_image,
        coordinates_um=coordinates_um,
    )
    return context, terra_path, terra_rows, n_terra


def _load_context_from_row(
    data_root,
    sample_id,
    manifest_row,
    image_key,
    max_cells_per_section=None,
    seed=0,
):
    context, _, _, _ = _load_context_components(
        data_root,
        sample_id,
        manifest_row,
        image_key,
        max_cells_per_section,
        seed,
    )
    return context


def _load_section_from_row(
    data_root,
    sample_id,
    manifest_row,
    image_key,
    terra_key,
    max_cells_per_section=None,
    seed=0,
):
    context, terra_path, terra_rows, n_terra = _load_context_components(
        data_root,
        sample_id,
        manifest_row,
        image_key,
        max_cells_per_section,
        seed,
    )
    with np.load(terra_path, allow_pickle=False) as data:
        if terra_key not in data.files:
            raise ValueError(f"{terra_path} is missing key {terra_key!r}")
        values = np.asarray(data[terra_key])
        if values.ndim != 2 or len(values) != n_terra:
            raise ValueError(f"{terra_path}: embeddings and cell IDs are misaligned")
        aligned_terra = np.asarray(values[terra_rows], dtype=np.float16)
    return SectionEmbeddings(
        sample_id=context.sample_id,
        cell_ids=context.cell_ids,
        image=context.image,
        terra=aligned_terra,
        coordinates_um=context.coordinates_um,
    )


def load_context_section(
    data_root,
    manifest_path,
    sample_id,
    *,
    image_key="uni_emb",
    max_cells_per_section=None,
    seed=0,
) -> ContextSection:
    """Load aligned UNI and micron coordinates without loading TERRA embeddings."""

    sample_id = str(sample_id)
    rows = _manifest_rows(manifest_path)
    if sample_id not in rows:
        raise KeyError(f"manifest is missing sample {sample_id!r}")
    return _load_context_from_row(
        data_root,
        sample_id,
        rows[sample_id],
        image_key,
        max_cells_per_section,
        seed,
    )


def load_section(
    data_root,
    manifest_path,
    sample_id,
    *,
    image_key="uni_emb",
    terra_key="cell_emb",
    max_cells_per_section=None,
    seed=0,
) -> SectionEmbeddings:
    """Load and align one section, converting H&E pixel centers to microns."""

    sample_id = str(sample_id)
    rows = _manifest_rows(manifest_path)
    if sample_id not in rows:
        raise KeyError(f"manifest is missing sample {sample_id!r}")
    return _load_section_from_row(
        data_root,
        sample_id,
        rows[sample_id],
        image_key,
        terra_key,
        max_cells_per_section,
        seed,
    )


def load_terra_targets(
    data_root,
    manifest_path,
    sample_id,
    *,
    terra_key="cell_emb",
    max_cells_per_section=None,
    seed=0,
) -> np.ndarray:
    """Load capped TERRA rows without materializing the large UNI embedding array."""

    sample_id = str(sample_id)
    rows = _manifest_rows(manifest_path)
    if sample_id not in rows:
        raise KeyError(f"manifest is missing sample {sample_id!r}")
    directory = _sample_directory(data_root, sample_id, rows[sample_id])
    image_path = directory / f"{sample_id}_uni2.npz"
    terra_path = directory / f"{sample_id}_terra.npz"
    cell_ids, _, terra_rows, _, n_terra = _aligned_rows(
        image_path, terra_path, sample_id
    )
    selected = _selected_rows(len(cell_ids), max_cells_per_section, seed, sample_id)
    terra_rows = terra_rows[selected]
    with np.load(terra_path, allow_pickle=False) as data:
        if terra_key not in data.files:
            raise ValueError(f"{terra_path} is missing key {terra_key!r}")
        terra = np.asarray(data[terra_key])
        if terra.ndim != 2 or len(terra) != n_terra:
            raise ValueError(f"{terra_path}: embeddings and cell IDs are misaligned")
        targets = np.asarray(terra[terra_rows], dtype=np.float16)
    if not np.isfinite(targets).all():
        raise ValueError(f"{terra_path}: TERRA embeddings must be finite")
    return targets


class SpatialWindowSampler:
    """Draw target-centered context rings from one tissue section."""

    def __init__(
        self,
        section: SectionEmbeddings,
        spec: WindowSpec,
        anchor_indices=None,
    ):
        if len(section.cell_ids) < spec.target_cells + spec.context_cells:
            raise ValueError(
                f"{section.sample_id}: too few cells for {spec.target_cells} targets "
                f"and {spec.context_cells} context cells"
            )
        if anchor_indices is None:
            anchor_indices = np.arange(len(section.cell_ids), dtype=int)
        anchor_indices = np.asarray(anchor_indices, dtype=int)
        if anchor_indices.ndim != 1 or len(anchor_indices) == 0:
            raise ValueError("anchor_indices must be a non-empty one-dimensional array")
        if np.any(anchor_indices < 0) or np.any(
            anchor_indices >= len(section.cell_ids)
        ):
            raise ValueError("anchor_indices contain an out-of-range row")
        if len(np.unique(anchor_indices)) != len(anchor_indices):
            raise ValueError("anchor_indices must be unique")
        self.section = section
        self.spec = spec
        self.anchor_indices = anchor_indices.copy()
        self._tree = cKDTree(section.coordinates_um)

    def _target_indices(self, anchor: int) -> np.ndarray:
        # Ask for one extra neighbour so the explicit anchor can always be placed
        # first, even when multiple cells have identical coordinates.
        count = min(len(self.section.cell_ids), self.spec.target_cells + 1)
        _, neighbours = self._tree.query(
            self.section.coordinates_um[anchor],
            k=count,
        )
        neighbours = np.atleast_1d(neighbours).astype(int)
        others = neighbours[neighbours != anchor][: self.spec.target_cells - 1]
        if len(others) != self.spec.target_cells - 1:
            raise RuntimeError("could not construct the nearest-cell target block")
        return np.concatenate(([anchor], others))

    def _context_candidates(self, target: int) -> tuple[np.ndarray, np.ndarray]:
        nearby = np.asarray(
            self._tree.query_ball_point(
                self.section.coordinates_um[target],
                self.spec.context_radius_um,
            ),
            dtype=int,
        )
        if len(nearby) == 0:
            return nearby, np.empty(0, dtype=np.float64)
        distances = np.linalg.norm(
            self.section.coordinates_um[nearby] - self.section.coordinates_um[target],
            axis=1,
        )
        return nearby, distances

    def sample(self, rng: np.random.Generator) -> dict:
        """Sample one valid window using only randomness supplied by ``rng``."""

        for _ in range(self.spec.max_attempts):
            anchor = int(self.anchor_indices[rng.integers(len(self.anchor_indices))])
            targets = self._target_indices(anchor)
            context = [
                sample_context_rows(
                    *self._context_candidates(target),
                    target_row=target,
                    n_context=self.spec.context_cells,
                    guard_radius_um=self.spec.target_guard_um,
                    rng=rng,
                )
                for target in targets
            ]
            if any(len(rows) < self.spec.context_cells for rows in context):
                continue
            return self._window(anchor, np.stack(context), targets)

        raise RuntimeError(
            f"{self.section.sample_id}: no valid spatial window found after "
            f"{self.spec.max_attempts} attempts; reduce target_guard_um or cell counts"
        )

    def _window(self, anchor, context, targets):
        origins = self.section.coordinates_um[targets, None, :]
        scale = self.spec.context_radius_um
        context_coordinates = (self.section.coordinates_um[context] - origins) / scale
        target_coordinates = np.zeros((len(targets), 1, 2), dtype=np.float32)
        return {
            "context_image": torch.from_numpy(
                np.asarray(self.section.image[context], dtype=np.float32)
            ),
            "context_coords": torch.from_numpy(context_coordinates.astype(np.float32)),
            "target_terra": torch.from_numpy(
                np.asarray(self.section.terra[targets, None, :], dtype=np.float32)
            ),
            "target_coords": torch.from_numpy(target_coordinates),
            "coordinate_normalization": "fixed_radius",
            "sample_id": self.section.sample_id,
            "anchor_cell_id": str(self.section.cell_ids[anchor]),
            "context_cell_ids": tuple(
                tuple(row) for row in self.section.cell_ids[context].astype(str)
            ),
            "target_cell_ids": tuple(self.section.cell_ids[targets].astype(str)),
        }


class SpatialJEPADataset(Dataset):
    """Deterministic, section-local spatial windows from fixed HEST splits.

    Index blocks are contiguous by section. Call :meth:`set_epoch` before each
    epoch and use ``shuffle=False`` in the DataLoader; the dataset deterministically
    permutes section visits itself.
    """

    def __init__(
        self,
        data_root,
        manifest_path,
        splits_path,
        split,
        windows_per_section,
        spec: WindowSpec,
        *,
        seed=0,
        image_key="uni_emb",
        terra_key="cell_emb",
        max_anchor_cells_per_section=None,
        max_open_sections=1,
    ):
        if int(windows_per_section) < 1:
            raise ValueError("windows_per_section must be positive")
        if int(max_open_sections) < 1:
            raise ValueError("max_open_sections must be positive")
        if (
            max_anchor_cells_per_section is not None
            and int(max_anchor_cells_per_section) < 1
        ):
            raise ValueError("max_anchor_cells_per_section must be positive or None")
        self.sample_ids = split_sample_ids(splits_path, split)
        self._manifest = _manifest_rows(manifest_path)
        missing = sorted(set(self.sample_ids) - set(self._manifest))
        if missing:
            raise KeyError(f"manifest is missing split samples {missing[:5]}")

        self.data_root = Path(data_root)
        self.manifest_path = Path(manifest_path)
        self.split = str(split)
        self.windows_per_section = int(windows_per_section)
        self.spec = spec
        self.seed = int(seed)
        self.image_key = str(image_key)
        self.terra_key = str(terra_key)
        self.max_anchor_cells_per_section = (
            None
            if max_anchor_cells_per_section is None
            else int(max_anchor_cells_per_section)
        )
        self.max_open_sections = int(max_open_sections)
        self.epoch = 0
        self._cache = OrderedDict()
        self._section_order = np.arange(len(self.sample_ids), dtype=int)
        self.set_epoch(0)

    @property
    def section_order(self):
        return tuple(self.sample_ids[index] for index in self._section_order)

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)
        rng = np.random.default_rng([self.seed, self.epoch])
        self._section_order = rng.permutation(len(self.sample_ids))

    def __len__(self):
        return len(self.sample_ids) * self.windows_per_section

    def _sampler(self, sample_index):
        sample_id = self.sample_ids[sample_index]
        if sample_id in self._cache:
            return self._cache.pop(sample_id)
        section = _load_section_from_row(
            self.data_root,
            sample_id,
            self._manifest[sample_id],
            self.image_key,
            self.terra_key,
        )
        anchor_indices = _selected_rows(
            len(section.cell_ids),
            self.max_anchor_cells_per_section,
            self.seed,
            sample_id,
        )
        sampler = SpatialWindowSampler(section, self.spec, anchor_indices)
        if len(self._cache) == self.max_open_sections:
            self._cache.popitem(last=False)
        return sampler

    def __getitem__(self, index):
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        section_visit, local_window = divmod(index, self.windows_per_section)
        sample_index = int(self._section_order[section_visit])
        sampler = self._sampler(sample_index)
        self._cache[sampler.section.sample_id] = sampler
        rng = np.random.default_rng([self.seed, self.epoch, sample_index, local_window])
        return sampler.sample(rng)

    def clear_cache(self):
        self._cache.clear()

    def load_section(self, sample_id) -> SectionEmbeddings:
        """Load one configured split section without retaining it in the LRU."""

        sample_id = str(sample_id)
        if sample_id not in self.sample_ids:
            raise KeyError(f"sample {sample_id!r} is outside split {self.split!r}")
        return _load_section_from_row(
            self.data_root,
            sample_id,
            self._manifest[sample_id],
            self.image_key,
            self.terra_key,
        )

    def iter_sections(self):
        """Stream configured sections in split-table order."""

        for sample_id in self.sample_ids:
            yield self.load_section(sample_id)

    def iter_target_arrays(self):
        """Stream capped TERRA arrays without loading UNI feature matrices."""

        for sample_id in self.sample_ids:
            yield load_terra_targets(
                self.data_root,
                self.manifest_path,
                sample_id,
                terra_key=self.terra_key,
                max_cells_per_section=self.max_anchor_cells_per_section,
                seed=self.seed,
            )

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_cache"] = OrderedDict()
        return state


def collate_windows(windows):
    """Flatten window and target axes into independent target-centered examples."""

    if not windows:
        raise ValueError("cannot collate an empty batch")
    tensor_keys = ("context_image", "context_coords", "target_terra", "target_coords")
    n_targets = windows[0]["target_terra"].shape[0]
    if any(window["target_terra"].shape[0] != n_targets for window in windows):
        raise ValueError("all windows must contain the same number of targets")
    batch = {
        key: torch.stack([window[key] for window in windows]).flatten(0, 1)
        for key in tensor_keys
    }
    batch["sample_id"] = [
        window["sample_id"] for window in windows for _ in range(n_targets)
    ]
    batch["anchor_cell_id"] = [
        window["anchor_cell_id"] for window in windows for _ in range(n_targets)
    ]
    batch["coordinate_normalization"] = [
        window["coordinate_normalization"]
        for window in windows
        for _ in range(n_targets)
    ]
    batch["context_cell_ids"] = [
        context_ids for window in windows for context_ids in window["context_cell_ids"]
    ]
    batch["target_cell_ids"] = [
        target_id for window in windows for target_id in window["target_cell_ids"]
    ]
    return batch


def fit_target_statistics(sections) -> TargetStatistics:
    """Fit population mean/std from sections or raw ``[cells, features]`` arrays."""

    if isinstance(sections, SpatialJEPADataset):
        sections = sections.iter_target_arrays()
    count = 0
    mean = second_moment = None
    for section in sections:
        targets = section.terra if isinstance(section, SectionEmbeddings) else section
        values = np.asarray(targets, dtype=np.float64)
        if values.ndim != 2 or len(values) == 0:
            raise ValueError("each section must contain non-empty 2D TERRA targets")
        section_count = len(values)
        section_mean = values.mean(axis=0)
        section_second = np.square(values - section_mean).sum(axis=0)
        if mean is None:
            mean = section_mean
            second_moment = section_second
            count = section_count
            continue
        if values.shape[1] != len(mean):
            raise ValueError("all sections must use the same TERRA feature dimension")
        total = count + section_count
        delta = section_mean - mean
        second_moment += (
            section_second + np.square(delta) * count * section_count / total
        )
        mean += delta * section_count / total
        count = total
    if count == 0:
        raise ValueError("cannot fit target statistics from zero sections")
    std = np.sqrt(second_moment / count)
    std = np.maximum(std, np.finfo(np.float32).eps)
    return TargetStatistics(mean.astype(np.float32), std.astype(np.float32), count)
