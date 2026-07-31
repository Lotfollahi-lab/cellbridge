"""Align reviewed hierarchy metadata to capped LoRA cell datasets."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from terra_multi.lora.data import canonical_id

LEVEL_COLUMNS = (
    "level_0",
    "level_1_compartment",
    "level_2_lineage",
    "level_3_identity",
)


def _status_column(path: Path) -> str:
    import pyarrow.parquet as pq

    columns = set(pq.ParquetFile(path).schema.names)
    if "mapping_status" in columns:
        return "mapping_status"
    if "status" in columns:
        return "status"
    raise ValueError(f"{path}: hierarchy labels have no mapping_status column")


def _load_aligned(path: Path, cell_ids: np.ndarray, sample_id: str) -> pd.DataFrame:
    status_column = _status_column(path)
    columns = [
        "sample_id",
        "cell_id",
        status_column,
        "hierarchy_depth",
        "use_in_hierarchy",
        *LEVEL_COLUMNS,
    ]
    table = pd.read_parquet(path, columns=columns).rename(
        columns={status_column: "mapping_status"}
    )
    table["sample_id"] = table["sample_id"].astype(str)
    if set(table["sample_id"]) != {str(sample_id)}:
        raise ValueError(f"{path}: sample_id does not match {sample_id}")
    table["cell_id"] = table["cell_id"].map(canonical_id)
    if table["cell_id"].duplicated().any():
        raise ValueError(f"{path}: duplicate hierarchy cell IDs")
    rows = pd.Index(table["cell_id"]).get_indexer(cell_ids)
    absent = rows < 0
    # Index 0 is a placeholder for absent cells; every column is overwritten
    # below. Avoiding -1 keeps pandas from silently wrapping to the last row.
    aligned = table.iloc[np.where(absent, 0, rows)].reset_index(drop=True)
    if absent.any():
        # Cells the granular clustering dropped have no identity at any level.
        # They are unusable for hierarchy terms in exactly the way an excluded
        # label is, but they must not block training: the paired loss is
        # label-free and needs every capped cell. Raising here would cap the
        # usable dataset at whatever the label file happens to cover.
        aligned.loc[absent, "mapping_status"] = "excluded"
        aligned.loc[absent, "hierarchy_depth"] = 0
        aligned.loc[absent, "use_in_hierarchy"] = False
        for column in LEVEL_COLUMNS:
            aligned.loc[absent, column] = ""
        print(
            f"{path.name}: {int(absent.sum()):,} of {len(cell_ids):,} capped cells "
            "have no hierarchy label; marked unusable"
        )
    aligned["cell_id"] = cell_ids
    aligned["sample_id"] = str(sample_id)
    return aligned


def _validate(table: pd.DataFrame, path: Path) -> None:
    if table.isna().any().any():
        raise ValueError(f"{path}: hierarchy fields must be non-null")
    status = table["mapping_status"].astype(str)
    unknown = sorted(set(status) - {"full", "partial", "excluded"})
    if unknown:
        raise ValueError(f"{path}: unknown mapping_status values {unknown}")
    depth = pd.to_numeric(table["hierarchy_depth"], errors="raise").to_numpy()
    if not np.equal(depth, depth.astype(int)).all() or np.any((depth < 0) | (depth > 3)):
        raise ValueError(f"{path}: hierarchy_depth must be an integer in [0, 3]")
    if not pd.api.types.is_bool_dtype(table["use_in_hierarchy"]):
        raise ValueError(f"{path}: use_in_hierarchy must be boolean")
    use = table["use_in_hierarchy"].to_numpy()
    if np.any((status == "full") & ((depth != 3) | ~use)):
        raise ValueError(f"{path}: full labels must be usable through identity")
    if np.any((status == "partial") & (((depth < 1) | (depth >= 3)) | ~use)):
        raise ValueError(f"{path}: partial labels have an unsafe hierarchy depth")
    if np.any((status == "excluded") & ((depth != 0) | use)):
        raise ValueError(f"{path}: excluded labels cannot be hierarchy-usable")
    values = table.loc[:, LEVEL_COLUMNS].astype(str).to_numpy()
    for row in np.flatnonzero(use):
        known_depth = int(depth[row])
        if any(not values[row, level] for level in range(known_depth + 1)):
            raise ValueError(f"{path}: missing node within hierarchy_depth")
        if any(values[row, level] for level in range(known_depth + 1, 4)):
            raise ValueError(f"{path}: partial label has nodes beyond hierarchy_depth")


class HierarchyDataset(Dataset):
    """Add encoded hierarchy fields without altering the underlying cell data."""

    def __init__(self, base, labels_dir: str | Path):
        self.base = base
        labels_dir = Path(labels_dir)
        aligned = []
        for sample_id, cell_ids in base.sample_cell_ids.items():
            path = labels_dir / f"{sample_id}.parquet"
            if not path.exists():
                raise FileNotFoundError(f"missing hierarchy labels: {path}")
            table = _load_aligned(path, cell_ids, sample_id)
            _validate(table, path)
            aligned.append(table)
        metadata = pd.concat(aligned, ignore_index=True)
        if len(metadata) != len(base):
            raise RuntimeError("hierarchy alignment changed dataset length")

        encoded = np.full((len(metadata), 4), -1, dtype=np.int64)
        for level, column in enumerate(LEVEL_COLUMNS):
            values = metadata[column].astype(str)
            nodes = sorted(value for value in values.unique() if value)
            codes = {node: code for code, node in enumerate(nodes)}
            encoded[:, level] = values.map(codes).fillna(-1).to_numpy(dtype=np.int64)
        status = metadata["mapping_status"].astype(str)
        raw_use = metadata["use_in_hierarchy"].to_numpy(dtype=bool)
        self.levels = torch.from_numpy(encoded)
        self.depth = torch.from_numpy(
            metadata["hierarchy_depth"].to_numpy(dtype=np.int64)
        )
        self.usable = torch.from_numpy(
            raw_use & status.isin(["full", "partial"]).to_numpy()
        )
        self.mapping_status = tuple(status)
        self.sample_ids_by_index = metadata["sample_id"].to_numpy(dtype=str)

    @property
    def sample_indices(self):
        return self.base.sample_indices

    @property
    def sample_ids(self):
        return self.base.sample_ids

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        item = self.base[index]
        item.update(
            hierarchy_levels=self.levels[index],
            hierarchy_depth=self.depth[index],
            hierarchy_usable=self.usable[index],
            mapping_status=self.mapping_status[index],
        )
        return item

    def close(self):
        self.base.close()
