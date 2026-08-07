"""Align reviewed hierarchy metadata to capped LoRA cell datasets."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from cellbridge.lora.data import canonical_id

LEVEL_COLUMNS = (
    "level_0",
    "level_1_compartment",
    "level_2_lineage",
    "level_3_identity",
)


def load_ontology_nodes(path: str | Path) -> pd.DataFrame:
    """Load the reviewed tree and assign stable global node/parent indices."""

    path = Path(path)
    nodes = pd.read_csv(path, keep_default_na=False)
    required = {"level", "parent", "node"}
    if not required <= set(nodes):
        raise ValueError(f"{path}: hierarchy nodes require {sorted(required)}")
    nodes = nodes.copy()
    nodes["level"] = pd.to_numeric(nodes["level"], errors="raise").astype(int)
    nodes[["parent", "node"]] = nodes[["parent", "node"]].astype(str)
    if nodes.empty or nodes[["level", "node"]].duplicated().any():
        raise ValueError(f"{path}: hierarchy nodes must be non-empty and unique")
    if (nodes["level"].to_numpy() < 0).any() or (nodes["level"].to_numpy() > 3).any():
        raise ValueError(f"{path}: hierarchy node levels must lie in [0, 3]")
    nodes.insert(0, "node_id", np.arange(len(nodes), dtype=np.int64))
    lookup = {
        (int(row.level), str(row.node)): int(row.node_id)
        for row in nodes.itertuples(index=False)
    }
    parents = []
    for row in nodes.itertuples(index=False):
        if row.level == 0:
            if row.parent:
                raise ValueError(f"{path}: level-0 nodes cannot have parents")
            parents.append(-1)
            continue
        parent = lookup.get((int(row.level) - 1, str(row.parent)))
        if parent is None or parent >= int(row.node_id):
            raise ValueError(f"{path}: invalid parent for node {row.node!r}")
        parents.append(parent)
    nodes["parent_id"] = np.asarray(parents, dtype=np.int64)
    return nodes


def encode_ontology_node_ids(metadata: pd.DataFrame, nodes: pd.DataFrame) -> np.ndarray:
    """Encode each known path node with IDs from ``hierarchy_nodes.csv``."""

    encoded = np.full((len(metadata), len(LEVEL_COLUMNS)), -1, dtype=np.int64)
    for level, column in enumerate(LEVEL_COLUMNS):
        level_nodes = nodes.loc[nodes["level"] == level, ["node", "node_id"]]
        lookup = dict(zip(level_nodes["node"].astype(str), level_nodes["node_id"]))
        values = metadata[column].astype(str)
        unknown = sorted(set(values[values != ""]) - set(lookup))
        if unknown:
            raise ValueError(f"ontology table is missing level-{level} nodes {unknown}")
        encoded[:, level] = values.map(lookup).fillna(-1).to_numpy(dtype=np.int64)
    return encoded


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
    if not np.equal(depth, depth.astype(int)).all() or np.any(
        (depth < 0) | (depth > 3)
    ):
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


def load_aligned_hierarchy(path, cell_ids, sample_id) -> pd.DataFrame:
    """Load and validate reviewed hierarchy rows in a requested cell order."""

    path = Path(path)
    cell_ids = np.asarray([canonical_id(value) for value in cell_ids])
    table = _load_aligned(path, cell_ids, str(sample_id))
    _validate(table, path)
    return table


class HierarchyDataset(Dataset):
    """Add encoded hierarchy fields without altering the underlying cell data."""

    def __init__(
        self,
        base,
        labels_dir: str | Path,
        nodes_path: str | Path | None = None,
    ):
        self.base = base
        labels_dir = Path(labels_dir)
        aligned = []
        for sample_id, cell_ids in base.sample_cell_ids.items():
            path = labels_dir / f"{sample_id}.parquet"
            if not path.exists():
                raise FileNotFoundError(f"missing hierarchy labels: {path}")
            aligned.append(load_aligned_hierarchy(path, cell_ids, sample_id))
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
        self.node_ids = None
        self.ontology_depths = self.ontology_parents = None
        if nodes_path is not None:
            nodes = load_ontology_nodes(nodes_path)
            self.node_ids = torch.from_numpy(encode_ontology_node_ids(metadata, nodes))
            self.ontology_depths = nodes["level"].to_numpy(dtype=np.int64)
            self.ontology_parents = nodes["parent_id"].to_numpy(dtype=np.int64)

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
        if self.node_ids is not None:
            item["hierarchy_node_ids"] = self.node_ids[index]
        return item

    def close(self):
        self.base.close()
