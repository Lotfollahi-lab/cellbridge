"""Patient-disjoint HEST cohorts for conformal evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from cellbridge.helpers.cohorts import load_split_table

ROLES = ("breast_train", "breast_val", "breast_test", "cross_organ_test")


@dataclass(frozen=True)
class PhenotypeSplit:
    base_train: tuple[str, ...]
    base_val: tuple[str, ...]
    breast_train: tuple[str, ...]
    breast_val: tuple[str, ...]
    breast_test: tuple[str, ...]
    cross_organ_test: tuple[str, ...]


def _read_manifest(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    table = pd.read_csv(path, keep_default_na=False)
    required = {"sample_id", "organ", "patient_id", "role"}
    if not required <= set(table):
        raise ValueError(f"{path} must contain columns {sorted(required)}")
    table = table.loc[:, sorted(required)].copy()
    for column in required:
        table[column] = table[column].astype(str).str.strip()
    if table["sample_id"].eq("").any() or table["role"].eq("").any():
        raise ValueError(f"{path} contains empty sample IDs or roles")
    duplicates = table.loc[table["sample_id"].duplicated(), "sample_id"].tolist()
    if duplicates:
        raise ValueError(f"{path} contains duplicate samples: {duplicates[:5]}")
    unknown = sorted(set(table["role"]) - set(ROLES))
    if unknown:
        raise ValueError(f"{path} contains unknown roles: {unknown}")
    missing = [role for role in ROLES if not table["role"].eq(role).any()]
    if missing:
        raise ValueError(f"{path} has empty roles: {missing}")
    return table


def _validate_patients(table: pd.DataFrame, path: str | Path) -> None:
    breast = table[table["role"].str.startswith("breast_")]
    if breast["patient_id"].eq("").any():
        raise ValueError(f"{path}: every breast sample requires a patient_id")
    roles_per_patient = breast.groupby("patient_id")["role"].nunique()
    leaking = roles_per_patient[roles_per_patient > 1].index.tolist()
    if leaking:
        raise ValueError(f"{path}: patients cross roles: {leaking}")
    if not breast["organ"].str.lower().eq("breast").all():
        raise ValueError(f"{path}: breast roles must contain only breast samples")
    cross = table[table["role"].eq("cross_organ_test")]
    if cross["organ"].str.lower().eq("breast").any():
        raise ValueError(f"{path}: cross_organ_test contains a breast sample")


def load_phenotype_split(
    manifest_path: str | Path,
    base_splits_path: str | Path,
    *,
    expected_counts: dict[str, int] | None = None,
) -> PhenotypeSplit:
    """Load fixed cohorts and prove that patients and roles are disjoint."""

    table = _read_manifest(manifest_path)
    _validate_patients(table, manifest_path)
    base = load_split_table(base_splits_path).set_index("sample_id")["split"]
    outside = [
        sample for sample in table["sample_id"] if base.get(sample) != "label_test"
    ]
    if outside:
        raise ValueError(
            f"{manifest_path}: annotated samples outside base label_test: {outside[:5]}"
        )

    groups = {
        role: tuple(table.loc[table["role"].eq(role), "sample_id"])
        for role in ROLES
    }
    if expected_counts:
        observed = {role: len(groups[role]) for role in ROLES}
        mismatched = {
            role: (observed[role], int(count))
            for role, count in expected_counts.items()
            if role not in observed or observed[role] != int(count)
        }
        if mismatched:
            raise ValueError(f"{manifest_path}: unexpected role counts {mismatched}")

    split = PhenotypeSplit(
        base_train=tuple(base[base.eq("distill_train")].index),
        base_val=tuple(base[base.eq("distill_val")].index),
        **groups,
    )
    partitions = [
        set(split.base_train) | set(split.breast_train),
        set(split.base_val) | set(split.breast_val),
        set(split.breast_test),
        set(split.cross_organ_test),
    ]
    if any(
        left & right
        for index, left in enumerate(partitions)
        for right in partitions[index + 1 :]
    ):
        raise ValueError("training, validation, and test cohorts are not disjoint")
    return split
