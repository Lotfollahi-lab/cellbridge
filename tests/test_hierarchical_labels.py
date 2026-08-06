import pandas as pd

from cellbridge.hierarchical.build_labels import run
from cellbridge.hierarchical.mapping import RULES, map_label, split_source_label


def test_reviewed_rules_cover_the_normalized_inventory():
    assert len(RULES) == 91
    assert all(map_label(label)["normalized_label"] == label for label in RULES)


def test_cluster_ids_do_not_create_biological_leaves():
    zero = map_label("0_Tumor_Cells")
    three = map_label("3_Tumor_Cells")

    assert split_source_label("12_Tumor_Cells") == ("12", "Tumor_Cells")
    assert zero["source_cluster"] == "0"
    assert three["source_cluster"] == "3"
    assert zero["cl_id"] == three["cl_id"] == "CL:0001063"
    assert zero["hierarchy_path"] == three["hierarchy_path"]
    assert zero["malignancy"] == three["malignancy"] == "malignant"


def test_partial_and_excluded_labels_have_safe_depths():
    composite = map_label("0_Epithelial_&_Tumor_Cells")
    necrotic = map_label("2_Necrotic_Cells")

    assert composite["mapping_status"] == "partial"
    assert composite["hierarchy_depth"] == 2
    assert composite["level_3_identity"] == ""
    assert composite["malignancy"] == "ambiguous"
    assert composite["use_in_hierarchy"]
    assert necrotic["mapping_status"] == "excluded"
    assert not necrotic["use_in_hierarchy"]


def test_builder_writes_audit_and_per_cell_labels(tmp_path):
    data = tmp_path / "hest"
    rows = {
        "A": [("a1", "0_Tumor_Cells"), ("a2", "1_T_Lymphocytes")],
        "B": [("b1", "3_Tumor_Cells"), ("b2", "2_Necrotic_Cells")],
    }
    for sample, values in rows.items():
        path = data / sample
        path.mkdir(parents=True)
        pd.DataFrame(values, columns=["cell_id", "group"]).to_csv(
            path / f"{sample}_labels.csv", index=False
        )
    splits = data / "splits.csv"
    pd.DataFrame({"sample_id": ["A", "B"], "split": ["label_test"] * 2}).to_csv(
        splits, index=False
    )
    output = tmp_path / "output"
    metadata = run(
        {
            "data_root": str(data),
            "splits_path": str(splits),
            "split": "label_test",
            "labels_pattern": "{data_root}/{sample_id}/{sample_id}_labels.csv",
            "label_id_column": "cell_id",
            "label_column": "group",
            "output_dir": str(output),
            "materialize_cells": True,
        }
    )

    mapping = pd.read_csv(output / "source_label_mapping.csv")
    cells = pd.read_parquet(output / "cells" / "A.parquet")
    assert metadata["source_labels"] == 4
    assert len(mapping) == 4
    assert set(cells["cl_id"]) == {"CL:0001063", "CL:0000084"}
    assert (output / "REPORT.md").exists()
