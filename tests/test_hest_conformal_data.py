import pandas as pd
import pytest

from cellbridge.hierarchical_conformal.hest_data import load_pruned_ontology_hierarchy
from cellbridge.hierarchical_conformal.splits import load_phenotype_split


def test_ontology_hierarchy_disambiguates_equal_compartment_and_lineage(tmp_path):
    path = tmp_path / "nodes.csv"
    pd.DataFrame(
        [
            (0, "", "cell"),
            (1, "cell", "epithelial"),
            (2, "epithelial", "epithelial"),
            (3, "epithelial", "epithelial cell"),
            (2, "epithelial", "neoplastic"),
            (3, "neoplastic", "neoplastic cell"),
        ],
        columns=["level", "parent", "node"],
    ).to_csv(path, index=False)

    hierarchy = load_pruned_ontology_hierarchy(
        path, ["epithelial cell", "neoplastic cell"]
    )
    assert hierarchy.leaves == ("epithelial cell", "neoplastic cell")
    assert hierarchy.depths["epithelial cell"] == 3
    assert "compartment::epithelial" in hierarchy.nodes
    assert "lineage::epithelial" in hierarchy.nodes


def test_ontology_hierarchy_rejects_unknown_identity(tmp_path):
    path = tmp_path / "nodes.csv"
    pd.DataFrame(
        [(0, "", "cell"), (1, "cell", "immune")],
        columns=["level", "parent", "node"],
    ).to_csv(path, index=False)
    with pytest.raises(ValueError, match="no depth-three identity"):
        load_pruned_ontology_hierarchy(path, ["unknown cell"])


def _write_split_inputs(tmp_path):
    base = tmp_path / "base.csv"
    pd.DataFrame(
        {
            "sample_id": ["base_train", "base_val", "A", "B", "C", "X"],
            "split": [
                "distill_train",
                "distill_val",
                "label_test",
                "label_test",
                "label_test",
                "label_test",
            ],
        }
    ).to_csv(base, index=False)
    manifest = tmp_path / "phenotype.csv"
    pd.DataFrame(
        {
            "sample_id": ["A", "B", "C", "X"],
            "organ": ["breast", "breast", "breast", "ovary"],
            "patient_id": ["p1", "p2", "p3", ""],
            "role": [
                "breast_train",
                "breast_val",
                "breast_test",
                "cross_organ_test",
            ],
        }
    ).to_csv(manifest, index=False)
    return base, manifest


def test_phenotype_split_is_patient_disjoint(tmp_path):
    base, manifest = _write_split_inputs(tmp_path)
    split = load_phenotype_split(manifest, base)

    assert split.base_train == ("base_train",)
    assert split.base_val == ("base_val",)
    assert split.breast_train == ("A",)
    assert split.breast_val == ("B",)
    assert split.breast_test == ("C",)
    assert split.cross_organ_test == ("X",)


def test_phenotype_split_rejects_patient_leakage(tmp_path):
    base, manifest = _write_split_inputs(tmp_path)
    table = pd.read_csv(manifest, keep_default_na=False)
    table.loc[table["sample_id"].eq("B"), "patient_id"] = "p1"
    table.to_csv(manifest, index=False)

    with pytest.raises(ValueError, match="patients cross roles"):
        load_phenotype_split(manifest, base)
