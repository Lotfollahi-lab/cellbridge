import pandas as pd
import pytest

from cellbridge.hierarchical_conformal.hest_data import load_pruned_ontology_hierarchy


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
