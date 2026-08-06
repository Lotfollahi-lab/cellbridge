import pandas as pd

from cellbridge.hierarchical.plot_tree import aggregate_node_counts, build_dot


def test_node_counts_aggregate_states_and_stop_partial_paths():
    usage = pd.DataFrame(
        {
            "mapping_status": ["full", "full", "partial", "excluded"],
            "hierarchy_path": [
                "cell > epithelial > tumor > neoplastic cell",
                "cell > epithelial > tumor > neoplastic cell",
                "cell > epithelial > epithelial",
                "",
            ],
            "n_cells": [10, 5, 7, 100],
        }
    )

    counts = aggregate_node_counts(usage)

    assert counts[(0, "", "cell")] == 22
    assert counts[(1, "cell", "epithelial")] == 22
    assert counts[(2, "epithelial", "tumor")] == 15
    assert counts[(2, "epithelial", "epithelial")] == 7
    assert counts[(3, "tumor", "neoplastic cell")] == 15


def test_dot_labels_every_node_with_formatted_counts():
    nodes = pd.DataFrame(
        {
            "level": [0, 1, 2, 3],
            "parent": ["", "cell", "epithelial", "tumor"],
            "node": ["cell", "epithelial", "tumor", "neoplastic cell"],
            "cl_id": ["CL:0000000", "", "", "CL:0001063"],
        }
    )
    usage = pd.DataFrame(
        {
            "status": ["full"],
            "hierarchy_path": ["cell > epithelial > tumor > neoplastic cell"],
            "n_cells": [12345],
        }
    )

    dot = build_dot(nodes, usage)

    assert dot.count("n=12,345") == 4
    assert "neoplastic cell\\nCL:0001063\\nn=12,345" in dot
