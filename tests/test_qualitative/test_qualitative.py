"""Notebook 09 helpers: patch statistics, tissue-map hover labels and per-user saving."""

import getpass
import json
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from cellbridge.qualitative import data, figures


def _rows():
    # Three evaluated cells in patch r000c000, one in r000c001, one without a prediction.
    return pd.DataFrame(
        {
            "row_id": ["a", "b", "c", "d", "e"],
            "x": [10.0, 20.0, 30.0, 110.0, 40.0],
            "y": [10.0, 50.0, 90.0, 10.0, 40.0],
            "evaluated": [True, True, True, True, False],
            "depth": [0.0, 2.0, np.nan, 1.0, np.nan],
            "set_size": [1, 3, 2, 0, np.nan],
            "ancestor_covered": [True, True, False, False, np.nan],
            "raw_covered": [True, False, False, False, np.nan],
            "truth": ["t", "t", "b", "t", None],
            "predicted": ["t", "b", "b", "b", None],
        }
    )


def test_patch_statistics_summarizes_each_grid_patch():
    table = data.patch_statistics(_rows(), size=100, shape=(150, 150), min_cells=1)
    first, second = table.set_index("patch").loc[["r000c000", "r000c001"]].itertuples()
    assert first.evaluated_cells == 3
    assert first.mean_depth == 1.0  # the excluded cell carries no depth
    assert first.excluded_fraction == pytest.approx(1 / 3)
    assert first.top1_accuracy == pytest.approx(2 / 3)
    assert (second.x0, second.y0, second.x1, second.y1) == (100, 0, 150, 100)  # clipped
    assert second.empty_fraction == 1.0
    assert data.patch_statistics(
        _rows(), 100, (150, 150), min_cells=2
    ).patch.tolist() == ["r000c000"]


def test_hover_labels_are_escaped_and_open_away_from_the_figure_edges(monkeypatch):
    shown = []
    monkeypatch.setattr(figures, "display", shown.append)
    fig = plt.figure(figsize=(4, 4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 200), ylim=(200, 0))
    table = pd.DataFrame(
        {"x0": [0, 120], "y0": [0, 120], "x1": [100, 200], "y1": [100, 200]}
    )
    figures.hover_targets(fig, ax, table, ['r000c000 · "depth" & more', "r001c001"])
    page = shown[0].data
    assert page.count('class="hit ') == 2
    assert (
        'class="hit from-left below" title="r000c000 · &quot;depth&quot; &amp; more"'
        in page
    )
    assert 'class="hit from-right above" title="r001c001"' in page
    assert '<span class="tip">r000c000 · &quot;depth&quot; &amp; more</span>' in page


def test_save_view_writes_a_folder_per_user_with_provenance(monkeypatch, tmp_path):
    monkeypatch.setattr(data, "EXPORT_ROOT", tmp_path)
    monkeypatch.setattr(
        data, "training_config", lambda: {"model": {"trainable_blocks": "last_4"}}
    )
    state = {
        "dataset": "hest",
        "choice": "HEST · TENX116 (pancreas)",
        "directory": tmp_path / "results",
        "mapping_path": tmp_path / "mapping.yaml",
        "config": {"conformal": {"ontology": "cl.yaml"}},
        "ontology": SimpleNamespace(release="2026-06-08", sha256="abc"),
        "metrics": pd.DataFrame({"target_coverage": [0.9]}, index=["branch_graph"]),
    }
    directory = data.save_view(
        state,
        "branch_graph",
        "tissue map",
        {"map": plt.figure()},
        {"patches": pd.DataFrame({"patch": ["r000c000"]})},
        {"patch_size": 1024},
    )
    assert directory.parent == tmp_path / getpass.getuser() / "hest"
    assert sorted(p.name for p in directory.iterdir()) == [
        "map.png",
        "patches.csv",
        "provenance.json",
    ]
    provenance = json.loads((directory / "provenance.json").read_text())
    assert provenance["saved_by"] == getpass.getuser()
    assert provenance["view"] == "tissue map" and provenance["patch_size"] == 1024
    assert (
        provenance["nominal_coverage"] == 0.9
        and provenance["trainable_blocks"] == "last_4"
    )


def test_selected_cell_text_handles_empty_sets_and_cells_without_predictions():
    cell = pd.Series(
        {
            "number": 7,
            "row_id": "x-1",
            "label": "Tumor",
            "evaluated": True,
            "set_size": 0.0,
            "ancestor_covered": False,
            "ancestor": "Empty set",
            "ancestor_term": "",
            "truth": "tumor cell",
            "predicted": "t cell",
            "prediction_set": "∅",
            "depth": np.nan,
        }
    )
    text = figures.selected_cell_text(None, cell)
    assert text.startswith(
        "**Selected cell #7** (x-1, circled in magenta; original annotation: Tumor)"
    )
    assert text.endswith(
        "prediction set ∅. The prediction set is empty, so there is no ancestor and no depth."
    )
    unscored = cell.copy()
    unscored["evaluated"] = False
    assert figures.selected_cell_text(None, unscored).endswith(
        "has no saved test prediction."
    )


def _example():
    from cellbridge.inference.conformal.graph import build_task_graph
    from cellbridge.ontology import Ontology, OntologyMapping

    ontology = Ontology(
        release="test",
        source="test",
        sha256="test",
        root="root",
        names={"root": "Cell", "t": "T cell", "b": "B cell"},
        parents={"root": (), "t": ("root",), "b": ("root",)},
        children={"root": ("t", "b"), "t": (), "b": ()},
    )
    mapping = OntologyMapping(
        dataset="test",
        release="test",
        sha256="test",
        source_to_class={"T": "t", "B": "b"},
        class_to_term={"t": "t", "b": "b"},
        class_to_term_name={"t": "T cell", "b": "B cell"},
        excluded_classes=frozenset(),
    )
    saved = {
        "ids": np.array(["single", "multiple", "excluded", "empty"]),
        "classes": np.array(["t", "b"]),
        "y_true": np.array(["t", "t", "b", "t"]),
        "probabilities": np.array([[0.9, 0.1], [0.2, 0.8], [0.8, 0.2], [0.1, 0.9]]),
        "branch_graph_sets": np.array([[1, 0], [1, 1], [1, 0], [0, 0]], dtype=bool),
    }
    cells = pd.DataFrame(
        {
            "row_id": ["empty", "single", "unscored", "excluded", "multiple"],
            "x": [10.0, 20.0, 30.0, 40.0, 50.0],
            "y": [10.0, 20.0, 30.0, 40.0, 50.0],
            "label": ["T", "T", "B", "B", "T"],
            "evaluated": [True, True, False, True, True],
            "number": range(1, 6),
        }
    )
    state = {
        "choice": "HEST · test",
        "dataset": "hest",
        "section": "test",
        "saved": saved,
        "graph": build_task_graph(mapping, ontology, saved["classes"]),
        "mapping": mapping,
        "ontology": ontology,
        "methods": ["branch_graph"],
        "metrics": pd.DataFrame({"target_coverage": [0.9]}, index=["branch_graph"]),
        "fields": pd.DataFrame({"sample_id": ["test"], "evaluated_cells": [4]}),
        "sample": SimpleNamespace(cell_ids=cells.row_id.to_numpy()),
    }
    field = {
        "cells": cells,
        "tissue": np.full((80, 80, 3), [180, 90, 130], dtype=np.uint8),
        "sample_id": "test",
        "image_path": "test.tif",
        "mpp": 0.5,
        "outside_image_annotations": 0,
        "outside_image_evaluated": 0,
    }
    return state, field


def test_evaluate_field_aligns_ids_and_preserves_unscored_cells():
    state, field = _example()
    original = field["cells"].copy(deep=True)
    rows = data.evaluate_field(state, field, "branch_graph")
    assert rows.row_id.tolist() == original.row_id.tolist()
    pd.testing.assert_frame_equal(field["cells"], original)
    scored = rows.set_index("row_id")
    assert scored.loc["single", "depth"] == 0
    assert scored.loc["multiple", "depth"] == 1
    assert scored.loc["multiple", "ancestor"] == "Cell"
    assert scored.loc["multiple", "expanded_set"] == "t; b"
    assert scored.loc["excluded", "ancestor"] == "T cell"
    assert not scored.loc["excluded", "ancestor_covered"]
    assert pd.isna(scored.loc["excluded", "depth"])
    assert scored.loc["empty", "prediction_set"] == "∅"
    assert scored.loc["empty", "ancestor"] == "Empty set"
    assert pd.isna(scored.loc["empty", "depth"])
    assert pd.isna(scored.loc["unscored", "set_size"])


@pytest.mark.parametrize("smooth", [False, True])
def test_selection_preserves_table_bounds_and_depth_markers(monkeypatch, smooth):
    state, field = _example()
    monkeypatch.setattr(figures, "display", lambda _: None)
    result = figures.render_selection(
        state, field, "branch_graph", (0, 0, 80, 80), "single", smooth
    )
    assert result["bounds"] == [0, 0, 80, 80]
    assert result["cell_id"] == "single" and result["smooth"] == smooth
    assert result["cells"].row_id.tolist() == field["cells"].row_id.tolist()
    depth_ax = result["figure"].axes[3]
    offsets = [collection.get_offsets().tolist() for collection in depth_ax.collections]
    assert [[40.0, 40.0]] in offsets  # Excluded reference.
    assert [[10.0, 10.0]] in offsets  # Empty set.
    assert [[30.0, 30.0]] in offsets  # No prediction.
    assert depth_ax.collections[-1].get_edgecolors()[0].tolist() == [1.0, 0.0, 1.0, 1.0]


@pytest.mark.parametrize("colour", list(figures.TERRA_COLOURS))
def test_terra_preserves_patch_membership_and_export_columns(monkeypatch, colour):
    state, field = _example()
    rows = data.evaluate_field(state, field, "branch_graph")
    pool = rows.assign(in_patch=True, umap_x=np.arange(5), umap_y=np.arange(5) ** 2)
    pool.loc[2, "in_patch"] = False
    monkeypatch.setattr(figures, "display", lambda _: None)
    monkeypatch.setattr(figures, "terra_matrix", lambda *_: np.ones((5, 3)))
    monkeypatch.setattr(figures, "terra_pool", lambda *_: (pool.copy(), np.eye(5)))
    result = figures.render_terra(
        state,
        field,
        rows,
        (0, 0, 80, 80),
        "branch_graph",
        "Cell embedding",
        colour,
        1,
    )
    assert result["cells"].row_id.tolist() == [
        "empty",
        "single",
        "excluded",
        "multiple",
    ]
    assert result["context"] == 1
    assert result["colour"] == figures.TERRA_COLOURS[colour][0]
    assert result["embedding"] == "cell_emb"
    assert {"umap_x", "umap_y", "neighbour_agreement"} <= set(result["cells"])
    assert result["cells"].neighbour_agreement.tolist() == pytest.approx(
        [2 / 3, 2 / 3, 0, 2 / 3]
    )


def test_explorer_controls_and_save_use_the_displayed_selection(monkeypatch):
    pytest.importorskip("ipywidgets")
    from cellbridge.qualitative import views

    state, field = _example()
    catalog = pd.DataFrame(
        {"dataset": ["hest"], "section": ["test"]}, index=[state["choice"]]
    )
    monkeypatch.setattr(views, "load_dataset", lambda *_: state)
    monkeypatch.setattr(views, "load_field", lambda *_: field)
    monkeypatch.setattr(views, "display", lambda _: None)
    monkeypatch.setattr(figures, "display", lambda _: None)
    exported = []
    monkeypatch.setattr(views, "save_view", lambda *args: exported.append(args))
    explorer = views.Explorer(catalog).show()
    assert explorer.method_widget.options == (("Branch graph", "branch_graph"),)
    assert explorer.size_widget.value == 512
    assert not explorer.save_button.disabled
    shown = explorer.last
    explorer.mode_widget.value = "Cell"
    explorer.size_widget.value = 224
    explorer.save_button.click()
    assert exported[0][3]["view"] is shown["figure"]
    assert exported[0][5]["bounds"] == shown["bounds"]
    explorer.show_button.click()
    assert explorer.last["cell_id"] is not None
    assert explorer.last is not shown


def test_tissue_and_terra_controls_keep_defaults_and_callbacks(monkeypatch):
    pytest.importorskip("ipywidgets")
    from cellbridge.qualitative import views

    state, _ = _example()
    catalog = pd.DataFrame({"dataset": ["hest"]}, index=[state["choice"]])
    calls = []
    monkeypatch.setattr(
        views.TissueMap, "on_tissue_show", lambda *_: calls.append("map")
    )
    monkeypatch.setattr(
        views.TerraView, "on_terra_fetch", lambda *_: calls.append("terra")
    )
    tissue = views.TissueMap(catalog)
    terra = views.TerraView(tissue)
    assert tissue.tissue_size_widget.value == 1024
    assert tissue.min_cells_widget.value == 25
    assert tissue.tissue_method_widget.value == "branch_graph"
    assert tissue.patch_button.disabled and tissue.save_button.disabled
    assert terra.terra_embedding_widget.value == "Cell embedding"
    assert terra.terra_colour_widget.value == "Ontology depth / coarseness"
    assert terra.terra_context_widget.value == 5000
    assert terra.terra_button.layout.width == "220px"
    tissue.tissue_button.click()
    terra.terra_button.click()
    assert calls == ["map", "terra"]


def test_catalog_keeps_dataset_order_organs_and_section_filter(monkeypatch, tmp_path):
    monkeypatch.setattr(data, "MODEL_RESULTS", tmp_path)
    monkeypatch.setattr(data, "training_config", dict)
    cohort = tmp_path / "cohort.csv"
    pd.DataFrame({"sample_id": ["TENX116"], "organ": ["lymph_node"]}).to_csv(
        cohort, index=False
    )
    monkeypatch.setattr(data, "load_config", lambda _: {"data": {"cohort": cohort}})
    for dataset, section in [
        ("nucls", None),
        ("phenocell", None),
        ("hest", "TENX116"),
        ("hest", "TENX143"),
        ("hest", "unknown"),
    ]:
        directory = data.prediction_directory(dataset, section)
        directory.mkdir(parents=True)
        (directory / "predictions.npz").touch()
    catalog = data.load_catalog()
    assert catalog.index.tolist() == [
        "NuCLS (breast)",
        "PhenoCell (colorectal)",
        "HEST · TENX116 (lymph node)",
        "HEST · unknown (organ not recorded)",
    ]
    assert catalog.dataset.tolist() == ["nucls", "phenocell", "hest", "hest"]


def test_load_dataset_checks_metrics_and_summarizes_fields(monkeypatch, tmp_path):
    state, field = _example()
    np.savez(tmp_path / "predictions.npz", **state["saved"])
    metrics = pd.DataFrame(
        {
            "conformal_method": ["branch_graph"],
            "method": ["cellbridge"],
            "arm": ["lora_adapted"],
            "task": ["all"],
            "probe": ["linear"],
            "marginal_coverage": [0.5],
            "mean_prediction_set_size": [1.0],
            "singleton_rate": [0.5],
        }
    )
    metrics.to_csv(tmp_path / "metrics.csv", index=False)
    metadata = field["cells"][["row_id", "x", "y", "label"]].assign(
        sample_id="test", split="test"
    )
    monkeypatch.setattr(
        data,
        "load_config",
        lambda _: {
            "data": {},
            "conformal": {"mapping": "mapping.yaml", "ontology": "ontology.yaml"},
        },
    )
    monkeypatch.setattr(data, "load_mapping", lambda _: state["mapping"])
    monkeypatch.setattr(data, "load_ontology", lambda _: state["ontology"])
    monkeypatch.setattr(data.nucls, "load_metadata", lambda _: metadata.copy())
    catalog = pd.DataFrame(
        {"dataset": ["nucls"], "section": [None], "directory": [tmp_path]},
        index=["test"],
    )
    loaded = data.load_dataset(catalog, "test")
    assert loaded["evaluated"].row_id.tolist() == state["saved"]["ids"].tolist()
    assert loaded["fields"].to_dict("records") == [
        {
            "sample_id": "test",
            "evaluated_cells": 4,
            "annotated_cells": 5,
            "top1_accuracy": 0.25,
        }
    ]
    metrics.loc[0, "marginal_coverage"] = 1.0
    metrics.to_csv(tmp_path / "metrics.csv", index=False)
    with pytest.raises(AssertionError):
        data.load_dataset(catalog, "test")
