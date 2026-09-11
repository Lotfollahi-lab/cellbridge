"""Notebook 09 helpers: patch statistics, tissue-map hover labels and per-user saving."""

import getpass
import json
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from cellbridge.qualitative import data, figures  # noqa: E402


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
    state = dict(
        dataset="hest",
        choice="HEST · TENX116 (pancreas)",
        directory=tmp_path / "results",
        mapping_path=tmp_path / "mapping.yaml",
        config={"conformal": {"ontology": "cl.yaml"}},
        ontology=SimpleNamespace(release="2026-06-08", sha256="abc"),
        metrics=pd.DataFrame({"target_coverage": [0.9]}, index=["branch_graph"]),
    )
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
