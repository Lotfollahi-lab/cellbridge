"""The notebook 09 widgets: the explorer, the HEST tissue map and the TERRA cross-check."""

from contextlib import contextmanager

import ipywidgets as widgets
import matplotlib.pyplot as plt
from IPython.display import Markdown, clear_output, display

from cellbridge.inference.embed.sources import hest
from cellbridge.qualitative.data import (
    METHODS,
    MODEL_RESULTS,
    OFFERED_METHODS,
    cells_in_bounds,
    evaluate_field,
    load_catalog,
    load_dataset,
    load_field,
    patch_choices,
    patch_statistics,
    save_view,
    terra_file,
)
from cellbridge.qualitative.figures import (
    TERRA_COLOURS,
    TERRA_MATRICES,
    TERRA_NEIGHBOURS,
    TISSUE_STATISTICS,
    render_section,
    render_selection,
    render_terra,
    summary_line,
)

plt.rcParams.update(
    {"text.usetex": False, "font.family": "DejaVu Sans", "font.size": 10}
)


@contextmanager
def _output(output):
    with output:
        clear_output(wait=True)
        yield


def _dropdown(description, options=(), *, width=None, **kwargs):
    if width is not None:
        kwargs["layout"] = widgets.Layout(width=width)
    return widgets.Dropdown(description=description, options=options, **kwargs)


def _button(description, callback, *, width=None, **kwargs):
    if width is not None:
        kwargs["layout"] = widgets.Layout(width=width)
    button = widgets.Button(description=description, **kwargs)
    button.on_click(callback)
    return button


def _method_options(state):
    offered = [method for method in OFFERED_METHODS if method in state["methods"]]
    assert offered, f"{state['choice']} has no saved results for {OFFERED_METHODS}"
    return [(METHODS[method], method) for method in offered]


def show_catalog():
    """Load every dataset and HEST section with saved cell-level predictions."""
    catalog = load_catalog()
    others = ", ".join(catalog.index[catalog.dataset.ne("hest")])
    print("CellBridge results:", MODEL_RESULTS)
    print(
        f"{len(catalog)} entries with saved cell-level predictions: {others} and "
        f"{int(catalog.dataset.eq('hest').sum())} HEST sections."
    )
    return catalog


class Explorer:
    """Browse one field of a dataset by patch or by cell."""

    def __init__(self, catalog):
        self.catalog = catalog
        self.dataset_widget = _dropdown("Dataset:", list(catalog.index), width="450px")
        self.load_button = _button("Load dataset", self.on_load, button_style="primary")
        self.field_widget = _dropdown("Field:", width="95%")
        self.patch_widget = _dropdown("Patch:", width="650px")
        self.mode_widget = widgets.ToggleButtons(
            options=["Patch", "Cell"], description="View:"
        )
        self.size_widget = _dropdown("Width (px):", [224, 512, 1024], value=512)
        self.whole_field_widget = widgets.Checkbox(
            value=False, description="Cells from whole field"
        )
        self.cell_search = widgets.Text(
            description="Find cell:",
            placeholder="Optional row ID or original annotation",
            layout=widgets.Layout(width="650px"),
        )
        self.cell_count = widgets.Label()
        self.cell_widget = widgets.Combobox(
            options=[],
            ensure_option=True,
            description="Cell:",
            layout=widgets.Layout(width="95%"),
        )
        self.method_widget = _dropdown("Method:", width="350px")
        self.smooth_widget = widgets.Checkbox(
            value=False, description="Smooth depth map"
        )
        self.show_button = _button(
            "Show selection", self.on_show, button_style="primary", disabled=True
        )
        self.save_button = _button("Save displayed view", self.on_save, disabled=True)
        self.status_output, self.catalog_output, self.plot_output = (
            widgets.Output(),
            widgets.Output(),
            widgets.Output(),
        )
        self.state = self.field = self.last = None
        self.cell_ids = {}
        self.field_widget.observe(self.change_field, names="value")
        self.size_widget.observe(self.update_patches, names="value")
        self.patch_widget.observe(self.update_cells, names="value")
        self.whole_field_widget.observe(self.update_cells, names="value")
        self.cell_search.observe(self.update_cells, names="value")
        self.box = widgets.VBox(
            [
                widgets.HBox([self.dataset_widget, self.load_button]),
                self.status_output,
                self.catalog_output,
                self.field_widget,
                widgets.HBox([self.size_widget, self.mode_widget]),
                self.patch_widget,
                self.whole_field_widget,
                self.cell_search,
                self.cell_count,
                self.cell_widget,
                widgets.HBox([self.method_widget, self.smooth_widget]),
                widgets.HBox([self.show_button, self.save_button]),
                self.plot_output,
            ]
        )

    def show(self):
        display(self.box)
        self.on_load()
        self.on_show()
        return self

    def update_cells(self, *_):
        field = self.field
        if field is None or self.patch_widget.value is None:
            return
        rows = (
            field["cells"]
            if self.whole_field_widget.value
            else cells_in_bounds(field["cells"], self.patch_widget.value)
        )
        query = self.cell_search.value.strip().lower()
        if query:
            rows = rows.loc[
                rows.row_id.str.lower().str.contains(query, regex=False)
                | rows.label.str.lower().str.contains(query, regex=False)
            ]
        self.cell_count.value = f"{len(rows):,} matching cells; showing up to 500 suggestions. Refine Find cell if needed."
        rows = rows.head(500)
        self.cell_ids = {
            f"#{r.number} | {r.row_id} | {r.label}"
            + ("" if r.evaluated else " [no test prediction]"): r.row_id
            for r in rows.itertuples()
        }
        self.cell_widget.options = list(self.cell_ids)
        evaluated_ids = set(rows.loc[rows.evaluated, "row_id"])
        self.cell_widget.value = next(
            (label for label, uid in self.cell_ids.items() if uid in evaluated_ids),
            self.cell_widget.options[0] if self.cell_widget.options else "",
        )

    def update_patches(self, *_):
        if self.field is not None:
            self.patch_widget.options = patch_choices(
                self.field, self.size_widget.value
            )
            self.patch_widget.value = self.patch_widget.options[0][1]
            self.update_cells()

    def change_field(self, *_):
        state = self.state
        if state is None or self.field_widget.value is None:
            return
        self.show_button.disabled = True
        with _output(self.status_output):
            print("Loading H&E and cell coordinates…")
            old = self.field
            self.field = None
            if old is not None and hasattr(old["tissue"], "store"):
                hest.close_wsi(old["tissue"])
            self.field = load_field(state, self.field_widget.value)
            self.update_patches()
            self.show_button.disabled = False
            print("Ready. Choose a patch/cell and click Show selection.")

    def on_load(self, _=None):
        self.show_button.disabled = self.save_button.disabled = True
        with _output(self.status_output):
            print("Loading saved predictions and checking alignment…")
            state = load_dataset(self.catalog, self.dataset_widget.value)
            self.last = None
            self.state = state
            self.method_widget.options = _method_options(state)
            self.method_widget.value = self.method_widget.options[0][1]
            # Replace options without triggering an intermediate field load.
            self.field_widget.unobserve(self.change_field, names="value")
            self.field_widget.options = []
            self.field_widget.options = [
                (f"{r.sample_id} · {r.evaluated_cells:,} evaluated cells", r.sample_id)
                for r in state["fields"].itertuples()
            ]
            self.field_widget.value = self.field_widget.options[0][1]
            self.field_widget.observe(self.change_field, names="value")
            self.change_field()
        with _output(self.catalog_output):
            display(
                Markdown(
                    f"**Loaded {state['choice']} · CellBridge**. All saved method metrics "
                    "checked; fields are listed by evaluated cells, not by performance."
                )
            )
        with self.plot_output:
            clear_output()

    def on_show(self, _=None):
        with _output(self.plot_output):
            self.save_button.disabled = True
            state, field = self.state, self.field
            cell_id, bounds = None, self.patch_widget.value
            if self.mode_widget.value == "Cell":
                cell_id = self.cell_ids.get(self.cell_widget.value)
                if cell_id is None:
                    print("Select a cell from the suggestions first.")
                    return
                row = field["cells"].set_index("row_id").loc[cell_id]
                height, width = field["tissue"].shape[:2]
                size = self.size_widget.value
                x0, y0 = (
                    max(0, round(row.x) - size // 2),
                    max(0, round(row.y) - size // 2),
                )
                bounds = (x0, y0, min(width, x0 + size), min(height, y0 + size))
            self.last = render_selection(
                state,
                field,
                self.method_widget.value,
                bounds,
                cell_id,
                self.smooth_widget.value,
            )
            self.save_button.disabled = False

    def on_save(self, _=None):
        if self.last is None:
            return
        details = {k: v for k, v in self.last.items() if k not in {"figure", "cells"}}
        directory = save_view(
            self.state,
            self.last["method"],
            "explorer",
            {"view": self.last["figure"]},
            {"cells": self.last["cells"]},
            details,
        )
        with _output(self.status_output):
            print("Saved displayed view, cell table and provenance to", directory)


class TissueMap:
    """Paint every grid patch of a HEST section by one statistic and open any patch."""

    def __init__(self, catalog):
        self.catalog = catalog
        choices = catalog.index[catalog.dataset.eq("hest")].tolist()
        assert choices, (
            f"No HEST sections with saved cell-level predictions under {MODEL_RESULTS}"
        )
        self.section_widget = _dropdown("Section:", choices, width="450px")
        self.statistic_widget = _dropdown(
            "Statistic:", list(TISSUE_STATISTICS), width="450px"
        )
        self.tissue_size_widget = _dropdown(
            "Patch (px):", [512, 1024, 2048], value=1024
        )
        self.min_cells_widget = _dropdown("Min cells:", [1, 10, 25, 50, 100], value=25)
        self.tissue_method_widget = _dropdown(
            "Method:",
            [(METHODS[m], m) for m in OFFERED_METHODS],
            value=OFFERED_METHODS[0],
            width="350px",
        )
        self.tissue_smooth_widget = widgets.Checkbox(
            value=False, description="Smooth depth map"
        )
        self.tissue_button = _button(
            "Show tissue map", self.on_tissue_show, button_style="primary"
        )
        self.tissue_patch_widget = _dropdown("Patch:", [], width="95%")
        self.patch_button = _button(
            "Open selected patch", self.on_patch_open, disabled=True
        )
        self.save_button = _button("Save displayed view", self.on_save, disabled=True)
        self.tissue_status, self.tissue_output, self.patch_output = (
            widgets.Output(),
            widgets.Output(),
            widgets.Output(),
        )
        self.choice = self.state = self.field = self.map = self.opened = None
        self.rows, self.patches = {}, {}
        self.box = widgets.VBox(
            [
                widgets.HBox([self.section_widget, self.tissue_button]),
                widgets.HBox(
                    [
                        self.statistic_widget,
                        self.tissue_size_widget,
                        self.min_cells_widget,
                    ]
                ),
                widgets.HBox([self.tissue_method_widget, self.tissue_smooth_widget]),
                self.tissue_status,
                self.tissue_output,
                self.tissue_patch_widget,
                widgets.HBox([self.patch_button, self.save_button]),
                self.patch_output,
            ]
        )

    def show(self):
        display(self.box)
        self.on_tissue_show()
        return self

    def section_data(self, choice):
        """Load one HEST section, keeping only the current whole-slide image open."""
        if self.choice != choice:
            if self.field is not None:
                hest.close_wsi(self.field["tissue"])
            self.choice = self.state = self.field = None
            self.rows, self.patches = {}, {}
            state = load_dataset(self.catalog, choice)
            self.choice, self.state = choice, state
            self.field = load_field(state, self.catalog.loc[choice, "section"])
        return self.state, self.field

    def on_tissue_show(self, _=None):
        self.patch_button.disabled = self.save_button.disabled = True
        self.map = self.opened = None
        size, min_cells, statistic = (
            self.tissue_size_widget.value,
            self.min_cells_widget.value,
            self.statistic_widget.value,
        )
        with _output(self.tissue_status):
            print("Loading the section and scoring its cells…")
            state, field = self.section_data(self.section_widget.value)
            self.tissue_method_widget.options = _method_options(state)
            method = self.tissue_method_widget.value
            if method not in self.rows:
                self.rows[method] = evaluate_field(state, field, method)
            rows = self.rows[method]
            column, higher_is_better, _ = TISSUE_STATISTICS[statistic]
            table = patch_statistics(rows, size, field["tissue"].shape[:2], min_cells)
            table = table.sort_values(
                column, ascending=higher_is_better, na_position="last"
            )
            self.patches = {
                f"{r.patch} | x={r.x0}, y={r.y0} | {r.evaluated_cells:,} cells | "
                f"{statistic}: {getattr(r, column):.2f}": (r.x0, r.y0, r.x1, r.y1)
                for r in table.itertuples()
            }
            self.tissue_patch_widget.options = list(self.patches)
            if self.tissue_patch_widget.options:
                self.tissue_patch_widget.value = self.tissue_patch_widget.options[0]
                self.patch_button.disabled = False
            print("Ready. Patches are listed worst first; pick one and open it.")
        with self.patch_output:
            clear_output()
        with _output(self.tissue_output):
            evaluated = rows.loc[rows.evaluated]
            nominal = float(state["metrics"].loc[method, "target_coverage"])
            display(
                Markdown(
                    f"**Whole section: {len(evaluated):,} evaluated cells** among "
                    f"{len(rows):,} annotated cells. Saved nominal coverage: {nominal:.0%}."
                    f"\n\n{summary_line(evaluated)}"
                )
            )
            if table.empty:
                display(
                    Markdown(
                        f"**No patch holds {min_cells} evaluated cells at {size} px.** "
                        "Lower either control and show the map again."
                    )
                )
                return
            figure = render_section(
                state, field, table, size, statistic, method, min_cells
            )
            self.map = {
                "figure": figure,
                "table": table.drop(columns=["row", "column"]),
                "method": method,
                "statistic": statistic,
                "patch_size": size,
                "min_cells": min_cells,
            }
            self.save_button.disabled = False
            display(
                Markdown(
                    "Hover any patch on the map for its name, bounds and value; the **Patch** "
                    f"menu below lists the same patches, worst first by {statistic}."
                )
            )

    def on_patch_open(self, _=None):
        with _output(self.patch_output):
            bounds = self.patches.get(self.tissue_patch_widget.value)
            if bounds is None:
                print("Show a tissue map first, then choose a patch.")
                return
            self.opened = render_selection(
                self.state,
                self.field,
                self.tissue_method_widget.value,
                bounds,
                None,
                self.tissue_smooth_widget.value,
            )

    def on_save(self, _=None):
        if self.map is None:
            return
        figures, tables = {"map": self.map["figure"]}, {"patches": self.map["table"]}
        details = {
            "method": self.map["method"],
            "sample_id": self.field["sample_id"],
            "image_path": self.field["image_path"],
            "mpp": self.field["mpp"],
            "statistic": self.map["statistic"],
            "patch_size": self.map["patch_size"],
            "min_cells": self.map["min_cells"],
            "painted_patches": len(self.map["table"]),
        }
        if self.opened is not None:
            figures["view"], tables["cells"] = (
                self.opened["figure"],
                self.opened["cells"],
            )
            details.update(bounds=self.opened["bounds"], smooth=self.opened["smooth"])
        directory = save_view(
            self.state, self.map["method"], "tissue map", figures, tables, details
        )
        with _output(self.tissue_status):
            opened = ", the opened patch" if self.opened is not None else ""
            print(
                f"Saved the tissue map{opened}, their tables and provenance to",
                directory,
            )


class TerraView:
    """Project the patch chosen on a tissue map into its saved TERRA embeddings."""

    def __init__(self, tissue_map):
        self.tissue_map = tissue_map
        self.terra_embedding_widget = _dropdown(
            "Embedding:", list(TERRA_MATRICES), width="420px"
        )
        self.terra_colour_widget = _dropdown(
            "Colour by:",
            list(TERRA_COLOURS),
            value="Ontology depth / coarseness",
            width="420px",
        )
        self.terra_context_widget = _dropdown(
            "Context:",
            [
                ("Patch only", 0),
                ("5,000 cells", 5000),
                ("20,000 cells", 20000),
                ("50,000 cells", 50000),
            ],
            value=5000,
        )
        self.terra_button = _button(
            "Fetch TERRA embeddings",
            self.on_terra_fetch,
            button_style="primary",
            width="220px",
        )
        self.save_button = _button("Save displayed view", self.on_save, disabled=True)
        self.terra_status, self.terra_output = widgets.Output(), widgets.Output()
        self.last = None
        self.box = widgets.VBox(
            [
                widgets.HBox([self.terra_embedding_widget, self.terra_colour_widget]),
                widgets.HBox(
                    [self.terra_context_widget, self.terra_button, self.save_button]
                ),
                self.terra_status,
                self.terra_output,
            ]
        )

    def show(self):
        display(self.box)
        print(
            "Choose a patch in the tissue section above, then click Fetch TERRA embeddings."
        )
        return self

    def on_terra_fetch(self, _=None):
        tissue = self.tissue_map
        state, field = tissue.state, tissue.field
        bounds = tissue.patches.get(tissue.tissue_patch_widget.value)
        with _output(self.terra_status):
            if state is None or field is None or bounds is None:
                print("Show a tissue map above and choose a patch first.")
                return
            assert state["dataset"] == "hest", (
                "TERRA embeddings are saved for HEST sections only"
            )
            print(
                f"Reading the TERRA {self.terra_embedding_widget.value.lower()} of {state['choice']} and "
                "projecting; a large context takes a few minutes…"
            )
        self.save_button.disabled = True
        self.last = None
        method = tissue.tissue_method_widget.value
        if method not in tissue.rows:
            tissue.rows[method] = evaluate_field(state, field, method)
        with _output(self.terra_output):
            shown = render_terra(
                state,
                field,
                tissue.rows[method],
                bounds,
                method,
                self.terra_embedding_widget.value,
                self.terra_colour_widget.value,
                self.terra_context_widget.value,
            )
            self.last = dict(
                shown,
                state=state,
                sample_id=field["sample_id"],
                patch_name=tissue.tissue_patch_widget.value.split(" | ")[0],
            )
            self.save_button.disabled = False
        with _output(self.terra_status):
            print(
                f"Projected patch {self.last['patch_name']} · "
                f"{self.terra_embedding_widget.value.lower()} · "
                f"{self.last['context']:,} context cells."
            )

    def on_save(self, _=None):
        if self.last is None:
            return
        last = self.last
        details = {
            "method": last["method"],
            "sample_id": last["sample_id"],
            "patch": last["patch_name"],
            "bounds": last["bounds"],
            "embedding": last["embedding"],
            "colour": last["colour"],
            "context_cells": last["context"],
            "neighbours": TERRA_NEIGHBOURS,
            "terra_file": str(terra_file(last["state"])),
        }
        directory = save_view(
            last["state"],
            last["method"],
            "TERRA",
            {"terra": last["figure"]},
            {"cells": last["cells"]},
            details,
        )
        with _output(self.terra_status):
            print("Saved the TERRA view, its cell table and provenance to", directory)
