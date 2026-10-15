"""Load, check and score the saved CellBridge conformal predictions shown in notebook 07."""

import getpass
import json
import os
import pickle
from collections import deque
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from cellbridge.config import _defaults, config_root, load_config
from cellbridge.inference.conformal.graph import build_task_graph
from cellbridge.inference.embed.sources import hest, nucls
from cellbridge.inference.embed.sources import lmdb as tiled
from cellbridge.ontology import load_mapping, load_ontology

REPO = config_root()
OUTPUTS_ROOT = Path(
    os.environ.get("CELLBRIDGE_OUTPUTS_ROOT", _defaults()["CELLBRIDGE_OUTPUTS_ROOT"])
)
MODEL_RESULTS = Path(
    os.environ.get(
        "CELLBRIDGE_QUALITATIVE_RESULTS",
        OUTPUTS_ROOT
        / "reproducibility/inference/cellbridge/uni_terra_four_layers_adapted_all_cells",
    )
)
TRAIN_CONFIG = Path(
    os.environ.get(
        "CELLBRIDGE_TRAIN_CONFIG",
        OUTPUTS_ROOT
        / "reproducibility/train/uni_terra_four_layers_adapted_all_cells/resolved_config.yaml",
    )
)
EXPORT_ROOT = OUTPUTS_ROOT / "visualizations/res_figures/qualitative_conformal"

METHODS = {
    "split": "Global split",
    "graph": "Global graph",
    "branch_graph": "Branch graph",
}
# Paper-facing views use branch graph; saved paper methods are validated on load.
OFFERED_METHODS = ["branch_graph"]
DATASET_ORGANS = {"nucls": "breast", "phenocell": "colorectal"}


def training_config():
    """Read optional training provenance; prediction review does not require it."""
    return load_config(TRAIN_CONFIG) if TRAIN_CONFIG.is_file() else {}


def prediction_directory(dataset, section=None, *, results_root=MODEL_RESULTS):
    base = Path(results_root) / dataset
    suffix = Path("evaluation/lora_adapted/all/linear")
    if section:
        per_section = base / section / suffix / section / "conformal_prediction"
        if per_section.is_dir():
            return per_section
        return base / suffix / section / "conformal_prediction"
    return base / suffix / "conformal_prediction"


def load_catalog(results_root=MODEL_RESULTS):
    """Discover saved OOD and HEST predictions, including both CLI output layouts."""
    results_root = Path(results_root)
    catalog = []
    for dataset, label in [("nucls", "NuCLS"), ("phenocell", "PhenoCell")]:
        directory = prediction_directory(dataset, results_root=results_root)
        if (directory / "predictions.npz").is_file():
            organ = DATASET_ORGANS[dataset]
            catalog.append(
                dict(
                    choice=f"{label} ({organ})",
                    dataset=dataset,
                    section=None,
                    organ=organ,
                    directory=directory,
                )
            )
    paths = sorted(
        (results_root / "hest").glob(
            "**/evaluation/lora_adapted/all/linear/*/conformal_prediction/predictions.npz"
        )
    )
    if paths:
        cohort_path = Path(
            load_config(REPO / "configs/inference/hest.yaml")["data"]["cohort"]
        )
        cohort = (
            pd.read_csv(cohort_path, dtype={"sample_id": str})
            if cohort_path.is_file()
            else None
        )
        organs = (
            dict(zip(cohort.sample_id, cohort.organ.str.replace("_", " ")))
            if cohort is not None and "organ" in cohort
            else {}
        )
        for section in sorted(
            {path.parent.parent.name for path in paths} - {"TENX143"}
        ):
            directory = prediction_directory("hest", section, results_root=results_root)
            if (directory / "predictions.npz").is_file():
                organ = organs.get(section, "organ not recorded")
                catalog.append(
                    dict(
                        choice=f"HEST · {section} ({organ})",
                        dataset="hest",
                        section=section,
                        organ=organ,
                        directory=directory,
                    )
                )
    if not catalog:
        raise FileNotFoundError(
            f"No CellBridge conformal predictions under {results_root}"
        )
    return pd.DataFrame(catalog).set_index("choice")


def load_dataset(catalog, choice):
    spec = catalog.loc[choice]
    dataset, section, directory = spec.dataset, spec.section, spec.directory
    config = load_config(REPO / f"configs/inference/{dataset}.yaml")
    data = config["data"]
    mapping_path = (
        Path(data["mapping_dir"]) / f"{section}.yaml"
        if dataset == "hest"
        else Path(config["conformal"]["mapping"])
    )
    mapping = load_mapping(mapping_path)
    ontology = load_ontology(config["conformal"]["ontology"])
    with np.load(directory / "predictions.npz", allow_pickle=False) as archive:
        saved = {key: archive[key] for key in archive.files}
    ids, classes = saved["ids"].astype(str), saved["classes"].astype(str)
    assert pd.Index(ids).is_unique and pd.Index(classes).is_unique
    available = [m for m in METHODS if m + "_sets" in saved]
    metrics = pd.read_csv(directory / "metrics.csv").set_index(
        "conformal_method", verify_integrity=True
    )
    assert metrics["method"].eq("cellbridge").all()
    assert metrics["arm"].eq("lora_adapted").all()
    assert metrics["task"].eq("all").all() and metrics["probe"].eq("linear").all()
    sample = None
    if dataset == "hest":
        sample = hest.load_sample(data["manifest"], data["root"], section)
        labels_path, _ = hest.evaluation_paths(data, section)
        labels = pd.read_csv(labels_path, dtype={"cell_id": str}).rename(
            columns={"group": "label"}
        )
        metadata = pd.DataFrame(
            {
                "row_id": sample.cell_ids.astype(str),
                "sample_id": section,
                "x": sample.centers[:, 0],
                "y": sample.centers[:, 1],
            }
        )
        metadata = metadata.merge(
            labels[["cell_id", "label"]],
            left_on="row_id",
            right_on="cell_id",
            how="left",
            validate="one_to_one",
        )
        metadata["label"] = metadata["label"].fillna("No reference annotation")
    else:
        # Include all available source annotations in overlays, even excluded classes.
        source_data = dict(data, exclude_labels=[])
        metadata = (nucls if dataset == "nucls" else tiled).load_metadata(source_data)
    metadata["row_id"] = metadata["row_id"].astype(str)
    lookup = metadata.set_index("row_id", drop=False, verify_integrity=True)
    evaluated = lookup.loc[ids].copy()
    if dataset != "hest":
        assert evaluated["split"].eq("test").all()
    truth, included = mapping.map_labels(evaluated["label"].astype(str).to_numpy())
    assert included.all()
    np.testing.assert_array_equal(truth, saved["y_true"].astype(str))
    columns = pd.Index(classes).get_indexer(truth)
    assert (columns >= 0).all()
    probabilities = saved["probabilities"]
    assert (
        probabilities.shape == (len(ids), len(classes))
        and np.isfinite(probabilities).all()
    )
    np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-5)
    for method in available:
        masks = saved[method + "_sets"]
        assert masks.dtype == bool and masks.shape == probabilities.shape
        actual = [
            masks[np.arange(len(ids)), columns].mean(),
            masks.sum(axis=1).mean(),
            (masks.sum(axis=1) == 1).mean(),
        ]
        expected = metrics.loc[
            method, ["marginal_coverage", "mean_prediction_set_size", "singleton_rate"]
        ]
        np.testing.assert_allclose(actual, expected.to_numpy(float))
    evaluated["truth"] = truth
    evaluated["predicted"] = classes[probabilities.argmax(axis=1)]
    graph = build_task_graph(mapping, ontology, classes)
    state = dict(
        choice=choice,
        dataset=dataset,
        section=section,
        config=config,
        mapping=mapping,
        mapping_path=mapping_path,
        ontology=ontology,
        graph=graph,
        saved=saved,
        metadata=metadata,
        evaluated=evaluated,
        methods=available,
        metrics=metrics,
        directory=directory,
        sample=sample,
    )
    counts = evaluated.groupby("sample_id").size().rename("evaluated_cells").to_frame()
    counts["annotated_cells"] = metadata.groupby("sample_id").size()
    counts["top1_accuracy"] = evaluated.groupby("sample_id").apply(
        lambda rows: rows.truth.eq(rows.predicted).mean()
    )
    state["fields"] = counts.reset_index().sort_values(
        ["evaluated_cells", "sample_id"], ascending=[False, True]
    )
    return state


def load_field(state, sample_id):
    data = state["config"]["data"]
    cells = state["metadata"].loc[state["metadata"].sample_id.eq(sample_id)].copy()
    if state["dataset"] == "nucls":
        names = cells.image_name.unique()
        assert len(names) == 1
        path = Path(data["images"]) / names[0]
        with Image.open(path) as image:
            tissue = np.asarray(image.convert("RGB"))
        mpp = data["source_mpp"]
    elif state["dataset"] == "phenocell":
        import lmdb

        print(
            "PhenoCell: reading original tiles; the first load can take several minutes."
        )
        # This dataset stores a fixed block of consecutive keys for every field.
        n_tiles = int(
            np.prod(np.ceil(np.asarray(data["image_shape"]) / data["tile_size"]))
        )
        with lmdb.open(
            str(data["lmdb"]), readonly=True, lock=False, readahead=True
        ) as env:
            assert (
                env.stat()["entries"] == state["metadata"].sample_id.nunique() * n_tiles
            )
            entries = None
            # Metadata order supplies a fast candidate; verify the stored name.
            # Fall back to scanning block headers if that order ever changes.
            sample_order = state["metadata"].sample_id.drop_duplicates().tolist()
            guess = sample_order.index(sample_id) * n_tiles
            starts = dict.fromkeys([guess, *range(0, env.stat()["entries"], n_tiles)])
            with env.begin() as txn:
                for start in starts:
                    first = pickle.loads(txn.get(f"{start:08d}".encode()))
                    if str(first["sample_name"]) != sample_id:
                        continue
                    entries = {}
                    for key in range(start, start + n_tiles):
                        item = pickle.loads(txn.get(f"{key:08d}".encode()))
                        assert str(item["sample_name"]) == sample_id
                        index = tiled._tile_index(item["tile_name"])
                        assert index not in entries
                        entries[index] = item
                    break
        if entries is None:
            raise ValueError(f"Field {sample_id} is absent from the PhenoCell LMDB")
        sample = tiled._load_sample(
            sample_id,
            entries,
            cells,
            image_shape=data["image_shape"],
            tile_size=data["tile_size"],
            value_scale=data["value_scale"],
            skip_missing=False,
        )
        tissue, cells, mpp = sample.image, sample.cells, data["source_mpp"]
        path = Path(data["lmdb"])
    else:
        sample = state["sample"]
        tissue, mpp, path = (
            hest.open_wsi(sample.wsi_path),
            sample.pixel_size_um,
            sample.wsi_path,
        )
    height, width = tissue.shape[:2]
    inside = (cells.x >= 0) & (cells.x < width) & (cells.y >= 0) & (cells.y < height)
    omitted = int((~inside).sum())
    omitted_evaluated = int(
        cells.loc[~inside, "row_id"].isin(state["evaluated"].index).sum()
    )
    if state["dataset"] != "hest":
        assert inside.all(), "Source coordinates lie outside the reconstructed image"
    cells = cells.loc[inside].copy()
    if omitted:
        print(
            f"Omitted {omitted} annotations outside H&E bounds ({omitted_evaluated} evaluated)."
        )
    cells = cells.sort_values("row_id").reset_index(drop=True)
    cells["number"] = np.arange(1, len(cells) + 1)
    cells["evaluated"] = cells.row_id.isin(state["evaluated"].index)
    return dict(
        sample_id=sample_id,
        tissue=tissue,
        cells=cells,
        mpp=float(mpp),
        image_path=str(path),
        outside_image_annotations=omitted,
        outside_image_evaluated=omitted_evaluated,
    )


def ancestor_functions(state):
    graph = state["graph"]

    @lru_cache(None)
    def root_depth(node):
        return max((root_depth(p) + 1 for p in graph.parents[node]), default=0)

    @lru_cache(None)
    def close(columns):
        if not columns:
            return "", (), (), ""
        candidates = {
            node
            for node, descendants in graph.descendant_columns.items()
            if set(columns).issubset(descendants)
        }
        lowest = [
            n for n in candidates if not candidates.intersection(graph.children[n])
        ]
        lowest.sort(key=lambda n: (len(graph.descendant_columns[n]), -root_depth(n), n))
        anchor = lowest[0]
        term = graph.parents[anchor][0] if anchor in graph.class_nodes else anchor
        terms = tuple(
            graph.parents[n][0] if n in graph.class_nodes else n for n in lowest
        )
        return anchor, graph.descendant_columns[anchor], terms, term

    return close


PREDICTION_COLUMNS = (
    "truth",
    "predicted",
    "prediction_set",
    "set_size",
    "raw_covered",
    "ancestor",
    "ancestor_term",
    "all_lowest_ancestors",
    "expanded_set",
    "ancestor_covered",
    "depth",
)


def evaluate_field(state, field, method):
    """Join saved predictions to field cells by ID; close each distinct set once."""
    graph, saved = state["graph"], state["saved"]
    classes = saved["classes"].astype(str)
    cells = field["cells"]
    ids = cells.loc[cells.evaluated, "row_id"]
    positions = pd.Index(saved["ids"].astype(str)).get_indexer(ids)
    if (positions < 0).any():
        raise ValueError("Evaluated cells are absent from saved predictions")
    close = ancestor_functions(state)
    class_columns = {label: index for index, label in enumerate(classes)}
    records = []
    for position in positions:
        mask = saved[method + "_sets"][position]
        columns = tuple(np.flatnonzero(mask))
        anchor, expanded, lowest, term = close(columns)
        truth = str(saved["y_true"][position])
        true_column = class_columns[truth]
        covered = true_column in expanded
        records.append(
            dict(
                truth=truth,
                predicted=classes[saved["probabilities"][position].argmax()],
                prediction_set="; ".join(classes[list(columns)]) or "∅",
                set_size=len(columns),
                raw_covered=bool(mask[true_column]),
                ancestor_term=term,
                ancestor=state["ontology"].names[term] if term else "Empty set",
                all_lowest_ancestors="; ".join(
                    f"{state['ontology'].names[t]} ({t})" for t in lowest
                ),
                expanded_set="; ".join(classes[list(expanded)]) or "∅",
                ancestor_covered=covered,
                depth=max(0, graph.ancestors_for_column(true_column)[anchor] - 1)
                if covered
                else np.nan,
            )
        )
    predictions = pd.DataFrame(records, columns=PREDICTION_COLUMNS)
    predictions.insert(0, "row_id", ids.to_numpy())
    return cells.merge(predictions, on="row_id", how="left", validate="one_to_one")


def patch_choices(field, size=512):
    cells = field["cells"].loc[field["cells"].evaluated]
    bins = pd.DataFrame(
        {"column": (cells.x // size).astype(int), "row": (cells.y // size).astype(int)}
    )
    counts = bins.groupby(["row", "column"]).size().sort_index()
    height, width = field["tissue"].shape[:2]
    return [
        (
            f"x={c * size}, y={r * size} · {n} evaluated cells",
            (
                c * size,
                r * size,
                min((c + 1) * size, width),
                min((r + 1) * size, height),
            ),
        )
        for (r, c), n in counts.items()
    ]


def cells_in_bounds(cells, bounds):
    x0, y0, x1, y1 = bounds
    return cells.loc[
        (cells.x >= x0) & (cells.x < x1) & (cells.y >= y0) & (cells.y < y1)
    ].copy()


def shortest_path(ontology, start, target):
    pending = deque([(start,)])
    seen = {start}
    while pending:
        path = pending.popleft()
        if path[-1] == target:
            return path
        for parent in sorted(ontology.parents[path[-1]]):
            if parent not in seen:
                seen.add(parent)
                pending.append((*path, parent))
    return ()


def section_thumbnail(field, target_width=1600):
    """Read the smallest whole-slide pyramid level at least this many pixels wide."""
    import tifffile

    height, width = field["tissue"].shape[:2]
    with tifffile.TiffFile(field["image_path"]) as handle:
        levels = handle.series[0].levels
        level = int(
            np.clip(
                np.floor(np.log2(max(height, width) / target_width)), 0, len(levels) - 1
            )
        )
        return np.asarray(levels[level].asarray())[..., :3]


def patch_statistics(rows, size, shape, min_cells):
    """Summarize the evaluated cells of every grid patch of one section."""
    height, width = shape
    cells = rows.loc[rows.evaluated].copy()
    cells["row"], cells["column"] = (
        (cells.y // size).astype(int),
        (cells.x // size).astype(int),
    )
    cells["ancestor_covered"] = cells.ancestor_covered.astype(bool)
    cells["raw_covered"] = cells.raw_covered.astype(bool)
    cells["excluded"] = cells.set_size.gt(0) & ~cells.ancestor_covered
    cells["empty"] = cells.set_size.eq(0)
    cells["correct"] = cells.truth.eq(cells.predicted)
    # Excluded cells carry no depth, so patch depth averages the ancestor-covered cells only.
    table = (
        cells.groupby(["row", "column"])
        .agg(
            evaluated_cells=("row_id", "size"),
            mean_depth=("depth", "mean"),
            median_depth=("depth", "median"),
            mean_set_size=("set_size", "mean"),
            excluded_fraction=("excluded", "mean"),
            empty_fraction=("empty", "mean"),
            ancestor_coverage=("ancestor_covered", "mean"),
            original_coverage=("raw_covered", "mean"),
            top1_accuracy=("correct", "mean"),
        )
        .reset_index()
    )
    table.insert(
        0, "patch", [f"r{r:03d}c{c:03d}" for r, c in zip(table.row, table.column)]
    )
    table["x0"], table["y0"] = table.column * size, table.row * size
    table["x1"] = np.minimum(table.x0 + size, width)
    table["y1"] = np.minimum(table.y0 + size, height)
    return table.loc[table.evaluated_cells >= min_cells].reset_index(drop=True)


TERRA_CACHE = {}


def terra_file(state):
    """Locate one section's saved TERRA embeddings, resolved as the HEST loader does."""
    data = state["config"]["data"]
    manifest = pd.read_csv(data["manifest"])
    matches = manifest.loc[manifest["id"].map(hest._identifier).eq(state["section"])]
    assert len(matches) == 1, f"Expected one manifest row for {state['section']}"
    directory = matches.iloc[0].get("output_dir")
    directory = (
        Path(data["root"]) / state["section"] if pd.isna(directory) else Path(directory)
    )
    return directory / f"{state['section']}_terra.npz"


def terra_matrix(state, name):
    """Read one TERRA matrix; its rows follow the section's cell order."""
    path = terra_file(state).resolve()
    key = (path, name)
    if key not in TERRA_CACHE:
        TERRA_CACHE.clear()  # a section-sized matrix is a few hundred MB; keep only the current one
        with np.load(path, allow_pickle=False) as archive:
            ids = np.asarray([hest._identifier(value) for value in archive["cell_ids"]])
            np.testing.assert_array_equal(ids, state["sample"].cell_ids)
            TERRA_CACHE[key] = np.ascontiguousarray(archive[name], dtype=np.float32)
    return TERRA_CACHE[key]


def save_view(state, method, view, figures, tables, details):
    """Save one displayed view to a new time-stamped folder under the user's own name."""
    user = getpass.getuser()
    saved_at = datetime.now(timezone.utc)
    directory = (
        EXPORT_ROOT / user / state["dataset"] / saved_at.strftime("%Y%m%dT%H%M%S%fZ")
    )
    directory.mkdir(parents=True, exist_ok=False)
    for name, figure in figures.items():
        extension = "pdf" if details.get("metric") == "set_size" else "png"
        figure.savefig(directory / f"{name}.{extension}", dpi=180, bbox_inches="tight")
    for name, table in tables.items():
        table.to_csv(directory / f"{name}.csv", index=False)
    provenance = dict(details)
    provenance.update(
        model="CellBridge",
        representation="uni_terra_four_layers_adapted_all_cells/lora_adapted",
        task="all",
        probe="linear",
        training_config=str(TRAIN_CONFIG) if TRAIN_CONFIG.is_file() else None,
        trainable_blocks=training_config().get("model", {}).get("trainable_blocks"),
        choice=state["choice"],
        predictions=str(state["directory"] / "predictions.npz"),
        mapping=str(state["mapping_path"]),
        ontology=state["config"]["conformal"]["ontology"],
        ontology_release=state["ontology"].release,
        ontology_sha256=state["ontology"].sha256,
        nominal_coverage=float(state["metrics"].loc[method, "target_coverage"]),
        depth_definition="Shortest true-term-to-LCA distance; excluded/empty sets have no depth",
        lca_rule="Lowest candidates; fewest descendant classes, greatest root depth, then ID",
        view=view,
        saved_by=user,
        saved_at=saved_at.isoformat(),
    )
    if details.get("metric") == "set_size":
        provenance.pop("depth_definition")
        provenance.pop("lca_rule")
        provenance["set_size_definition"] = (
            "Number of labels in the saved final prediction set; graph methods already include ontology closure. "
            "Means include every evaluated cell, including empty sets as zero."
        )
    (directory / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return directory
