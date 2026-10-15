"""Figures and tables for notebook 09: the patch view, the section map and the TERRA view."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import HTML, Markdown, display
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap
from matplotlib.lines import Line2D

from cellbridge.qualitative.data import (
    METHODS,
    cells_in_bounds,
    evaluate_field,
    section_thumbnail,
    shortest_path,
    terra_matrix,
)


# One fixed depth colour ramp, shared by the per-cell, smoothed and patch maps.
DEPTH_CMAP = LinearSegmentedColormap.from_list(
    "ontology_depth", ["#4055cf", "#63d6e5", "#c5f0b1", "#fff28a", "#f6a15c", "#bf3438"]
)


SET_SIZE_CMAP = LinearSegmentedColormap.from_list(
    "prediction_set_size", ["#20865C", "#8ABF7B", "#E5C65B", "#D97A38", "#B52B32"]
)


def set_size_scale(nclasses):
    """Zero is grey; singleton to all classes runs green to red."""
    cmap = ListedColormap(["#555555", *SET_SIZE_CMAP(np.linspace(0, 1, nclasses))])
    return cmap, BoundaryNorm(np.arange(-0.5, nclasses + 1.5), cmap.N)


def _format(value, spec):
    return "—" if pd.isna(value) else format(value, spec)


def summary_line(evaluated, metric="depth"):
    """Headline numbers for a group of evaluated cells, as one line of text."""
    if metric == "set_size":
        return (
            f"Mean saved set size {evaluated.set_size.mean():.2f} · "
            f"set coverage {evaluated.raw_covered.astype(bool).mean():.1%} · "
            f"Top-1 correct {evaluated.truth.eq(evaluated.predicted).mean():.1%} · "
            f"empty sets {int(evaluated.set_size.eq(0).sum())}."
        )
    return (
        f"Top-1 correct {evaluated.truth.eq(evaluated.predicted).mean():.1%} · "
        f"original-set coverage {evaluated.raw_covered.astype(bool).mean():.1%} · "
        f"ancestor-expanded coverage {evaluated.ancestor_covered.astype(bool).mean():.1%} · "
        f"mean set size {evaluated.set_size.mean():.2f} · "
        f"mean depth (covered only) {_format(evaluated.depth.mean(), '.2f')} · "
        f"empty sets {int(evaluated.set_size.eq(0).sum())}."
    )


def selected_cell_text(state, chosen, metric="depth"):
    """A selected cell's prediction and common ancestor, with one ontology path per label."""
    head = (
        f"**Selected cell #{chosen.number}** ({chosen.row_id}, circled in magenta; "
        f"original annotation: {chosen.label})"
    )
    if not chosen.evaluated:
        return f"{head} has no saved test prediction."
    if metric == "set_size":
        return (
            f"{head}: reference **{chosen.truth}**, predicted **{chosen.predicted}**, "
            f"saved prediction set **{chosen.prediction_set or 'empty'}**, "
            f"set size **{int(chosen.set_size)}**, "
            f"reference in set: **{'yes' if chosen.raw_covered else 'no'}**."
        )
    if chosen.set_size == 0:
        outcome = "The prediction set is empty, so there is no ancestor and no depth."
    elif not chosen.ancestor_covered:
        outcome = f"The common ancestor **{chosen.ancestor}** excludes the reference, so there is no depth."
    else:
        outcome = f"Common ancestor **{chosen.ancestor}**, depth {int(chosen.depth)}."
    text = (
        f"{head}: reference **{chosen.truth}**, predicted **{chosen.predicted}**, "
        f"prediction set {chosen.prediction_set}. {outcome}"
    )
    if chosen.ancestor_term:
        ontology = state["ontology"]
        terms = state["mapping"].class_terms(chosen.prediction_set.split("; "))
        paths = [
            f"- **{label}**: "
            + " → ".join(
                f"{ontology.names[t]} ({t})"
                for t in shortest_path(ontology, term, chosen.ancestor_term)
            )
            for label, term in terms.items()
        ]
        text += (
            "\n\nCandidate paths up to the ancestor, one shortest path per label:\n\n"
            + "\n".join(paths)
        )
    return text


def smooth_depth(tissue, rows, mpp, bounds, metric="depth"):
    from scipy.ndimage import gaussian_filter
    from scipy.spatial import cKDTree

    x0, y0, x1, y1 = bounds
    step, bandwidth_um = 4, 15.0
    covered = rows.loc[rows.evaluated & rows[metric].notna()]
    if covered.empty:
        return None
    xe, ye = np.arange(x0, x1 + step, step), np.arange(y0, y1 + step, step)
    counts, _, _ = np.histogram2d(covered.y, covered.x, bins=(ye, xe))
    totals, _, _ = np.histogram2d(
        covered.y, covered.x, bins=(ye, xe), weights=covered[metric]
    )
    sigma = bandwidth_um / (mpp * step)
    density = gaussian_filter(counts, sigma, mode="constant", truncate=3)
    numerator = gaussian_filter(totals, sigma, mode="constant", truncate=3)
    mean = np.divide(
        numerator, density, out=np.full_like(density, np.nan), where=density > 0
    )
    xx, yy = np.meshgrid((xe[:-1] + xe[1:]) / 2, (ye[:-1] + ye[1:]) / 2)
    near, _ = cKDTree(covered[["x", "y"]]).query(
        np.column_stack([xx.ravel(), yy.ravel()])
    )
    rgb = tissue[
        (yy - y0).astype(int).clip(0, tissue.shape[0] - 1),
        (xx - x0).astype(int).clip(0, tissue.shape[1] - 1),
    ].astype(float)
    sat = (rgb.max(axis=2) - rgb.min(axis=2)) / np.maximum(rgb.max(axis=2), 1)
    foreground = gaussian_filter((sat > 0.06).astype(float), 1) > 0.5
    valid = (
        foreground
        & (near.reshape(mean.shape) * mpp <= 30)
        & (density * 2 * np.pi * sigma**2 >= 3)
    )
    mean[~valid] = np.nan
    return mean, (xe[0], xe[-1], ye[-1], ye[0])


def render_selection(state, field, method, bounds, cell_id=None, smooth=False, metric="depth"):
    all_rows = evaluate_field(state, field, method)
    rows = cells_in_bounds(all_rows, bounds)
    evaluated = rows.loc[rows.evaluated]
    x0, y0, x1, y1 = map(int, bounds)
    tissue = np.asarray(field["tissue"][y0:y1, x0:x1])[..., :3]
    extent = (x0 - 0.5, x1 - 0.5, y1 - 0.5, y0 - 0.5)
    classes = state["saved"]["classes"].astype(str)
    colours = dict(zip(classes, plt.get_cmap("tab20")(np.linspace(0, 1, len(classes)))))
    if metric == "set_size" and "macrophage" in colours:
        colours["macrophage"] = "#FF6600"
    graph = state["graph"]
    # One fixed scale for every method and field within the dataset/section.
    maximum = max(max(distances.values()) - 1 for distances in graph.distances)
    cmap = ListedColormap(DEPTH_CMAP(np.linspace(0, 1, maximum + 1)))
    norm = BoundaryNorm(np.arange(-0.5, maximum + 1.5), cmap.N)
    continuous_cmap = DEPTH_CMAP
    if metric == "set_size":
        maximum = len(classes)
        cmap, norm = set_size_scale(maximum)
        continuous_cmap = LinearSegmentedColormap.from_list("mean_set_size", cmap.colors)
    fig, axes = plt.subplots(2, 2, figsize=(15, 12), layout="constrained")
    for ax in axes.flat:
        ax.imshow(tissue, extent=extent, alpha=1 if ax is axes[0, 0] else 0.65)
        ax.set(xlim=(x0 - 0.5, x1 - 0.5), ylim=(y1 - 0.5, y0 - 0.5))
        ax.axis("off")
    axes[0, 0].set_title("H&E · selected cell circled when applicable")
    axes[0, 1].set_title("Reference labels · numbered as in the saved cell table")
    axes[1, 0].set_title("CellBridge · highest-scoring label")
    axes[1, 1].set_title(
        ("Prediction-set size" if metric == "set_size" else "Ontology depth / coarseness")
        + (" · local mean" if smooth else " · per cell")
    )
    for ax, column in [(axes[0, 1], "truth"), (axes[1, 0], "predicted")]:
        for label in classes:
            group = evaluated.loc[evaluated[column].eq(label)]
            ax.scatter(
                group.x,
                group.y,
                s=50,
                color=colours[label],
                edgecolors="black" if metric == "set_size" else "white",
                linewidths=0.4,
            )
    missing = rows.loc[~rows.evaluated] if metric == "depth" else rows.iloc[:0]
    # Harmonized reference labels are still visible for non-evaluated cells, when available.
    reference = missing.label.map(state["mapping"].source_to_class)
    for label in classes:
        group = missing.loc[reference.eq(label)]
        axes[0, 1].scatter(
            group.x, group.y, s=54, facecolors="none", edgecolors=[colours[label]]
        )
    unknown = missing.loc[~reference.isin(classes)]
    axes[0, 1].scatter(unknown.x, unknown.y, s=54, facecolors="none", edgecolors="grey")
    for ax in axes[1]:
        ax.scatter(missing.x, missing.y, s=54, facecolors="none", edgecolors="grey")
    covered = evaluated.loc[evaluated[metric].notna()]
    if smooth:
        smoothed = smooth_depth(tissue, rows, field["mpp"], bounds, metric)
        if smoothed is not None and np.isfinite(smoothed[0]).any():
            axes[1, 1].imshow(
                smoothed[0],
                extent=smoothed[1],
                cmap=continuous_cmap,
                vmin=0,
                vmax=maximum,
                alpha=0.78,
                interpolation="nearest",
            )
        else:
            axes[1, 1].text(
                0.5,
                0.03,
                "No supported neighbourhoods to smooth",
                ha="center",
                transform=axes[1, 1].transAxes,
                bbox=dict(facecolor="white", alpha=0.9),
            )
    else:
        axes[1, 1].scatter(
            covered.x,
            covered.y,
            c=covered[metric],
            cmap=cmap,
            norm=norm,
            s=64,
            edgecolors="black" if metric == "set_size" else "white",
            linewidths=0.4,
        )
    if metric == "depth":
        excluded = evaluated.loc[evaluated.depth.isna() & evaluated.set_size.gt(0)]
        empty = evaluated.loc[evaluated.set_size.eq(0)]
        axes[1, 1].scatter(excluded.x, excluded.y, marker="x", c="black", s=80)
        axes[1, 1].scatter(empty.x, empty.y, marker="+", c="#c62828", s=110)
    bar = plt.cm.ScalarMappable(
        cmap=continuous_cmap if smooth else cmap,
        norm=plt.Normalize(0, maximum) if smooth else norm,
    )
    fig.colorbar(
        bar,
        ax=axes[1, 1],
        shrink=0.7,
        ticks=range(maximum + 1),
        label="Labels in saved prediction set" if metric == "set_size" else "Steps up from reference to ancestor",
    )
    if len(rows) <= 60:
        for row in (evaluated if metric == "set_size" else rows).itertuples():
            axes[0, 1].annotate(
                str(row.number),
                (row.x, row.y),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=7,
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none", pad=0.2),
            )
    if cell_id is not None:
        chosen = rows.loc[rows.row_id.eq(cell_id)].iloc[0]
        for ax in axes.flat:
            ax.scatter(
                [chosen.x],
                [chosen.y],
                s=520,
                facecolors="none",
                edgecolors="magenta",
                linewidths=2,
            )
    scale_um = 10 ** np.floor(np.log10((x1 - x0) * field["mpp"] / 4))
    bx, by = x0 + 0.06 * (x1 - x0), y1 - 0.07 * (y1 - y0)
    axes[0, 0].plot([bx, bx + scale_um / field["mpp"]], [by, by], c="black", lw=3)
    axes[0, 0].text(
        bx,
        by - 0.025 * (y1 - y0),
        f"{scale_um:g} µm",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
    )
    handles = [
        Line2D([], [], marker="o", linestyle="none", color=colours[c], label=c)
        for c in classes
    ]
    axes[0, 1].legend(
        handles=handles, loc="upper left", bbox_to_anchor=(1, 1), fontsize=8
    )
    if metric == "depth":
        axes[1, 1].legend(
            handles=[
                Line2D(
                    [],
                    [],
                    marker="x",
                    color="black",
                    linestyle="none",
                    label="Reference excluded by ancestor",
                ),
                Line2D(
                    [],
                    [],
                    marker="+",
                    color="#c62828",
                    linestyle="none",
                    label="Empty prediction set",
                ),
                Line2D(
                    [],
                    [],
                    marker="o",
                    markerfacecolor="none",
                    color="grey",
                    linestyle="none",
                    label="No test prediction",
                ),
            ],
            loc="upper center",
            bbox_to_anchor=(0.5, 0),
            fontsize=8,
        )
    fig.suptitle(
        f"{state['choice']} · CellBridge · {METHODS[method]}\n"
        f"{field['sample_id']}\nx={x0}:{x1}, y={y0}:{y1}",
        fontsize=11,
        wrap=True,
    )
    display(fig)
    plt.close(fig)
    nominal = float(state["metrics"].loc[method, "target_coverage"])
    summary = f"\n\n{summary_line(evaluated, metric)}" if len(evaluated) else ""
    display(
        Markdown(
            f"**{len(rows):,} annotated cells; {len(evaluated):,} with test predictions.** "
            f"Saved nominal coverage: {nominal:.0%}. Numbers are omitted above 60 cells; "
            f"choose a smaller patch or cell view.{summary}"
        )
    )
    columns = [
        "number",
        "row_id",
        "x",
        "y",
        "label",
        "evaluated",
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
    ]
    if metric == "set_size":
        columns = columns[:columns.index("ancestor")]
    if cell_id is not None:
        chosen = rows.loc[rows.row_id.eq(cell_id)].iloc[0]
        display(Markdown(selected_cell_text(state, chosen, metric)))
    return dict(
        figure=fig,
        cells=rows[columns],
        method=method,
        bounds=list(map(int, bounds)),
        cell_id=cell_id,
        smooth=smooth,
        metric=metric,
        sample_id=field["sample_id"],
        image_path=field["image_path"],
        mpp=field["mpp"],
        outside_image_annotations=field["outside_image_annotations"],
        outside_image_evaluated=field["outside_image_evaluated"],
    )


TISSUE_STATISTICS = {
    # label: patch column, higher is better, keeps the fixed ontology-depth scale
    "Mean depth / coarseness": ("mean_depth", False, True),
    "Median depth / coarseness": ("median_depth", False, True),
    "Mean prediction-set size": ("mean_set_size", False, False),
    "Reference excluded by ancestor": ("excluded_fraction", False, False),
    "Empty prediction sets": ("empty_fraction", False, False),
    "Ancestor-expanded coverage": ("ancestor_coverage", True, False),
    "Original-set coverage": ("original_coverage", True, False),
    "Top-1 accuracy": ("top1_accuracy", True, False),
}


# Notebook viewers do not all draw the browser's own title tooltip, so every patch carries
# a hover label of its own; title stays on as a fallback for plain HTML exports.
HOVER_STYLE = """<style>
#{name} {{position:relative;display:inline-block;max-width:100%}}
#{name} > img {{width:100%;display:block}}
#{name} .hit {{position:absolute;cursor:help}}
#{name} .hit:hover {{outline:1.5px solid #111;background:rgba(255,255,255,.25);z-index:2}}
#{name} .tip {{display:none;position:absolute;z-index:3;white-space:nowrap;pointer-events:none;
 padding:4px 7px;border-radius:4px;background:rgba(20,20,20,.93);color:#fff;
 font:11px/1.35 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
 box-shadow:0 1px 4px rgba(0,0,0,.4)}}
#{name} .hit:hover .tip {{display:block}}
#{name} .from-left .tip {{left:0}}
#{name} .from-right .tip {{right:0}}
#{name} .below .tip {{top:100%;margin-top:4px}}
#{name} .above .tip {{bottom:100%;margin-bottom:4px}}
</style>"""


def colour_scale(state, statistic, values):
    """Depth keeps the ontology's fixed scale; other statistics stretch to this section."""
    if statistic == "Mean prediction-set size":
        cmap, _ = set_size_scale(len(state["saved"]["classes"]))
        continuous = LinearSegmentedColormap.from_list("mean_set_size", cmap.colors)
        return continuous, 0.0, len(state["saved"]["classes"]), statistic
    _, higher_is_better, fixed = TISSUE_STATISTICS[statistic]
    cmap = DEPTH_CMAP.reversed() if higher_is_better else DEPTH_CMAP
    if fixed:
        return (
            cmap,
            0.0,
            max(max(d.values()) - 1 for d in state["graph"].distances),
            statistic,
        )
    low, high = np.nanpercentile(values, [2, 98])
    return (
        cmap,
        float(low),
        float(max(high, low + 1e-6)),
        f"{statistic} · scaled to this section",
    )


def hover_targets(fig, ax, table, tips):
    """Display the drawn figure with one hover label per patch."""
    import base64
    import html
    import io
    import uuid

    fig.canvas.draw()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=fig.dpi)
    width, height = fig.get_size_inches() * fig.dpi
    near = ax.transData.transform(np.column_stack([table.x0, table.y0]))
    far = ax.transData.transform(np.column_stack([table.x1, table.y1]))
    name = f"tissue-map-{uuid.uuid4().hex[:8]}"
    boxes = []
    for (left, top), (right, bottom), tip in zip(near, far, tips):
        # Open the label away from whichever edge of the figure the patch sits against.
        anchor = "from-right" if left / width > 0.55 else "from-left"
        side = "above" if top / height < 0.45 else "below"
        label = html.escape(str(tip))
        boxes.append(
            f'<div class="hit {anchor} {side}" title="{label}" '
            f'style="left:{left / width:.4%};top:{1 - top / height:.4%};'
            f'width:{(right - left) / width:.4%};height:{(top - bottom) / height:.4%}">'
            f'<span class="tip">{label}</span></div>'
        )
    display(
        HTML(
            HOVER_STYLE.format(name=name) + f'<div id="{name}">'
            '<img src="data:image/png;base64,'
            f'{base64.b64encode(buffer.getvalue()).decode()}">'
            + "".join(boxes)
            + "</div>"
        )
    )


def render_section(state, field, table, size, statistic, method, min_cells):
    column = TISSUE_STATISTICS[statistic][0]
    cmap, low, high, bar_label = colour_scale(state, statistic, table[column])
    height, width = field["tissue"].shape[:2]
    grid = np.full((int(np.ceil(height / size)), int(np.ceil(width / size))), np.nan)
    grid[table.row.to_numpy(), table.column.to_numpy()] = table[column].to_numpy(float)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5), layout="constrained")
    thumbnail = section_thumbnail(field)
    for ax in axes:
        ax.imshow(
            thumbnail, extent=(0, width, height, 0), alpha=1 if ax is axes[0] else 0.5
        )
        ax.set(xlim=(0, width), ylim=(height, 0))
        ax.axis("off")
    axes[0].set_title("H&E · whole section")
    axes[1].set_title(f"{statistic} · {size} px patches")
    image = axes[1].imshow(
        grid,
        extent=(0, grid.shape[1] * size, grid.shape[0] * size, 0),
        cmap=cmap,
        vmin=low,
        vmax=high,
        alpha=0.8,
        interpolation="nearest",
    )
    fig.colorbar(image, ax=axes[1], shrink=0.7, label=bar_label)
    scale_um = 10 ** np.floor(np.log10(width * field["mpp"] / 4))
    bar_x, bar_y = 0.04 * width, 0.97 * height
    axes[0].plot(
        [bar_x, bar_x + scale_um / field["mpp"]], [bar_y, bar_y], c="black", lw=3
    )
    axes[0].text(
        bar_x,
        bar_y - 0.015 * height,
        f"{scale_um:g} µm",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
    )
    fig.suptitle(
        f"{state['choice']} · CellBridge · {METHODS[method]}\n"
        f"{len(table):,} patches with at least {min_cells} evaluated cells · "
        f"{size} px ≈ {size * field['mpp']:.0f} µm",
        fontsize=11,
    )
    tips = [
        f"{patch} · x={x0}:{x1}, y={y0}:{y1} · {count:,} evaluated cells · {statistic}: {value:.2f}"
        for patch, x0, x1, y0, y1, count, value in zip(
            table.patch,
            table.x0,
            table.x1,
            table.y0,
            table.y1,
            table.evaluated_cells,
            table[column],
        )
    ]
    hover_targets(fig, axes[1], table, tips)
    plt.close(fig)
    return fig


TERRA_MATRICES = {
    "Cell embedding": "cell_emb",
    "Neighbourhood embedding": "neighborhood_emb",
}


TERRA_COLOURS = {
    # label: cell column, discrete classes
    "Reference label": ("truth", True),
    "CellBridge predicted label": ("predicted", True),
    "Ontology depth / coarseness": ("depth", False),
    "Prediction-set size": ("set_size", False),
}


TERRA_NEIGHBOURS = 15


def terra_pool(state, rows, matrix, positions, context, seed=0):
    """Project the patch cells, then a uniform random sample of the rest of the section."""
    import umap
    from sklearn.decomposition import PCA

    others = np.setdiff1d(np.arange(len(matrix)), positions)
    if context < len(others):
        others = np.sort(
            np.random.default_rng(seed).choice(others, context, replace=False)
        )
    order = np.concatenate([positions, others])
    assert len(order) >= 10, (
        "Fewer than ten cells to project; take a larger patch or add context."
    )
    values = matrix[order]
    reduced = PCA(n_components=min(50, *values.shape), random_state=seed).fit_transform(
        values
    )
    points = np.asarray(
        umap.UMAP(
            n_neighbors=max(2, min(15, len(order) - 1)), min_dist=0.1, random_state=seed
        ).fit_transform(reduced)
    )
    pool = pd.DataFrame(
        {
            "row_id": np.asarray(state["sample"].cell_ids)[order],
            "in_patch": np.arange(len(order)) < len(positions),
            "umap_x": points[:, 0],
            "umap_y": points[:, 1],
        }
    )
    pool = pool.merge(rows, on="row_id", how="left", validate="one_to_one")
    pool["evaluated"] = pool.evaluated.fillna(False).astype(bool)
    return pool, reduced


def terra_agreement(reduced, pool, neighbours=TERRA_NEIGHBOURS):
    """Share of a cell's nearest TERRA neighbours carrying its own reference label."""
    from sklearn.neighbors import NearestNeighbors

    known = np.flatnonzero(pool.evaluated.to_numpy())
    shares = np.full(len(pool), np.nan)
    if len(known) > 1:
        count = min(neighbours, len(known) - 1)
        found = (
            NearestNeighbors(n_neighbors=count + 1)
            .fit(reduced[known])
            .kneighbors(reduced[known], return_distance=False)
        )
        truth = pool.truth.to_numpy()[known]
        # A point is normally its own first neighbour; drop it wherever it is returned.
        elsewhere = found != np.arange(len(known))[:, None]
        shares[known] = ((truth[found] == truth[:, None]) & elsewhere).sum(
            1
        ) / elsewhere.sum(1)
    return shares


def render_terra(state, field, rows, bounds, method, name, colour, context, metric="depth"):
    """Project one patch into TERRA space beside its tissue and its neighbourhood agreement."""
    matrix = terra_matrix(state, TERRA_MATRICES[name])
    patch = cells_in_bounds(rows, bounds)
    positions = pd.Index(state["sample"].cell_ids).get_indexer(patch.row_id)
    assert (positions >= 0).all(), "A patch cell is absent from the TERRA embedding"
    pool, reduced = terra_pool(state, rows, matrix, positions, context)
    pool["neighbour_agreement"] = terra_agreement(reduced, pool)
    column, discrete = TERRA_COLOURS[colour]
    classes = state["saved"]["classes"].astype(str)
    maximum = max(max(distances.values()) - 1 for distances in state["graph"].distances)
    inside, outside = pool.loc[pool.in_patch], pool.loc[~pool.in_patch]
    scored = inside.loc[inside.evaluated]
    x0, y0, x1, y1 = map(int, bounds)
    tissue = np.asarray(field["tissue"][y0:y1, x0:x1])[..., :3]
    fig, axes = plt.subplots(1, 3, figsize=(19, 6.4), layout="constrained")
    axes[0].scatter(outside.umap_x, outside.umap_y, s=3, c="#dcdcdc", linewidths=0)
    axes[0].set(xlabel="UMAP 1", ylabel="UMAP 2", xticks=[], yticks=[])
    axes[1].imshow(tissue, extent=(x0 - 0.5, x1 - 0.5, y1 - 0.5, y0 - 0.5), alpha=0.65)
    axes[1].set(xlim=(x0 - 0.5, x1 - 0.5), ylim=(y1 - 0.5, y0 - 0.5))
    axes[1].axis("off")
    blank = inside.loc[~inside.evaluated] if metric == "depth" else inside.iloc[:0]
    axes[0].scatter(
        blank.umap_x,
        blank.umap_y,
        s=24,
        facecolors="none",
        edgecolors="grey",
        linewidths=0.5,
    )
    axes[1].scatter(
        blank.x, blank.y, s=44, facecolors="none", edgecolors="grey", linewidths=0.5
    )
    if discrete:
        colours = dict(
            zip(classes, plt.get_cmap("tab20")(np.linspace(0, 1, len(classes))))
        )
        for label in classes:
            group = scored.loc[scored[column].eq(label)]
            axes[0].scatter(
                group.umap_x,
                group.umap_y,
                s=32,
                color=colours[label],
                edgecolors="white",
                linewidths=0.4,
            )
            axes[1].scatter(
                group.x,
                group.y,
                s=52,
                color=colours[label],
                edgecolors="white",
                linewidths=0.4,
            )
        fig.legend(
            handles=[
                Line2D([], [], marker="o", linestyle="none", color=colours[c], label=c)
                for c in classes
                if scored[column].eq(c).any()
            ],
            loc="outside lower center",
            ncol=5,
            fontsize=8,
        )
    else:
        if column == "depth":
            cmap = ListedColormap(DEPTH_CMAP(np.linspace(0, 1, maximum + 1)))
            norm, ticks = (
                BoundaryNorm(np.arange(-0.5, maximum + 1.5), cmap.N),
                range(maximum + 1),
            )
            marked = scored.loc[scored.depth.notna()]
        else:
            cmap, norm = set_size_scale(len(classes))
            ticks, marked = range(len(classes) + 1), scored
        axes[0].scatter(
            marked.umap_x,
            marked.umap_y,
            c=marked[column],
            cmap=cmap,
            norm=norm,
            s=32,
            edgecolors="white",
            linewidths=0.4,
        )
        axes[1].scatter(
            marked.x,
            marked.y,
            c=marked[column],
            cmap=cmap,
            norm=norm,
            s=52,
            edgecolors="white",
            linewidths=0.4,
        )
        fig.colorbar(
            plt.cm.ScalarMappable(cmap=cmap, norm=norm),
            ax=axes[1],
            shrink=0.75,
            ticks=ticks,
            label=colour,
        )
        if column == "depth":
            excluded = scored.loc[scored.depth.isna() & scored.set_size.gt(0)]
            empty = scored.loc[scored.set_size.eq(0)]
            axes[0].scatter(
                excluded.umap_x, excluded.umap_y, marker="x", c="black", s=44
            )
            axes[0].scatter(empty.umap_x, empty.umap_y, marker="+", c="#c62828", s=60)
            axes[1].scatter(excluded.x, excluded.y, marker="x", c="black", s=64)
            axes[1].scatter(empty.x, empty.y, marker="+", c="#c62828", s=88)
    if scored.neighbour_agreement.notna().any():
        # Depth levels keep the ontology ramp; excluded and empty sets keep their × and + colours.
        buckets = [
            "Empty set"
            if size == 0
            else "Excluded"
            if pd.isna(depth)
            else str(int(depth))
            for depth, size in zip(scored.depth, scored.set_size)
        ]
        if metric == "set_size":
            maximum = len(classes)
            buckets = scored.set_size.astype(int).astype(str)
        summary = (
            scored.assign(bucket=buckets)
            .groupby("bucket")
            .agg(agreement=("neighbour_agreement", "mean"), cells=("row_id", "size"))
        )
        summary = summary.reindex(
            [
                b
                for b in [*map(str, range(maximum + 1)), "Excluded", "Empty set"]
                if b in summary.index
            ]
        )
        palette = {"Excluded": "black", "Empty set": "#c62828"}
        bars = axes[2].bar(
            summary.index,
            summary.agreement,
            edgecolor="white",
            color=[
                set_size_scale(len(classes))[0](int(b)) if metric == "set_size"
                else palette[b] if b in palette else DEPTH_CMAP(int(b) / max(maximum, 1))
                for b in summary.index
            ],
        )
        axes[2].bar_label(
            bars, labels=[f"n={n:,}" for n in summary.cells], fontsize=8, padding=2
        )
        axes[2].set(
            ylim=(0, 1.15),
            xlabel="Prediction-set size" if metric == "set_size" else "Ontology depth / coarseness",
            ylabel=f"Share of the {TERRA_NEIGHBOURS} nearest TERRA neighbours\n"
            "with the same reference label",
        )
        axes[2].spines[["top", "right"]].set_visible(False)
    else:
        axes[2].text(
            0.5,
            0.5,
            "No evaluated cells to compare",
            ha="center",
            va="center",
            transform=axes[2].transAxes,
        )
        axes[2].axis("off")
    axes[0].set_title(
        f"TERRA {name.lower()} · UMAP\n"
        f"{len(inside):,} patch cells over {len(outside):,} context cells"
    )
    axes[1].set_title(f"The same patch in tissue\ncoloured by {colour.lower()}")
    axes[2].set_title(
        f"Transcriptomic neighbourhood agreement\n"
        f"{TERRA_NEIGHBOURS} nearest neighbours in TERRA PCA space"
    )
    fig.suptitle(
        f"{state['choice']} · TERRA {name.lower()} · CellBridge · {METHODS[method]}\n"
        f"{field['sample_id']} · x={x0}:{x1}, y={y0}:{y1}",
        fontsize=11,
    )
    display(fig)
    plt.close(fig)
    summary = (
        f"\n\nMean neighbour agreement {_format(scored.neighbour_agreement.mean(), '.1%')} · "
        f"mean depth (covered only) {_format(scored.depth.mean(), '.2f')} · "
        f"ancestor-expanded coverage {scored.ancestor_covered.astype(bool).mean():.1%} · "
        f"mean set size {scored.set_size.mean():.2f} · "
        f"reference excluded {int((scored.depth.isna() & scored.set_size.gt(0)).sum())} · "
        f"empty sets {int(scored.set_size.eq(0).sum())}."
        if len(scored)
        else ""
    )
    if metric == "set_size":
        summary = (
            f"\n\nMean neighbour agreement {_format(scored.neighbour_agreement.mean(), '.1%')} · "
            + summary_line(scored, metric)
        ) if len(scored) else ""
    display(
        Markdown(
            f"**{len(inside):,} cells in the patch; {len(scored):,} with test predictions**, "
            f"projected with {len(outside):,} context cells from the rest of the section. "
            f"TERRA embeds these cells' Xenium counts; CellBridge never saw them.{summary}"
        )
    )
    cells = inside[
        [
            "number",
            "row_id",
            "x",
            "y",
            "label",
            "evaluated",
            "truth",
            "predicted",
            "prediction_set",
            "set_size",
            "ancestor",
            "depth",
            "umap_x",
            "umap_y",
            "neighbour_agreement",
        ]
    ]
    if metric == "set_size":
        cells = cells.drop(columns=["ancestor", "depth"])
    return dict(
        figure=fig,
        cells=cells,
        metric=metric,
        pool=pool,
        patch=scored,
        bounds=[x0, y0, x1, y1],
        method=method,
        embedding=TERRA_MATRICES[name],
        colour=column,
        context=len(outside),
    )
