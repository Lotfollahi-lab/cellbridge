import numpy as np
import pandas as pd
import pytest
import torch

from cellbridge.jepa.data import (
    ContextSection,
    SectionEmbeddings,
    SpatialJEPADataset,
    SpatialWindowSampler,
    WindowSpec,
    collate_windows,
    fit_target_statistics,
    load_context_section,
    load_section,
    load_terra_targets,
    split_sample_ids,
)


def make_section(sample_id="A", side=9, spacing=10.0):
    coordinates = np.asarray(
        [(x * spacing, y * spacing) for y in range(side) for x in range(side)],
        dtype=np.float64,
    )
    rows = np.arange(len(coordinates), dtype=np.float32)
    return SectionEmbeddings(
        sample_id=sample_id,
        cell_ids=np.asarray([f"{sample_id}-{row}" for row in range(len(rows))]),
        image=np.column_stack((rows, rows + 0.5, rows + 1.0)),
        terra=np.column_stack((rows, -rows)),
        coordinates_um=coordinates,
    )


def write_section(root, sample_id, side=7, pixel_size_um=2.0):
    section = make_section(sample_id, side=side, spacing=5.0)
    directory = root / sample_id
    directory.mkdir(parents=True)
    # Deliberately reverse UNI order: the loader must align TERRA and coordinates.
    order = np.arange(len(section.cell_ids))[::-1]
    np.savez(
        directory / f"{sample_id}_uni2.npz",
        cell_ids=section.cell_ids[order],
        uni_emb=section.image[order],
    )
    np.savez(
        directory / f"{sample_id}_terra.npz",
        cell_ids=section.cell_ids,
        cell_emb=section.terra,
        he_x=section.coordinates_um[:, 0],
        he_y=section.coordinates_um[:, 1],
    )
    return {
        "id": sample_id,
        "pixel_size_um": pixel_size_um,
        "output_dir": str(directory),
    }


def test_split_sample_ids_requires_disjoint_section_assignments(tmp_path):
    path = tmp_path / "splits.csv"
    pd.DataFrame(
        {"sample_id": ["A", "B", "C"], "split": ["train", "train", "val"]}
    ).to_csv(path, index=False)

    assert split_sample_ids(path, "train") == ["A", "B"]
    assert set(split_sample_ids(path, "train")).isdisjoint(
        split_sample_ids(path, "val")
    )

    pd.DataFrame({"sample_id": ["A", "A"], "split": ["train", "val"]}).to_csv(
        path, index=False
    )
    with pytest.raises(ValueError, match="disjoint"):
        split_sample_ids(path, "train")


def test_load_section_aligns_ids_converts_pixels_to_microns_and_caps(tmp_path):
    manifest = tmp_path / "manifest.csv"
    pd.DataFrame([write_section(tmp_path, "A", side=3)]).to_csv(manifest, index=False)

    section = load_section(tmp_path, manifest, "A")
    assert section.cell_ids.tolist() == [f"A-{row}" for row in range(8, -1, -1)]
    np.testing.assert_array_equal(section.image[:, 0], np.arange(8, -1, -1))
    np.testing.assert_array_equal(section.terra[:, 0], np.arange(8, -1, -1))
    assert section.image.dtype == np.float16
    assert section.terra.dtype == np.float16
    # Stored H&E centers are pixels; manifest pixel size is 2 microns per pixel.
    np.testing.assert_array_equal(section.coordinates_um[0], [20.0, 20.0])

    first = load_section(tmp_path, manifest, "A", max_cells_per_section=5, seed=7)
    second = load_section(tmp_path, manifest, "A", max_cells_per_section=5, seed=7)
    assert len(first.cell_ids) == 5
    np.testing.assert_array_equal(first.cell_ids, second.cell_ids)
    np.testing.assert_array_equal(
        first.terra,
        load_terra_targets(
            tmp_path,
            manifest,
            "A",
            max_cells_per_section=5,
            seed=7,
        ),
    )


def test_load_context_section_does_not_require_terra_embeddings(tmp_path):
    manifest = tmp_path / "manifest.csv"
    row = write_section(tmp_path, "A", side=3)
    terra_path = tmp_path / "A" / "A_terra.npz"
    with np.load(terra_path, allow_pickle=False) as data:
        values = {key: data[key] for key in data.files if key != "cell_emb"}
    np.savez(terra_path, **values)
    pd.DataFrame([row]).to_csv(manifest, index=False)

    section = load_context_section(tmp_path, manifest, "A")

    assert isinstance(section, ContextSection)
    assert len(section.cell_ids) == 9
    assert section.image.dtype == np.float16
    np.testing.assert_array_equal(section.coordinates_um[0], [20.0, 20.0])
    with pytest.raises(ValueError, match="missing key 'cell_emb'"):
        load_section(tmp_path, manifest, "A")


def test_spatial_window_has_target_centered_uniform_context_rings():
    section = make_section()
    spec = WindowSpec(
        target_cells=3,
        context_cells=6,
        context_radius_um=45.0,
        target_guard_um=15.0,
    )
    sampler = SpatialWindowSampler(section, spec, anchor_indices=[40])
    first = sampler.sample(np.random.default_rng(4))
    second = sampler.sample(np.random.default_rng(4))

    assert first["target_cell_ids"] == second["target_cell_ids"]
    assert first["context_cell_ids"] == second["context_cell_ids"]
    assert torch.equal(first["context_image"], second["context_image"])
    assert first["anchor_cell_id"] == "A-40"
    assert first["target_cell_ids"][0] == first["anchor_cell_id"]
    assert first["coordinate_normalization"] == "fixed_radius"
    assert first["context_image"].shape == (3, 6, 3)
    assert first["context_coords"].shape == (3, 6, 2)
    assert first["target_terra"].shape == (3, 1, 2)
    assert first["target_coords"].shape == (3, 1, 2)
    assert torch.count_nonzero(first["target_coords"]) == 0

    row_for_id = {cell_id: row for row, cell_id in enumerate(section.cell_ids)}
    anchor = row_for_id[first["anchor_cell_id"]]
    targets = np.asarray([row_for_id[cell_id] for cell_id in first["target_cell_ids"]])
    assert set(targets[1:]).isdisjoint(sampler.anchor_indices)
    target_distances = np.linalg.norm(
        section.coordinates_um - section.coordinates_um[anchor], axis=1
    )
    nearest_threshold = np.partition(target_distances, spec.target_cells - 1)[
        spec.target_cells - 1
    ]
    assert target_distances[targets].max() <= nearest_threshold

    for target_row, context_ids, context_coords in zip(
        targets,
        first["context_cell_ids"],
        first["context_coords"],
        strict=True,
    ):
        context = np.asarray([row_for_id[cell_id] for cell_id in context_ids])
        distances = np.linalg.norm(
            section.coordinates_um[context] - section.coordinates_um[target_row],
            axis=1,
        )
        assert len(np.unique(context)) == spec.context_cells
        assert np.all(distances >= spec.target_guard_um)
        assert np.all(distances <= spec.context_radius_um)
        np.testing.assert_allclose(
            context_coords.numpy(),
            (section.coordinates_um[context] - section.coordinates_um[target_row])
            / spec.context_radius_um,
        )

    # Sampling is uniform rather than a fixed nearest-neighbour selection.
    sampled_rings = {
        SpatialWindowSampler(section, spec, anchor_indices=[40]).sample(
            np.random.default_rng(seed)
        )["context_cell_ids"][0]
        for seed in range(5)
    }
    assert len(sampled_rings) > 1


def test_spatial_window_reports_impossible_geometry():
    section = make_section(side=4, spacing=20.0)
    sampler = SpatialWindowSampler(
        section,
        WindowSpec(
            target_cells=2,
            context_cells=3,
            context_radius_um=10.0,
            target_guard_um=5.0,
            max_attempts=3,
        ),
    )
    with pytest.raises(RuntimeError, match="no valid spatial window"):
        sampler.sample(np.random.default_rng(0))


def test_dataset_visits_sections_in_blocks_and_collates_fixed_shapes(tmp_path):
    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame(
        [write_section(tmp_path, sample_id) for sample_id in ("A", "B", "C")]
    ).to_csv(manifest_path, index=False)
    splits_path = tmp_path / "splits.csv"
    pd.DataFrame(
        {
            "sample_id": ["A", "B", "C"],
            "split": ["train", "train", "val"],
        }
    ).to_csv(splits_path, index=False)
    spec = WindowSpec(
        target_cells=2,
        context_cells=4,
        context_radius_um=60.0,
        target_guard_um=5.0,
    )
    dataset = SpatialJEPADataset(
        tmp_path,
        manifest_path,
        splits_path,
        "train",
        windows_per_section=2,
        spec=spec,
        seed=3,
        max_anchor_cells_per_section=30,
    )
    duplicate = SpatialJEPADataset(
        tmp_path,
        manifest_path,
        splits_path,
        "train",
        windows_per_section=2,
        spec=spec,
        seed=3,
        max_anchor_cells_per_section=30,
    )

    assert len(dataset) == 4
    sampler = dataset._sampler(0)
    assert len(sampler.section.cell_ids) == 49
    assert len(sampler.anchor_indices) == 30
    assert sampler.section.image.dtype == np.float16
    assert sampler.section.terra.dtype == np.float16
    assert len(dataset.load_section("A").cell_ids) == 49
    windows = [dataset[index] for index in range(len(dataset))]
    assert [window["sample_id"] for window in windows] == [
        dataset.section_order[0],
        dataset.section_order[0],
        dataset.section_order[1],
        dataset.section_order[1],
    ]
    assert [window["target_cell_ids"] for window in windows] == [
        duplicate[index]["target_cell_ids"] for index in range(len(duplicate))
    ]

    batch = collate_windows(windows[:2])
    assert batch["context_image"].shape == (4, 4, 3)
    assert batch["context_coords"].shape == (4, 4, 2)
    assert batch["target_terra"].shape == (4, 1, 2)
    assert batch["target_coords"].shape == (4, 1, 2)
    assert batch["coordinate_normalization"] == ["fixed_radius"] * 4
    assert batch["target_cell_ids"] == [
        target_id for window in windows[:2] for target_id in window["target_cell_ids"]
    ]
    assert len(batch["context_cell_ids"]) == 4
    assert fit_target_statistics(dataset).count == 60


def test_fit_target_statistics_streams_sections():
    first = SectionEmbeddings(
        "A",
        np.asarray(["a", "b"]),
        np.ones((2, 1), dtype=np.float32),
        np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        np.asarray([[0.0, 0.0], [1.0, 1.0]]),
    )
    second = SectionEmbeddings(
        "B",
        np.asarray(["c"]),
        np.ones((1, 1), dtype=np.float32),
        np.asarray([[5.0, 6.0]], dtype=np.float32),
        np.asarray([[2.0, 2.0]]),
    )
    statistics = fit_target_statistics([first, second.terra])

    assert statistics.count == 3
    np.testing.assert_allclose(statistics.mean, [3.0, 4.0])
    np.testing.assert_allclose(statistics.std, np.sqrt([8.0 / 3.0, 8.0 / 3.0]))
