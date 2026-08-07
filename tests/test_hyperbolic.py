import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn
import yaml

geoopt = pytest.importorskip("geoopt")

from cellbridge.exploration.phenotype_pairs import sample_ordering_triplets  # noqa: E402
from cellbridge.hyperbolic.evaluation import evaluate_geometry  # noqa: E402
from cellbridge.hyperbolic.losses import (  # noqa: E402
    cone_half_aperture,
    entailment_cone_loss,
    hyperbolic_contrastive_loss,
)
from cellbridge.hyperbolic.manifold import LorentzOps  # noqa: E402
from cellbridge.hyperbolic.metrics import norm_correctness_correlation  # noqa: E402
from cellbridge.lora.model import LoRAUNIDistiller, compute_loss  # noqa: E402
from cellbridge.lora.train import make_optimizer  # noqa: E402


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(5, 4, bias=False)

    def forward(self, inputs):
        return self.linear(inputs)


def test_lorentz_maps_round_trip_and_points_stay_on_manifold():
    manifold = LorentzOps(3, curvature=0.7).double()
    tangent = torch.tensor([[0.2, -0.4, 0.1], [1.0, 0.3, -0.2]]).double()
    points = manifold.expmap0(tangent)

    assert manifold.manifold.check_point_on_manifold(points)
    torch.testing.assert_close(manifold.logmap0(points), tangent)
    torch.testing.assert_close(
        manifold.distance(points[0], points[1]),
        manifold.distance(points[1], points[0]),
    )


def test_lorentz_aperture_matches_requested_poincare_formula():
    manifold = LorentzOps(2, curvature=0.6).double()
    point = manifold.expmap0(torch.tensor([[0.9, 0.2]]).double())
    aperture, active = cone_half_aperture(point, manifold, K=0.1)
    sqrt_c = manifold.curvature.sqrt()
    poincare = sqrt_c * point[:, 1:] / (sqrt_c * point[:, :1] + 1)
    radius = poincare.norm(dim=-1)
    expected = torch.asin(0.1 * (1 - radius.square()) / radius)

    assert active.item()
    torch.testing.assert_close(aperture, expected)


def test_entailment_cone_accepts_radial_descendant_and_rejects_off_axis():
    manifold = LorentzOps(2).double()
    apex = manifold.expmap0(torch.tensor([[1.0, 0.0]]).double())
    radial = manifold.expmap0(torch.tensor([[2.0, 0.0]]).double())
    off_axis = manifold.expmap0(torch.tensor([[0.0, 2.0]]).double())

    inside = entailment_cone_loss(radial, apex, manifold, K=0.1)
    outside = entailment_cone_loss(off_axis, apex, manifold, K=0.1)
    assert inside["cone_violation_rate"].item() == 0
    assert outside["cone_violation_rate"].item() == 1
    assert outside["loss_cone"] > inside["loss_cone"]


def test_hyperbolic_model_maps_both_modalities_and_uses_riemannian_adam():
    model = LoRAUNIDistiller(
        TinyEncoder(),
        image_dim=4,
        terra_dim=6,
        trainable_blocks="none",
        geometry_mode="hyperbolic",
        alignment_dim=3,
        ontology_depths=[0, 1, 2],
        ontology_parents=[-1, 0, 1],
    )
    image_points = model(torch.randn(5, 5))["projected"]
    terra_points = model.project_terra(torch.randn(5, 6))
    losses = compute_loss(
        {"projected": image_points, "terra_projected": terra_points},
        torch.randn(5, 6),
        manifold=model.lorentz,
    )
    optimizers, counts = make_optimizer(
        model,
        {
            "training": {
                "lr_lora": 1e-4,
                "lr_heads": 3e-4,
                "lr_curvature": 1e-4,
                "lr_manifold": 3e-4,
                "weight_decay": 1e-4,
            }
        },
    )

    assert image_points.shape == terra_points.shape == (5, 4)
    assert model.probe_projection(image_points).shape == (5, 3)
    assert torch.isfinite(losses["loss_total"])
    assert isinstance(model.ontology.points, geoopt.ManifoldParameter)
    assert isinstance(optimizers[1], geoopt.optim.RiemannianAdam)
    assert counts["manifold"] == 12


def test_geodesic_contrastive_prefers_correct_pairing():
    manifold = LorentzOps(2).double()
    points = manifold.expmap0(torch.tensor([[1.0, 0.0], [0.0, 1.0]]).double())
    aligned = hyperbolic_contrastive_loss(points, points, manifold)["loss_contrastive"]
    swapped = hyperbolic_contrastive_loss(points.flip(0), points, manifold)[
        "loss_contrastive"
    ]
    assert aligned < swapped


def test_original_triplet_sampler_accepts_a_geodesic_similarity_callback():
    cells = pd.DataFrame(
        [
            (sample, phenotype, lineage)
            for sample in ("A", "B")
            for phenotype, lineage in (("p1", "l1"), ("p2", "l1"), ("p3", "l2"))
        ],
        columns=["sample_id", "phenotype", "lineage"],
    )
    cells["embedding_index"] = np.arange(len(cells))
    embeddings = np.eye(len(cells), dtype=np.float32)
    expected = sample_ordering_triplets(embeddings, cells, 30, 11)
    actual = sample_ordering_triplets(
        embeddings,
        cells,
        30,
        11,
        similarity=lambda left, right: np.einsum("ij,ij->i", left, right),
    )
    pd.testing.assert_frame_equal(actual, expected)


def test_norm_correctness_correlation_reports_direction_and_constant_guard():
    assert norm_correctness_correlation([1, 2, 3, 4], [0, 0, 0, 0], [1, 1, 0, 0]) > 0
    assert np.isnan(norm_correctness_correlation([1, 1], [0, 1], [0, 0]))


def test_geometry_evaluator_runs_exact_triplets_on_exported_tangents(
    tmp_path, monkeypatch
):
    data_root = tmp_path / "hest"
    artifact_root = tmp_path / "artifacts"
    samples = ("TENX1", "TENX2")
    labels = ("Tumor_Cells", "Normal_Epithelial_Cells", "T_Lymphocytes")
    tangents = np.asarray([[0.9, 0.0], [0.8, 0.1], [0.0, 0.9]], dtype=np.float32)
    paths = {}
    for offset, sample_id in enumerate(samples):
        directory = data_root / sample_id
        directory.mkdir(parents=True)
        cell_ids = np.asarray([f"{sample_id}-{index}" for index in range(3)])
        pd.DataFrame({"cell_id": cell_ids, "group": labels}).to_csv(
            directory / f"{sample_id}_granular_cell_labels.csv", index=False
        )
        path = artifact_root / f"{sample_id}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        values = tangents + np.asarray([0.02 * offset, 0.0], dtype=np.float32)
        np.savez(
            path,
            cell_ids=cell_ids,
            lora_projected=values,
            terra_projected=values,
        )
        paths[sample_id] = path

    metric_path = tmp_path / "metric.yaml"
    metric_path.write_text(
        yaml.safe_dump(
            {
                "data_root": str(data_root),
                "sample_ids": list(samples),
                "label_id_column": "cell_id",
                "label_column": "group",
                "sampling": {
                    "seed": 7,
                    "max_cells_per_phenotype_per_sample": 1,
                    "ordering_triplets": 25,
                },
            }
        )
    )
    model = LoRAUNIDistiller(
        TinyEncoder(),
        image_dim=4,
        terra_dim=6,
        trainable_blocks="none",
        geometry_mode="hyperbolic",
        alignment_dim=2,
    ).eval()
    cfg = {
        "eval": {"hierarchy_metric_config": str(metric_path), "geometry_chunk_size": 8},
        "hierarchy": {"cones": {"enabled": False}},
    }
    monkeypatch.setattr(
        "cellbridge.hyperbolic.evaluation._feature_paths", lambda *_: paths
    )

    result = evaluate_geometry(cfg, model, tmp_path / "evaluation")

    assert result["modality"].tolist() == ["h&e", "terra"]
    assert result["n_triplets"].tolist() == [25, 25]
    assert result["hierarchy_violation_rate"].between(0, 1).all()
    assert (tmp_path / "evaluation" / "ordering_triplets_image.csv.gz").exists()
    assert (tmp_path / "evaluation" / "hierarchy_metrics.json").exists()
