"""Geoopt-backed operations for a learnable-curvature Lorentz model."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn

try:
    import geoopt
except ImportError as error:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "Hyperbolic CellBridge requires geoopt; install requirements-hyperbolic.txt"
    ) from error


class LorentzOps(nn.Module):
    """Stable Lorentz operations with curvature ``-c``.

    Geoopt parameterizes its hyperboloid by ``k`` with
    ``<x, x>_L = -k``. A space of curvature ``-c`` therefore uses ``k = 1/c``.
    Encoder heads emit only the spatial tangent coordinates; this wrapper adds
    the zero time coordinate required by :class:`geoopt.Lorentz`.
    """

    def __init__(
        self,
        dim: int,
        curvature: float = 1.0,
        *,
        learnable_curvature: bool = True,
        min_curvature: float = 1e-3,
        max_curvature: float = 10.0,
        max_tangent_norm: float = 12.0,
    ) -> None:
        super().__init__()
        if int(dim) < 1:
            raise ValueError("Lorentz dimension must be positive")
        if not 0 < float(min_curvature) <= float(curvature) <= float(max_curvature):
            raise ValueError("curvature must lie within its positive configured bounds")
        if float(max_tangent_norm) <= 0:
            raise ValueError("max_tangent_norm must be positive")

        self.dim = int(dim)
        self.min_curvature = float(min_curvature)
        self.max_curvature = float(max_curvature)
        self.max_tangent_norm = float(max_tangent_norm)
        self.manifold = geoopt.Lorentz(
            k=torch.tensor(1.0 / float(curvature)),
            learnable=bool(learnable_curvature),
        )

    @property
    def curvature(self) -> torch.Tensor:
        """Positive magnitude ``c`` of the sectional curvature ``-c``."""

        return self.manifold.k.reciprocal()

    @property
    def curvature_parameter(self) -> nn.Parameter | None:
        parameter = self.manifold.k
        return parameter if parameter.requires_grad else None

    def _check_tangent(self, tangent: torch.Tensor) -> None:
        if tangent.ndim < 1 or tangent.shape[-1] != self.dim:
            raise ValueError(f"tangent vectors must end in dimension {self.dim}")

    def _check_point(self, point: torch.Tensor) -> None:
        if point.ndim < 1 or point.shape[-1] != self.dim + 1:
            raise ValueError(f"Lorentz points must end in dimension {self.dim + 1}")

    def clamp_tangent(self, tangent: torch.Tensor) -> torch.Tensor:
        """Bound radial distance before expmap to stay away from the ideal boundary."""

        self._check_tangent(tangent)
        if tangent.dtype in {torch.float16, torch.bfloat16}:
            tangent = tangent.float()
        with torch.autocast(device_type=tangent.device.type, enabled=False):
            norm = tangent.norm(dim=-1, keepdim=True)
            scale = (self.max_tangent_norm / norm.clamp_min(1e-15)).clamp(max=1.0)
            return tangent * scale

    def expmap0(self, tangent: torch.Tensor) -> torch.Tensor:
        tangent = self.clamp_tangent(tangent)
        with torch.autocast(device_type=tangent.device.type, enabled=False):
            ambient = torch.cat((torch.zeros_like(tangent[..., :1]), tangent), dim=-1)
            return self.manifold.expmap0(ambient, project=True)

    def logmap0(self, point: torch.Tensor) -> torch.Tensor:
        self._check_point(point)
        if point.dtype in {torch.float16, torch.bfloat16}:
            point = point.float()
        with torch.autocast(device_type=point.device.type, enabled=False):
            tangent = self.manifold.logmap0(self.manifold.projx(point))
            return tangent[..., 1:]

    def project(self, point: torch.Tensor) -> torch.Tensor:
        self._check_point(point)
        if point.dtype in {torch.float16, torch.bfloat16}:
            point = point.float()
        with torch.autocast(device_type=point.device.type, enabled=False):
            return self.manifold.projx(point)

    def distance(
        self, left: torch.Tensor, right: torch.Tensor, *, keepdim: bool = False
    ) -> torch.Tensor:
        self._check_point(left)
        self._check_point(right)
        if left.dtype in {torch.float16, torch.bfloat16}:
            left = left.float()
        if right.dtype in {torch.float16, torch.bfloat16}:
            right = right.float()
        with torch.autocast(device_type=left.device.type, enabled=False):
            return self.manifold.dist(left, right, keepdim=keepdim)

    def pairwise_distance(
        self, left: torch.Tensor, right: torch.Tensor
    ) -> torch.Tensor:
        if left.ndim != 2 or right.ndim != 2:
            raise ValueError("pairwise distance requires two point matrices")
        return self.distance(left[:, None, :], right[None, :, :])

    def radius(self, point: torch.Tensor) -> torch.Tensor:
        self._check_point(point)
        if point.dtype in {torch.float16, torch.bfloat16}:
            point = point.float()
        with torch.autocast(device_type=point.device.type, enabled=False):
            return self.manifold.dist0(point)

    def parameter(self, point: torch.Tensor) -> geoopt.ManifoldParameter:
        return geoopt.ManifoldParameter(self.project(point), manifold=self.manifold)

    @torch.no_grad()
    def clamp_(self, *points: torch.Tensor) -> None:
        """Clamp curvature and finite radii after optimizer steps."""

        self.manifold.k.clamp_(
            min=1.0 / self.max_curvature,
            max=1.0 / self.min_curvature,
        )
        for point in points:
            tangent = self.logmap0(point)
            point.copy_(self.expmap0(tangent))


class OntologyEmbedding(nn.Module):
    """Learnable Lorentz points initialized along the reviewed ontology tree."""

    def __init__(
        self,
        ops: LorentzOps,
        depths: Sequence[int],
        parents: Sequence[int],
        *,
        init_radius: float = 0.5,
    ) -> None:
        super().__init__()
        depths = torch.as_tensor(depths, dtype=torch.long)
        parents = torch.as_tensor(parents, dtype=torch.long)
        if depths.ndim != 1 or parents.shape != depths.shape or len(depths) == 0:
            raise ValueError(
                "ontology depths and parents must be equal non-empty vectors"
            )
        if torch.any((depths < 0) | (depths > 3)):
            raise ValueError("ontology depths must lie in [0, 3]")
        if float(init_radius) <= 0:
            raise ValueError("ontology init_radius must be positive")
        for index, (depth, parent) in enumerate(zip(depths, parents, strict=True)):
            if depth == 0 and parent != -1:
                raise ValueError("ontology roots cannot have parents")
            if depth > 0 and not 0 <= parent < index:
                raise ValueError("ontology parents must precede their children")

        self.ops = ops
        self.register_buffer("depths", depths)
        self.register_buffer("parents", parents)
        tangent = torch.zeros(len(depths), ops.dim)
        for index in range(len(depths)):
            depth = int(depths[index])
            if depth == 0:
                continue
            parent = int(parents[index])
            direction = tangent[parent]
            if not bool(direction.norm() > 0):
                direction = torch.randn(ops.dim)
            else:
                direction = direction / direction.norm() + 0.05 * torch.randn(ops.dim)
            tangent[index] = direction / direction.norm() * (float(init_radius) * depth)
        self.points = ops.parameter(ops.expmap0(tangent))

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        return self.points[indices]

    @torch.no_grad()
    def clamp_(self) -> None:
        self.ops.clamp_(self.points)
