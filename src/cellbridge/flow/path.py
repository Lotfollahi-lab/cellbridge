"""STFlow's linear endpoint-prediction path, generalized to TERRA latents."""

from __future__ import annotations

import torch


def linear_path(
    target: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Sample ``x_t = (1 - t) x_0 + t x_1`` with a Gaussian prior."""

    if target.ndim != 2 or len(target) == 0:
        raise ValueError("target must be a non-empty [batch, features] tensor")
    noise = torch.randn(
        target.shape,
        dtype=target.dtype,
        device=target.device,
        generator=generator,
    )
    time = torch.rand(
        len(target), dtype=target.dtype, device=target.device, generator=generator
    )
    state = (1.0 - time[:, None]) * noise + time[:, None] * target
    return state, time, noise


@torch.no_grad()
def sample_endpoint(
    model,
    condition: torch.Tensor,
    coordinates: torch.Tensor,
    target_dim: int,
    *,
    steps: int,
    generator: torch.Generator,
    prompts=None,
) -> torch.Tensor:
    """Integrate STFlow's endpoint-derived velocity and return ``x_1``."""

    if steps < 2:
        raise ValueError("steps must be at least two")
    state = torch.randn(
        (len(condition), target_dim),
        device=condition.device,
        dtype=condition.dtype,
        generator=generator,
    )
    times = torch.linspace(0.01, 1.0, steps, device=state.device, dtype=state.dtype)
    for index, time_value in enumerate(times):
        time = time_value.expand(len(state))
        endpoint = model(state, time, condition, coordinates, prompts=prompts)
        if index == len(times) - 1:
            return endpoint
        delta = times[index + 1] - time_value
        velocity = (endpoint - state) / (1.0 - time_value)
        state = state + delta * velocity
    raise RuntimeError("unreachable")
