"""Fitted spectral interventions applied only to a MoE router input."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


class SecondMomentAccumulator:
    """Accumulate a representation matrix through sufficient statistics."""

    def __init__(self, dimension: int, center: bool = False):
        self.dimension = dimension
        self.center = center
        self.count = 0
        self.vector_sum = torch.zeros(dimension, dtype=torch.float64)
        self.gram = torch.zeros(dimension, dimension, dtype=torch.float64)

    def update(self, values: torch.Tensor) -> None:
        matrix = values.detach().reshape(-1, self.dimension).to(device="cpu", dtype=torch.float64)
        self.count += matrix.shape[0]
        self.vector_sum += matrix.sum(dim=0)
        self.gram += matrix.transpose(0, 1) @ matrix

    def finalize(self) -> "SpectralStatistics":
        if self.count == 0:
            raise ValueError("Cannot fit a spectral intervention without calibration samples.")

        mean = self.vector_sum / self.count if self.center else torch.zeros_like(self.vector_sum)
        scatter = self.gram
        if self.center:
            scatter = scatter - self.count * torch.outer(mean, mean)
        scatter = (scatter + scatter.transpose(0, 1)) * 0.5

        eigenvalues, eigenvectors = torch.linalg.eigh(scatter)
        order = torch.argsort(eigenvalues, descending=True)
        eigenvalues = eigenvalues[order].clamp_min(0.0)
        basis = eigenvectors[:, order]
        singular_values = eigenvalues.sqrt()
        return SpectralStatistics(
            count=self.count,
            mean=mean.to(dtype=torch.float32),
            basis=basis.to(dtype=torch.float32),
            singular_values=singular_values.to(dtype=torch.float32),
            centered=self.center,
        )


@dataclass(frozen=True)
class SpectralStatistics:
    """Fixed feature-space basis fitted on calibration router inputs."""

    count: int
    mean: torch.Tensor
    basis: torch.Tensor
    singular_values: torch.Tensor
    centered: bool

    @property
    def dimension(self) -> int:
        return int(self.basis.shape[0])


class NormPreservingIntervention(nn.Module):
    """Base class for row-wise L2-norm-preserving interventions."""

    def __init__(self, statistics: SpectralStatistics, eps: float = 1e-12):
        super().__init__()
        self.register_buffer("mean", statistics.mean.clone(), persistent=False)
        self.register_buffer("basis", statistics.basis.clone(), persistent=False)
        self.register_buffer("singular_values", statistics.singular_values.clone(), persistent=False)
        self.centered = statistics.centered
        self.eps = eps

    def _spectral_transform(self, values: torch.Tensor, gains: torch.Tensor) -> torch.Tensor:
        centered = values - self.mean if self.centered else values
        coordinates = centered @ self.basis
        transformed = (coordinates * gains) @ self.basis.transpose(0, 1)
        return transformed + self.mean if self.centered else transformed

    def _restore_row_norm(self, original: torch.Tensor, transformed: torch.Tensor) -> torch.Tensor:
        original_norm = torch.linalg.vector_norm(original, dim=-1, keepdim=True)
        transformed_norm = torch.linalg.vector_norm(transformed, dim=-1, keepdim=True)
        invalid = (original_norm > self.eps) & (transformed_norm <= self.eps)
        if invalid.any():
            count = int(invalid.sum().item())
            raise RuntimeError(
                f"The spectral intervention annihilated {count} non-zero router inputs; "
                "increase retained rank or the spectral floor."
            )
        scale = torch.where(
            original_norm > self.eps,
            original_norm / transformed_norm.clamp_min(self.eps),
            torch.ones_like(original_norm),
        )
        return transformed * scale


class IdentityIntervention(nn.Module):
    """Exact identity used to establish the unmodified checkpoint baseline."""

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values


class TailCollapseIntervention(NormPreservingIntervention):
    """Scale singular directions after ``retained_rank`` by ``gamma``."""

    def __init__(
        self,
        statistics: SpectralStatistics,
        retained_rank: int,
        gamma: float,
        eps: float = 1e-12,
    ):
        super().__init__(statistics, eps=eps)
        if retained_rank < 1 or retained_rank > statistics.dimension:
            raise ValueError(
                f"retained_rank must be in [1, {statistics.dimension}], got {retained_rank}."
            )
        if gamma < 0.0 or gamma > 1.0:
            raise ValueError(f"gamma must be in [0, 1], got {gamma}.")
        self.retained_rank = retained_rank
        self.gamma = float(gamma)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if self.gamma == 1.0:
            return values
        gains = values.new_full((self.basis.shape[1],), self.gamma)
        gains[: self.retained_rank] = 1.0
        transformed = self._spectral_transform(values, gains)
        return self._restore_row_norm(values, transformed)


class SpectralFlatteningIntervention(NormPreservingIntervention):
    """Flatten the fitted spectrum while retaining each sample's L2 norm."""

    def __init__(
        self,
        statistics: SpectralStatistics,
        strength: float,
        floor_ratio: float = 1e-3,
        max_gain: float = 100.0,
        eps: float = 1e-12,
    ):
        super().__init__(statistics, eps=eps)
        if strength < 0.0 or strength > 1.0:
            raise ValueError(f"strength must be in [0, 1], got {strength}.")
        if floor_ratio <= 0.0 or floor_ratio > 1.0:
            raise ValueError(f"floor_ratio must be in (0, 1], got {floor_ratio}.")
        if max_gain < 1.0:
            raise ValueError(f"max_gain must be at least 1, got {max_gain}.")
        self.strength = float(strength)
        self.floor_ratio = float(floor_ratio)
        self.max_gain = float(max_gain)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if self.strength == 0.0:
            return values
        singular_values = self.singular_values.to(dtype=values.dtype)
        leading = singular_values[0].clamp_min(self.eps)
        floor = leading * self.floor_ratio
        safe_values = singular_values.clamp_min(floor)
        gains = (leading / safe_values).pow(self.strength).clamp_max(self.max_gain)
        transformed = self._spectral_transform(values, gains)
        return self._restore_row_norm(values, transformed)
