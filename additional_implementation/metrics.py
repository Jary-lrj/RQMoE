"""Streaming effective-rank and routing diagnostics."""

from __future__ import annotations

import math
from typing import Any, Dict

import torch

from .spectral_intervention import SecondMomentAccumulator


def _entropy(probabilities: torch.Tensor) -> torch.Tensor:
    positive = probabilities > 0
    terms = torch.zeros_like(probabilities)
    terms[positive] = probabilities[positive] * probabilities[positive].log()
    return -terms.sum()


def effective_rank(singular_values: torch.Tensor) -> float:
    values = singular_values.to(dtype=torch.float64).clamp_min(0.0)
    total = values.sum()
    if total <= 0:
        return 0.0
    probabilities = values / total
    return float(torch.exp(_entropy(probabilities)).item())


class RoutingDiagnostics:
    """Aggregate diagnostics over the full evaluation set."""

    def __init__(self, dimension: int, num_experts: int, top_k: int, center_spectrum: bool):
        self.num_experts = num_experts
        self.top_k = top_k
        self.spectrum = SecondMomentAccumulator(dimension, center=center_spectrum)
        self.top1_counts = torch.zeros(num_experts, dtype=torch.float64)
        self.selected_counts = torch.zeros(num_experts, dtype=torch.float64)
        self.gate_weight_sums = torch.zeros(num_experts, dtype=torch.float64)
        self.sample_entropy_sum = 0.0
        self.sample_count = 0
        self.zero_norm_count = 0
        self.max_norm_relative_error = 0.0

    def update_representation(self, original: torch.Tensor, router_input: torch.Tensor) -> None:
        self.spectrum.update(router_input)
        original_norm = torch.linalg.vector_norm(original.detach(), dim=-1)
        router_norm = torch.linalg.vector_norm(router_input.detach(), dim=-1)
        self.zero_norm_count += int((original_norm <= 1e-12).sum().item())
        nonzero = original_norm > 1e-12
        if nonzero.any():
            relative_error = (router_norm[nonzero] - original_norm[nonzero]).abs() / original_norm[nonzero]
            self.max_norm_relative_error = max(
                self.max_norm_relative_error,
                float(relative_error.max().item()),
            )

    def update_logits(self, logits: torch.Tensor) -> None:
        detached = logits.detach().to(dtype=torch.float64, device="cpu")
        probabilities = torch.softmax(detached, dim=-1)
        per_sample_entropy = -(probabilities * probabilities.clamp_min(1e-300).log()).sum(dim=-1)
        self.sample_entropy_sum += float(per_sample_entropy.sum().item())
        self.sample_count += detached.shape[0]

        top1 = detached.argmax(dim=-1)
        self.top1_counts += torch.bincount(top1, minlength=self.num_experts)
        selected_values, selected_indices = detached.topk(self.top_k, dim=-1)
        selected = selected_indices.reshape(-1)
        self.selected_counts += torch.bincount(selected, minlength=self.num_experts)
        selected_weights = torch.softmax(selected_values, dim=-1)
        self.gate_weight_sums.scatter_add_(
            0,
            selected,
            selected_weights.reshape(-1),
        )

    def _load_metrics(self, counts: torch.Tensor, prefix: str) -> Dict[str, Any]:
        total = counts.sum()
        if total <= 0:
            probabilities = torch.zeros_like(counts)
            normalized_entropy = 0.0
            cv_squared = 0.0
            max_load_ratio = 0.0
        else:
            probabilities = counts / total
            normalizer = math.log(self.num_experts) if self.num_experts > 1 else 1.0
            normalized_entropy = float((_entropy(probabilities) / normalizer).item())
            cv_squared = float((self.num_experts * probabilities.square().sum() - 1.0).item())
            max_load_ratio = float((self.num_experts * probabilities.max()).item())
        return {
            f"{prefix}_counts": [int(value) for value in counts.tolist()],
            f"{prefix}_probabilities": [float(value) for value in probabilities.tolist()],
            f"{prefix}_normalized_entropy": normalized_entropy,
            f"{prefix}_cv_squared": cv_squared,
            f"{prefix}_max_load_ratio": max_load_ratio,
        }

    def _gate_weight_metrics(self) -> Dict[str, Any]:
        metrics = self._load_metrics(self.gate_weight_sums, "gate_weight_load")
        metrics["gate_weight_load_sums"] = [
            float(value) for value in self.gate_weight_sums.tolist()
        ]
        metrics.pop("gate_weight_load_counts")
        return metrics

    def finalize(self) -> Dict[str, Any]:
        statistics = self.spectrum.finalize()
        normalizer = math.log(self.num_experts) if self.num_experts > 1 else 1.0
        mean_router_entropy = self.sample_entropy_sum / max(self.sample_count, 1)
        result: Dict[str, Any] = {
            "sample_count": self.sample_count,
            "effective_rank": effective_rank(statistics.singular_values),
            "normalized_effective_rank": effective_rank(statistics.singular_values)
            / statistics.dimension,
            "singular_values": [float(value) for value in statistics.singular_values.tolist()],
            "mean_router_entropy": mean_router_entropy,
            "mean_normalized_router_entropy": mean_router_entropy / normalizer,
            "zero_norm_count": self.zero_norm_count,
            "max_norm_relative_error": self.max_norm_relative_error,
        }
        result.update(self._load_metrics(self.top1_counts, "top1_load"))
        result.update(self._load_metrics(self.selected_counts, "selected_load"))
        result.update(self._gate_weight_metrics())
        return result
