"""Controlled Switch-style and Loss-Free MoE routing for the alpha sweep."""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from .project import DeepFM_MoE


class ControlledSparseMoE(nn.Module):
    """Sparse MoE with faithful dense-probability balancing controls.

    The legacy implementation exposes only the renormalized sparse gates.  For
    Top-1 routing those gates are one-hot, so they cannot provide the dense
    routing probabilities required by the Switch auxiliary loss.  This module
    keeps those probabilities explicitly and also supports the canonical
    sign-update Loss-Free Balancing algorithm from Wang et al. (2024).
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        num_experts: int,
        hidden_sizes: list[int],
        top_k: int,
        dropout: float,
        balancing_method: str,
        loss_free_update_rate: float,
    ) -> None:
        super().__init__()
        if balancing_method not in {"auxiliary", "loss_free"}:
            raise ValueError(f"Unknown balancing method: {balancing_method}")
        if not 1 <= top_k <= num_experts:
            raise ValueError("top_k must lie in [1, num_experts].")

        self.input_size = input_size
        self.output_size = output_size
        self.num_experts = num_experts
        self.top_k_eval = top_k
        self.balancing_method = balancing_method
        self.loss_free_update_rate = loss_free_update_rate

        self.gate = nn.Linear(input_size, num_experts)
        self.experts = nn.ModuleList()
        for _ in range(num_experts):
            layers: list[nn.Module] = []
            dimensions = [input_size, *hidden_sizes, output_size]
            for index in range(len(dimensions) - 1):
                layers.append(nn.Dropout(dropout))
                layers.append(nn.Linear(dimensions[index], dimensions[index + 1]))
                if index < len(dimensions) - 2:
                    layers.append(nn.LeakyReLU())
            self.experts.append(nn.Sequential(*layers))

        self.register_buffer("loss_free_bias", torch.zeros(num_experts))
        self.register_buffer(
            "loss_free_update_count",
            torch.zeros((), dtype=torch.long),
        )

    @property
    def uses_loss_free_balancing(self) -> bool:
        return self.balancing_method == "loss_free"

    def _update_loss_free_bias(self, topk_indices: torch.Tensor) -> None:
        """Apply the paper's canonical sign update.

        The current batch has already been routed when this runs, so the new
        bias first affects the next batch.  This is equivalent to the
        post-training-step update in Algorithm 1 because the bias has no
        gradient and does not affect the current batch's mixture weights.
        """

        with torch.no_grad():
            counts = torch.bincount(
                topk_indices.detach().reshape(-1),
                minlength=self.num_experts,
            ).to(dtype=self.loss_free_bias.dtype)
            error_sign = torch.sign(counts.mean() - counts)
            self.loss_free_bias.add_(self.loss_free_update_rate * error_sign)
            self.loss_free_update_count.add_(1)

    def forward(
        self,
        inputs: torch.Tensor,
        return_gate: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        original_shape = inputs.shape
        flat_inputs = inputs.reshape(-1, self.input_size)
        logits = self.gate(flat_inputs)
        router_probabilities = F.softmax(logits, dim=-1)

        selection_scores = router_probabilities
        if self.uses_loss_free_balancing:
            selection_scores = selection_scores + self.loss_free_bias.to(
                dtype=selection_scores.dtype
            )
        topk_indices = selection_scores.topk(self.top_k_eval, dim=-1).indices

        selected_probabilities = router_probabilities.gather(1, topk_indices)
        selected_weights = selected_probabilities / selected_probabilities.sum(
            dim=1,
            keepdim=True,
        ).clamp_min(1e-12)
        gates = torch.zeros_like(router_probabilities)
        gates.scatter_(1, topk_indices, selected_weights)

        output = torch.zeros(
            flat_inputs.shape[0],
            self.output_size,
            device=flat_inputs.device,
            dtype=flat_inputs.dtype,
        )
        # Preserve the legacy Switch.py computation path.  Evaluating every
        # expert on the full batch avoids per-expert GPU synchronization and
        # gives all controlled conditions the same dropout RNG schedule.
        for expert_index, expert in enumerate(self.experts):
            expert_output = expert(flat_inputs)
            output += gates[:, expert_index].unsqueeze(1) * expert_output

        if self.training and self.uses_loss_free_balancing:
            self._update_loss_free_bias(topk_indices)

        output = output.reshape(*original_shape[:-1], self.output_size)
        if not return_gate:
            return output
        return output, {
            "logits": logits,
            "router_probabilities": router_probabilities,
            "gates": gates,
            "topk_idx": topk_indices,
        }


class BalancingSweepDeepFMMoE(DeepFM_MoE):
    """Legacy DeepFM-MoE backbone with controlled balancing mechanisms."""

    def __init__(self, config: Any, dataset: Any) -> None:
        super().__init__(config, dataset, item_freq_tensor=None)
        num_experts = int(config["num_experts"])
        top_k = int(config["top_k"])
        balancing_method = str(config["balancing_method"])
        loss_free_update_rate = float(config["loss_free_update_rate"])

        legacy_moe = self.moe_layers
        self.moe_layers = ControlledSparseMoE(
            input_size=legacy_moe.input_size,
            output_size=legacy_moe.output_size,
            num_experts=num_experts,
            hidden_sizes=[32],
            top_k=top_k,
            dropout=self.dropout_prob,
            balancing_method=balancing_method,
            loss_free_update_rate=loss_free_update_rate,
        )
        self.moe_layers.apply(self._init_weights)

    def _aux_load_balance(
        self,
        meta: Dict[str, torch.Tensor],
        num_experts: int,
    ) -> torch.Tensor:
        probabilities = meta["router_probabilities"]
        topk_indices = meta["topk_idx"]
        mean_probability = probabilities.mean(dim=0)
        counts = torch.bincount(
            topk_indices.detach().reshape(-1),
            minlength=num_experts,
        ).to(dtype=probabilities.dtype)
        load_fraction = counts / topk_indices.numel()
        return num_experts * torch.sum(mean_probability * load_fraction)


__all__ = ["BalancingSweepDeepFMMoE", "ControlledSparseMoE"]
