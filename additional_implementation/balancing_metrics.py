"""Whole-test prediction and expert-assignment metrics for balancing runs."""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List

import numpy as np
import torch
from sklearn.metrics import log_loss, roc_auc_score

from .balancing_model import BalancingSweepDeepFMMoE


def _interaction(batch: Any) -> Any:
    return batch[0] if isinstance(batch, (tuple, list)) else batch


def _model_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def _load_statistics(counts: torch.Tensor) -> Dict[str, Any]:
    counts = counts.to(dtype=torch.float64, device="cpu")
    total = counts.sum()
    if total <= 0:
        raise ValueError("No expert assignments were collected.")
    probabilities = counts / total
    positive = probabilities > 0
    entropy = float(
        -(probabilities[positive] * probabilities[positive].log()).sum().item()
    )
    normalizer = math.log(len(counts)) if len(counts) > 1 else 1.0
    mean_load = counts.mean()
    coefficient_of_variation = float(
        (counts.std(unbiased=False) / mean_load).item()
    )
    max_load_ratio = float((counts.max() / mean_load).item())
    return {
        "expert_load_counts": [int(value) for value in counts.tolist()],
        "expert_load_probabilities": [float(value) for value in probabilities.tolist()],
        "expert_load_entropy": entropy,
        "expert_load_normalized_entropy": entropy / normalizer,
        "expert_load_cv": coefficient_of_variation,
        "expert_load_cv_squared": coefficient_of_variation**2,
        "expert_max_load_ratio": max_load_ratio,
        "expert_max_violation": max_load_ratio - 1.0,
    }


def evaluate_predictions_and_load(
    model: BalancingSweepDeepFMMoE,
    evaluation_data: Iterable[Any],
) -> Dict[str, Any]:
    labels: List[np.ndarray] = []
    predictions: List[np.ndarray] = []
    counts = torch.zeros(model.moe_layers.num_experts, dtype=torch.float64)

    model.eval()
    with torch.no_grad():
        for batch in evaluation_data:
            interaction = _interaction(batch).to(_model_device(model))
            logits, meta = model.forward(interaction, return_gate=True)
            prediction = model.sigmoid(logits).reshape(-1)
            label = interaction[model.LABEL].reshape(-1)
            labels.append(label.detach().to(device="cpu", dtype=torch.float64).numpy())
            predictions.append(
                prediction.detach().to(device="cpu", dtype=torch.float64).numpy()
            )
            selected = meta["topk_idx"].detach().reshape(-1).to(device="cpu")
            counts += torch.bincount(
                selected,
                minlength=model.moe_layers.num_experts,
            ).to(dtype=torch.float64)

    all_labels = np.concatenate(labels)
    all_predictions = np.concatenate(predictions)
    result: Dict[str, Any] = {
        "test_samples": int(all_labels.size),
        "test_auc": float(roc_auc_score(all_labels, all_predictions)),
        "test_logloss": float(
            log_loss(all_labels, all_predictions, labels=[0.0, 1.0])
        ),
    }
    result.update(_load_statistics(counts))
    return result


__all__ = ["evaluate_predictions_and_load"]
