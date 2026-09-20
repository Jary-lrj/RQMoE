"""Controlled-intervention computation and paired metric assembly."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from sklearn.metrics import log_loss, roc_auc_score

from .metrics import RoutingDiagnostics
from .project import DeepFM_MoE
from .router_hook import RouterOnlyIntervention
from .spectral_intervention import (
    SecondMomentAccumulator,
    SpectralFlatteningIntervention,
    SpectralStatistics,
    TailCollapseIntervention,
)


@dataclass(frozen=True)
class InterventionCondition:
    kind: str
    value: float

    @property
    def name(self) -> str:
        if self.kind == "tail_collapse":
            return f"tail_collapse_gamma_{self.value:g}"
        return f"spectral_flattening_strength_{self.value:g}"


def batch_interaction(batch: Any) -> Any:
    return batch[0] if isinstance(batch, (tuple, list)) else batch


def model_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def fit_router_spectrum(
    model: DeepFM_MoE,
    calibration_data: Iterable[Any],
    maximum_samples: int,
    center: bool,
) -> SpectralStatistics:
    dimension = int(model.moe_layers.input_size)
    accumulator = SecondMomentAccumulator(dimension, center=center)
    collected = 0
    with torch.no_grad():
        for batch in calibration_data:
            interaction = batch_interaction(batch).to(model_device(model))
            embeddings = model.concat_embed_input_fields(interaction)
            router_input = embeddings.reshape(embeddings.shape[0], -1)
            remaining = maximum_samples - collected
            if router_input.shape[0] > remaining:
                router_input = router_input[:remaining]
            accumulator.update(router_input)
            collected += router_input.shape[0]
            if collected >= maximum_samples:
                break
    return accumulator.finalize()


def build_conditions(args: argparse.Namespace) -> List[InterventionCondition]:
    conditions = [InterventionCondition("tail_collapse", value) for value in args.gammas]
    conditions.extend(
        InterventionCondition("spectral_flattening", value)
        for value in args.flatten_strengths
    )
    return conditions


def build_intervention(
    condition: InterventionCondition,
    statistics: SpectralStatistics,
    args: argparse.Namespace,
) -> torch.nn.Module:
    if condition.kind == "tail_collapse":
        return TailCollapseIntervention(
            statistics=statistics,
            retained_rank=args.retained_rank,
            gamma=condition.value,
        )
    return SpectralFlatteningIntervention(
        statistics=statistics,
        strength=condition.value,
        floor_ratio=args.spectral_floor_ratio,
        max_gain=args.max_flatten_gain,
    )


def evaluate_auc_and_logloss(model: DeepFM_MoE, test_data: Iterable[Any]) -> Dict[str, float]:
    labels: List[np.ndarray] = []
    predictions: List[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch in test_data:
            interaction = batch_interaction(batch).to(model_device(model))
            prediction = model.predict(interaction).reshape(-1)
            label = interaction[model.LABEL].reshape(-1)
            predictions.append(prediction.detach().to(device="cpu", dtype=torch.float64).numpy())
            labels.append(label.detach().to(device="cpu", dtype=torch.float64).numpy())

    all_labels = np.concatenate(labels)
    all_predictions = np.concatenate(predictions)
    return {
        "auc": float(roc_auc_score(all_labels, all_predictions)),
        "logloss": float(log_loss(all_labels, all_predictions, labels=[0.0, 1.0])),
    }


def evaluate_conditions(
    model: DeepFM_MoE,
    test_data: Iterable[Any],
    statistics: SpectralStatistics,
    checkpoint_path: Path,
    checkpoint_training_top_k: int | None,
    checkpoint_saved_config_seed: int | None,
    top_k: int,
    protocol: str,
    seed: int,
    conditions: Sequence[InterventionCondition],
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    moe_layer = model.moe_layers
    if top_k < 1 or top_k > moe_layer.num_experts:
        raise ValueError(f"Top-k={top_k} is invalid for {moe_layer.num_experts} experts.")
    if args.retained_rank > statistics.dimension:
        raise ValueError(
            f"--retained-rank={args.retained_rank} exceeds router dimension {statistics.dimension}."
        )
    moe_layer.top_k_eval = top_k

    rows: List[Dict[str, Any]] = []
    for condition in conditions:
        intervention = build_intervention(condition, statistics, args).to(model_device(model))
        diagnostics = RoutingDiagnostics(
            dimension=statistics.dimension,
            num_experts=moe_layer.num_experts,
            top_k=top_k,
            center_spectrum=args.center_spectrum,
        )
        with RouterOnlyIntervention(moe_layer, intervention, diagnostics):
            evaluation = evaluate_auc_and_logloss(model, test_data)

        rows.append(
            {
                "protocol": protocol,
                "runtime_split_seed": seed,
                "checkpoint": str(checkpoint_path.resolve()),
                "checkpoint_training_top_k": checkpoint_training_top_k,
                "checkpoint_saved_config_seed": checkpoint_saved_config_seed,
                "top_k": top_k,
                "num_experts": moe_layer.num_experts,
                "router_dimension": statistics.dimension,
                "calibration_samples": statistics.count,
                "spectrum_centered": statistics.centered,
                "retained_rank": args.retained_rank,
                "condition": condition.name,
                "intervention_kind": condition.kind,
                "intervention_value": condition.value,
                **evaluation,
                **diagnostics.finalize(),
            }
        )
    return rows


def pair_delta_scale(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, float], Dict[int, Dict[str, Any]]] = {}
    for row in rows:
        key = (row["intervention_kind"], float(row["intervention_value"]))
        grouped.setdefault(key, {})[int(row["top_k"])] = row

    paired_rows: List[Dict[str, Any]] = []
    for (kind, value), by_top_k in grouped.items():
        if set(by_top_k) != {1, 4}:
            raise ValueError(f"Expected Top-k 1 and 4 for {(kind, value)}, found {sorted(by_top_k)}.")
        k1 = by_top_k[1]
        k4 = by_top_k[4]
        paired_rows.append(
            {
                "protocol": k1["protocol"],
                "runtime_split_seed": k1["runtime_split_seed"],
                "intervention_kind": kind,
                "intervention_value": value,
                "retained_rank": k1["retained_rank"],
                "auc_k1": k1["auc"],
                "auc_k4": k4["auc"],
                "delta_scale": k4["auc"] - k1["auc"],
                "effective_rank_k1": k1["effective_rank"],
                "effective_rank_k4": k4["effective_rank"],
                "top1_load_entropy_k1": k1["top1_load_normalized_entropy"],
                "top1_load_entropy_k4": k4["top1_load_normalized_entropy"],
                "top1_load_cv_squared_k1": k1["top1_load_cv_squared"],
                "top1_load_cv_squared_k4": k4["top1_load_cv_squared"],
                "mean_router_entropy_k1": k1["mean_normalized_router_entropy"],
                "mean_router_entropy_k4": k4["mean_normalized_router_entropy"],
                "gate_weight_load_entropy_k1": k1["gate_weight_load_normalized_entropy"],
                "gate_weight_load_entropy_k4": k4["gate_weight_load_normalized_entropy"],
                "gate_weight_load_cv_squared_k1": k1["gate_weight_load_cv_squared"],
                "gate_weight_load_cv_squared_k4": k4["gate_weight_load_cv_squared"],
                "checkpoint_k1": k1["checkpoint"],
                "checkpoint_k4": k4["checkpoint"],
            }
        )
    return paired_rows
