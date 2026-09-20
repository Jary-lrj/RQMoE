"""Checkpoint loading and compatibility checks for the original model."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from .configuration import canonical_config_value, config_value
from .project import Config, DeepFM_MoE, build_deepfm_moe


@dataclass(frozen=True)
class CheckpointBundle:
    path: Path
    payload: Mapping[str, Any]
    training_top_k: int | None
    saved_config_seed: int | None


def load_checkpoint_bundle(path: Path) -> CheckpointBundle:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"Expected a checkpoint mapping, got {type(checkpoint).__name__}.")

    state_dict = checkpoint.get("state_dict", checkpoint)
    required_keys = {"moe_layers.gate.weight", "moe_layers.experts.0.1.weight"}
    missing_keys = required_keys.difference(state_dict)
    if missing_keys:
        raise ValueError(
            f"{path} is not a Switch.DeepFM_MoE checkpoint; missing state keys {sorted(missing_keys)}. "
            "Use additional_implementation.train_shared_checkpoint or "
            "exps/running/run_SAG.py --model DeepFM_MoE."
        )

    saved_config = checkpoint.get("config")
    top_k = config_value(saved_config, "top_k")
    seed = config_value(saved_config, "seed")
    return CheckpointBundle(
        path=path,
        payload=checkpoint,
        training_top_k=int(top_k) if top_k is not None else None,
        saved_config_seed=int(seed) if seed is not None else None,
    )


def validate_checkpoint_config(runtime_config: Config, bundle: CheckpointBundle) -> None:
    saved_config = bundle.payload.get("config")
    if saved_config is None:
        warnings.warn(
            f"{bundle.path} does not contain a saved RecBole Config; dataset/schema compatibility "
            "cannot be verified.",
            stacklevel=2,
        )
        return

    critical_keys = (
        "dataset",
        "data_path",
        "atomic_data_path",
        "USER_ID_FIELD",
        "ITEM_ID_FIELD",
        "TIME_FIELD",
        "TIMESTAMP_FIELD",
        "LABEL_FIELD",
        "RATING_FIELD",
        "load_col",
        "unused_col",
        "threshold",
        "filter_inter_by_user_or_item",
        "user_inter_num_interval",
        "item_inter_num_interval",
        "val_interval",
        "normalize_all",
        "numerical_features",
        "eval_args",
        "repeatable",
        "embedding_size",
        "mlp_hidden_size",
        "dropout_prob",
        "num_experts",
        "seed",
    )
    mismatches = []
    for key in critical_keys:
        saved_value = config_value(saved_config, key)
        runtime_value = config_value(runtime_config, key)
        if saved_value is None or runtime_value is None:
            continue
        if canonical_config_value(saved_value) != canonical_config_value(runtime_value):
            mismatches.append(key)
    if mismatches:
        raise ValueError(
            f"Runtime YAML settings differ from checkpoint {bundle.path} for {mismatches}. "
            "Use the same YAML config; supply the historical run_SAG.py CLI seed via --seed."
        )


def load_frozen_model(config: Config, dataset: Any, bundle: CheckpointBundle) -> DeepFM_MoE:
    model = build_deepfm_moe(config, dataset).to(config["device"])
    state_dict = bundle.payload.get("state_dict", bundle.payload)
    model.load_state_dict(state_dict, strict=True)
    other_parameter = bundle.payload.get("other_parameter")
    if other_parameter is not None and hasattr(model, "load_other_parameter"):
        model.load_other_parameter(other_parameter)
    model.requires_grad_(False)
    model.eval()
    return model
