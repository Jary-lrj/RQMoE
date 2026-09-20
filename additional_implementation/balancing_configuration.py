"""Argument parsing and RecBole configuration for the balancing sweep."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import yaml

from .balancing_model import BalancingSweepDeepFMMoE
from .project import Config, REPOSITORY_ROOT, use_local_atomic_data_path


DEFAULT_CONFIG = (
    REPOSITORY_ROOT
    / "additional_implementation"
    / "configs"
    / "load_balancing_beauty.yaml"
)


def add_shared_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset", default="beauty")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=REPOSITORY_ROOT / "exps" / "running" / "dataset",
        help="Parent directory containing <dataset>/<dataset>.inter.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu-id", default="2")
    parser.add_argument("--num-experts", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--show-progress", action="store_true")


def condition_name(method: str, alpha: float, loss_free_update_rate: float) -> str:
    if method == "loss_free":
        return f"loss_free_u_{loss_free_update_rate:g}"
    return f"alpha_{alpha:g}"


def validate_paths(args: argparse.Namespace) -> Tuple[Path, Path, Path]:
    config_path = args.config.expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing base config: {config_path}")
    if not args.dataset or Path(args.dataset).name != args.dataset:
        raise ValueError("--dataset must be one directory name.")
    data_root = args.data_path.expanduser().resolve()
    interaction_file = data_root / args.dataset / f"{args.dataset}.inter"
    if not interaction_file.is_file():
        raise FileNotFoundError(f"Missing RecBole interaction file: {interaction_file}")
    if not 1 <= args.top_k <= args.num_experts:
        raise ValueError("--top-k must lie in [1, --num-experts].")
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    return config_path, data_root, output_dir


def prepare_condition_config(
    args: argparse.Namespace,
) -> tuple[Config, Path, str]:
    config_path, data_root, output_root = validate_paths(args)
    name = condition_name(args.method, args.alpha, args.loss_free_update_rate)
    condition_dir = output_root / name
    checkpoint_dir = condition_dir / "checkpoints"
    condition_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    with config_path.open("r", encoding="utf-8") as stream:
        loaded = yaml.safe_load(stream) or {}
    if not isinstance(loaded, Mapping):
        raise TypeError(f"Expected a YAML mapping in {config_path}.")

    resolved: Dict[str, Any] = dict(loaded)
    resolved.update(
        {
            "dataset": args.dataset,
            "data_path": str(data_root),
            "atomic_data_path": str(data_root / args.dataset),
            "seed": args.seed,
            "gpu_id": str(args.gpu_id),
            "show_progress": args.show_progress,
            "checkpoint_dir": str(checkpoint_dir),
            "num_experts": args.num_experts,
            "top_k": args.top_k,
            "balancing_method": args.method,
            "aux_lambda": args.alpha if args.method == "auxiliary" else 0.0,
            "loss_free_update_rate": args.loss_free_update_rate,
        }
    )
    resolved_path = condition_dir / "resolved_config.yaml"
    with resolved_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(resolved, stream, sort_keys=False)

    config = Config(
        model=BalancingSweepDeepFMMoE,
        dataset=args.dataset,
        config_file_list=[str(resolved_path)],
    )
    use_local_atomic_data_path(config)
    return config, resolved_path, name


__all__ = [
    "DEFAULT_CONFIG",
    "add_shared_arguments",
    "condition_name",
    "prepare_condition_config",
]
