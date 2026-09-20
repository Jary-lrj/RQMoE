"""Configuration for training a shared Top-k=4 DeepFM_MoE checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import yaml

from .project import (
    Config,
    DeepFM_MoE,
    REPOSITORY_ROOT,
    use_local_atomic_data_path,
)


TRAINING_TOP_K = 4
DEFAULT_CONFIG = (
    REPOSITORY_ROOT
    / "additional_implementation"
    / "configs"
    / "deepfm_moe_topk4.yaml"
)


def parse_training_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train one DeepFM_MoE checkpoint with Top-k=4 using local RecBole "
            "atomic files and standard whole-split evaluation."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=(
            "Dataset/model YAML. The default covers rating-based atomic datasets; "
            "pass a dataset-specific YAML for an existing binary label field."
        ),
    )
    parser.add_argument("--dataset", required=True, help="RecBole dataset and directory name.")
    parser.add_argument(
        "--data-path",
        type=Path,
        required=True,
        help="Parent directory containing <dataset>/<dataset>.inter.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--cache-dataset",
        action="store_true",
        help="Save RecBole's filtered Dataset object for faster repeated evaluation.",
    )
    parser.add_argument(
        "--num-experts",
        type=int,
        default=8,
        help="Total expert count. The manuscript setting is 8.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        help=(
            "Output directory. Defaults to additional_implementation/checkpoints/"
            "<dataset>/experts_<num-experts>/seed_<seed>."
        ),
    )
    parser.add_argument("--gpu-id", type=str)
    parser.add_argument("--use-gpu", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--show-progress", action=argparse.BooleanOptionalAction, default=None)
    return parser.parse_args()


def validate_training_args(args: argparse.Namespace) -> Tuple[Path, Path, Path]:
    config_path = args.config.expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Training config does not exist: {config_path}")

    if not args.dataset or Path(args.dataset).name != args.dataset:
        raise ValueError("--dataset must be one directory name, such as beauty or ml-1m.")
    if args.num_experts < TRAINING_TOP_K:
        raise ValueError(
            f"--num-experts must be at least the training Top-k ({TRAINING_TOP_K})."
        )

    data_root = args.data_path.expanduser().resolve()
    atomic_data_path = data_root / args.dataset
    interaction_file = atomic_data_path / f"{args.dataset}.inter"
    if not interaction_file.is_file():
        raise FileNotFoundError(
            f"Missing local RecBole interaction file: {interaction_file}. "
            "The runner requires pre-existing atomic files and will not trigger RecBole's downloader."
        )

    if args.checkpoint_dir is None:
        checkpoint_dir = (
            REPOSITORY_ROOT
            / "additional_implementation"
            / "checkpoints"
            / args.dataset
            / f"experts_{args.num_experts}"
            / f"seed_{args.seed}"
        )
    else:
        checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    return config_path, data_root, checkpoint_dir


def write_resolved_config(
    args: argparse.Namespace,
    config_path: Path,
    data_root: Path,
    checkpoint_dir: Path,
) -> Path:
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
            "num_experts": args.num_experts,
            "top_k": TRAINING_TOP_K,
            "checkpoint_dir": str(checkpoint_dir),
        }
    )
    if args.cache_dataset:
        resolved["save_dataset"] = True
        resolved["dataset_save_path"] = str(
            checkpoint_dir / f"{args.dataset}-Dataset.pth"
        )
    if args.gpu_id is not None:
        resolved["gpu_id"] = args.gpu_id
    if args.use_gpu is not None:
        resolved["use_gpu"] = args.use_gpu
    if args.show_progress is not None:
        resolved["show_progress"] = args.show_progress

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    resolved_path = checkpoint_dir / "training_config.yaml"
    with resolved_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(resolved, stream, sort_keys=False)
    return resolved_path


def validate_training_config(config: Config, expected_seed: int) -> None:
    if int(config["top_k"]) != TRAINING_TOP_K:
        raise ValueError(f"Training requires top_k={TRAINING_TOP_K}.")
    if int(config["seed"]) != expected_seed:
        raise ValueError("The runtime seed was not preserved in the RecBole config.")
    if int(config["num_experts"]) < TRAINING_TOP_K:
        raise ValueError("The total expert count must be at least the training Top-k.")

    eval_args = config["eval_args"]
    mode = eval_args.get("mode") if isinstance(eval_args, Mapping) else None
    modes = mode.values() if isinstance(mode, Mapping) else [mode]
    if any(str(value).lower() != "labeled" for value in modes):
        raise ValueError("Training requires labeled validation and test evaluation.")

    metrics = config["metrics"]
    metric_names = {str(metric).lower() for metric in metrics}
    if not {"auc", "logloss"}.issubset(metric_names):
        raise ValueError("Training config must report both AUC and LogLoss.")


def prepare_training_config(args: argparse.Namespace) -> Tuple[Config, Path]:
    config_path, data_root, checkpoint_dir = validate_training_args(args)
    resolved_path = write_resolved_config(
        args=args,
        config_path=config_path,
        data_root=data_root,
        checkpoint_dir=checkpoint_dir,
    )
    config = Config(
        model=DeepFM_MoE,
        dataset=args.dataset,
        config_file_list=[str(resolved_path)],
    )
    use_local_atomic_data_path(config)
    validate_training_config(config, args.seed)
    return config, resolved_path
