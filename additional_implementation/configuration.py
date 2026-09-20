"""Argument parsing and runtime configuration validation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Mapping

from .project import (
    Config,
    DeepFM_MoE,
    REPOSITORY_ROOT,
    use_local_atomic_data_path,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate router-only rank-collapse and spectral-flattening interventions "
            "without changing embeddings received by experts or the prediction path."
        )
    )
    parser.add_argument("--config", type=Path, required=True, help="Original RecBole YAML config.")
    parser.add_argument(
        "--shared-checkpoint",
        type=Path,
        help="One frozen checkpoint evaluated with inference Top-k 1 and 4.",
    )
    parser.add_argument("--checkpoint-k1", type=Path, help="Checkpoint trained with Top-k=1.")
    parser.add_argument("--checkpoint-k4", type=Path, help="Checkpoint trained with Top-k=4.")
    parser.add_argument("--retained-rank", type=int, required=True)
    parser.add_argument(
        "--gammas",
        nargs="+",
        type=float,
        default=[1.0, 0.75, 0.5, 0.25, 0.0],
    )
    parser.add_argument(
        "--flatten-strengths",
        nargs="+",
        type=float,
        default=[0.25, 0.5, 0.75, 1.0],
    )
    parser.add_argument("--spectral-floor-ratio", type=float, default=1e-3)
    parser.add_argument("--max-flatten-gain", type=float, default=100.0)
    parser.add_argument(
        "--center-spectrum",
        action="store_true",
        help="Center H before fitting its spectrum. The default follows uncentered H=UΣVᵀ.",
    )
    parser.add_argument(
        "--calibration-split",
        choices=["train", "valid"],
        default="train",
        help="Split used to fit V and singular values. Test data is intentionally unavailable here.",
    )
    parser.add_argument("--max-calibration-samples", type=int, default=200_000)
    parser.add_argument(
        "--seed",
        type=int,
        required=True,
        help="Actual CLI seed passed to run_SAG.py when the checkpoint was trained.",
    )
    parser.add_argument("--gpu-id", type=str)
    parser.add_argument("--use-gpu", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPOSITORY_ROOT / "additional_implementation" / "results",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> str:
    shared = args.shared_checkpoint is not None
    paired = args.checkpoint_k1 is not None or args.checkpoint_k4 is not None
    if shared == paired:
        raise ValueError(
            "Choose exactly one protocol: --shared-checkpoint, or both --checkpoint-k1 and --checkpoint-k4."
        )
    if paired and (args.checkpoint_k1 is None or args.checkpoint_k4 is None):
        raise ValueError("The paired protocol requires both --checkpoint-k1 and --checkpoint-k4.")

    paths = [args.config]
    paths.extend(
        path
        for path in (args.shared_checkpoint, args.checkpoint_k1, args.checkpoint_k4)
        if path is not None
    )
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required files: {missing}")
    if args.retained_rank < 1:
        raise ValueError("--retained-rank must be positive.")
    if args.max_calibration_samples < 1:
        raise ValueError("--max-calibration-samples must be positive.")
    if any(value < 0.0 or value > 1.0 for value in args.gammas):
        raise ValueError("All --gammas values must lie in [0, 1].")
    if any(value < 0.0 or value > 1.0 for value in args.flatten_strengths):
        raise ValueError("All --flatten-strengths values must lie in [0, 1].")
    return "shared_checkpoint" if shared else "paired_checkpoints"


def build_config(args: argparse.Namespace) -> Config:
    overrides: Dict[str, Any] = {"top_k": 1, "show_progress": False, "shuffle": False}
    if args.gpu_id is not None:
        overrides["gpu_id"] = args.gpu_id
    if args.use_gpu is not None:
        overrides["use_gpu"] = args.use_gpu
    config = Config(
        model=DeepFM_MoE,
        config_file_list=[str(args.config.resolve())],
        config_dict=overrides,
    )
    return use_local_atomic_data_path(config)


def config_value(config: Any, key: str) -> Any:
    try:
        return config[key]
    except (KeyError, TypeError):
        return None


def canonical_config_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): canonical_config_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [canonical_config_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def validate_labeled_evaluation(config: Config) -> None:
    eval_args = config_value(config, "eval_args")
    mode = eval_args.get("mode") if isinstance(eval_args, Mapping) else None
    if isinstance(mode, Mapping):
        labeled = all(str(value).lower() == "labeled" for value in mode.values())
    else:
        labeled = str(mode).lower() == "labeled"
    if not labeled:
        raise ValueError(
            "This runner computes pointwise AUC from interaction labels and requires "
            "eval_args.mode='labeled'."
        )


def validate_runtime_seed(config: Config, runtime_seed: int) -> None:
    configured_seed = config_value(config, "seed")
    if configured_seed is None:
        raise ValueError("The RecBole config must record the training/split seed.")
    if int(configured_seed) != runtime_seed:
        raise ValueError(
            f"--seed={runtime_seed} differs from config seed={configured_seed}. "
            "Use the resolved training_config.yaml stored beside the checkpoint."
        )
