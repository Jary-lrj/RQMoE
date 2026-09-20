"""Train and evaluate one auxiliary-loss or Loss-Free balancing condition."""

from __future__ import annotations

import argparse
import json
import time
from logging import getLogger
from pathlib import Path
from typing import Any, Dict

import torch
from recbole.trainer import Trainer
from recbole.utils import init_logger

from .balancing_configuration import add_shared_arguments, prepare_condition_config
from .balancing_metrics import evaluate_predictions_and_load
from .balancing_model import BalancingSweepDeepFMMoE
from .project import create_dataset, data_preparation, init_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_shared_arguments(parser)
    parser.add_argument(
        "--method",
        choices=["auxiliary", "loss_free"],
        required=True,
    )
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--loss-free-update-rate", type=float, default=1e-3)
    args = parser.parse_args()
    if args.alpha < 0:
        parser.error("--alpha must be non-negative.")
    if args.loss_free_update_rate <= 0:
        parser.error("--loss-free-update-rate must be positive.")
    if args.method == "loss_free" and args.alpha != 0:
        parser.error("Loss-Free Balancing requires --alpha=0.")
    return args


def _plain_metrics(values: Dict[str, Any]) -> Dict[str, float]:
    return {str(key): float(value) for key, value in values.items()}


def _checkpoint_epoch(path: Path) -> int | None:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    epoch = checkpoint.get("epoch") if isinstance(checkpoint, dict) else None
    return int(epoch) if epoch is not None else None


def run(args: argparse.Namespace) -> Path:
    config, resolved_path, name = prepare_condition_config(args)
    condition_dir = resolved_path.parent
    init_seed(config["seed"], config["reproducibility"])
    init_logger(config)
    logger = getLogger()
    logger.info(config)

    started = time.perf_counter()
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    model = BalancingSweepDeepFMMoE(config, train_data.dataset).to(config["device"])
    trainer = Trainer(config, model)
    best_valid_score, best_valid_result = trainer.fit(
        train_data,
        valid_data,
        saved=True,
        show_progress=config["show_progress"],
    )
    recbole_test_result = trainer.evaluate(
        test_data,
        load_best_model=True,
        show_progress=config["show_progress"],
    )
    evaluation = evaluate_predictions_and_load(model, test_data)
    elapsed_seconds = time.perf_counter() - started

    checkpoint_path = Path(trainer.saved_model_file).resolve()
    result: Dict[str, Any] = {
        "condition": name,
        "balancing_method": args.method,
        "alpha": args.alpha if args.method == "auxiliary" else None,
        "loss_free_update_rate": (
            args.loss_free_update_rate if args.method == "loss_free" else None
        ),
        "loss_free_update_rule": (
            "sign_load_error" if args.method == "loss_free" else None
        ),
        "dataset": args.dataset,
        "seed": args.seed,
        "num_experts": args.num_experts,
        "top_k": args.top_k,
        "best_epoch": _checkpoint_epoch(checkpoint_path),
        "best_valid_score": float(best_valid_score),
        "best_valid_result": _plain_metrics(best_valid_result),
        "recbole_test_result": _plain_metrics(recbole_test_result),
        "checkpoint": str(checkpoint_path),
        "resolved_config": str(resolved_path),
        "elapsed_seconds": elapsed_seconds,
        "loss_free_bias": [
            float(value)
            for value in model.moe_layers.loss_free_bias.detach().cpu().tolist()
        ],
        "loss_free_update_count": int(
            model.moe_layers.loss_free_update_count.detach().cpu().item()
        ),
        **evaluation,
    }
    result_path = condition_dir / "result.json"
    with result_path.open("w", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    logger.info("Balancing result: %s", result)
    print(f"result={result_path}")
    return result_path


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
