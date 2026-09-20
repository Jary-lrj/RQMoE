"""Train a Top-k=4 checkpoint for controlled inference-time Top-k evaluation."""

from __future__ import annotations

import argparse
from logging import getLogger
from pathlib import Path
from typing import Tuple

from recbole.trainer import Trainer
from recbole.utils import init_logger

from .project import build_deepfm_moe, create_dataset, data_preparation, init_seed
from .training_configuration import (
    TRAINING_TOP_K,
    parse_training_args,
    prepare_training_config,
)


def run(args: argparse.Namespace) -> Tuple[Path, Path]:
    config, resolved_config = prepare_training_config(args)
    init_seed(config["seed"], config["reproducibility"])
    init_logger(config)
    logger = getLogger()
    logger.info(config)

    dataset = create_dataset(config)
    logger.info(dataset)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    model = build_deepfm_moe(config, train_data.dataset).to(config["device"])
    if model.moe_layers.top_k_eval != TRAINING_TOP_K:
        raise ValueError(
            f"Constructed model uses Top-k={model.moe_layers.top_k_eval}; "
            f"expected {TRAINING_TOP_K}."
        )
    logger.info(model)

    trainer = Trainer(config, model)
    best_valid_score, best_valid_result = trainer.fit(
        train_data,
        valid_data,
        saved=True,
        show_progress=config["show_progress"],
    )
    test_result = trainer.evaluate(
        test_data,
        load_best_model=True,
        show_progress=config["show_progress"],
    )

    checkpoint_path = Path(trainer.saved_model_file).resolve()
    logger.info("Best valid score: %s", best_valid_score)
    logger.info("Best valid result: %s", best_valid_result)
    logger.info("Whole-test result: %s", test_result)
    logger.info("Shared Top-k=4 checkpoint: %s", checkpoint_path)
    print(f"checkpoint={checkpoint_path}")
    print(f"resolved_config={resolved_config}")
    return checkpoint_path, resolved_config


def main() -> None:
    run(parse_training_args())


if __name__ == "__main__":
    main()
