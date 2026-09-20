"""CLI entry point for the controlled router-input intervention experiment."""

from __future__ import annotations

import argparse
import warnings
from typing import Dict

import torch

from .checkpointing import (
    CheckpointBundle,
    load_checkpoint_bundle,
    load_frozen_model,
    validate_checkpoint_config,
)
from .configuration import (
    build_config,
    parse_args,
    validate_args,
    validate_labeled_evaluation,
)
from .experiment import (
    build_conditions,
    evaluate_conditions,
    fit_router_spectrum,
    pair_delta_scale,
)
from .project import create_dataset, data_preparation, init_seed
from .reporting import write_results


def load_protocol_checkpoints(
    args: argparse.Namespace,
    protocol: str,
) -> Dict[int, CheckpointBundle]:
    if protocol == "shared_checkpoint":
        bundle = load_checkpoint_bundle(args.shared_checkpoint)
        if bundle.training_top_k is None:
            warnings.warn(
                "The shared checkpoint does not record its training Top-k. A k=4-trained checkpoint "
                "is recommended for the controlled inference Top-k comparison.",
                stacklevel=2,
            )
        elif bundle.training_top_k != 4:
            raise ValueError(
                f"The shared checkpoint records training Top-k={bundle.training_top_k}; "
                "use a k=4-trained checkpoint for the controlled protocol."
            )
        return {1: bundle}

    bundles = {
        1: load_checkpoint_bundle(args.checkpoint_k1),
        4: load_checkpoint_bundle(args.checkpoint_k4),
    }
    for expected_top_k, bundle in bundles.items():
        if bundle.training_top_k is not None and bundle.training_top_k != expected_top_k:
            raise ValueError(
                f"{bundle.path} records training Top-k={bundle.training_top_k}; "
                f"expected Top-k={expected_top_k}."
            )
    return bundles


def evaluate_checkpoint(
    bundle: CheckpointBundle,
    top_k: int,
    config,
    train_data,
    calibration_data,
    test_data,
    protocol: str,
    conditions,
    args: argparse.Namespace,
):
    model = load_frozen_model(config, train_data.dataset, bundle)
    statistics = fit_router_spectrum(
        model,
        calibration_data,
        maximum_samples=args.max_calibration_samples,
        center=args.center_spectrum,
    )
    rows = evaluate_conditions(
        model=model,
        test_data=test_data,
        statistics=statistics,
        checkpoint_path=bundle.path,
        checkpoint_training_top_k=bundle.training_top_k,
        checkpoint_saved_config_seed=bundle.saved_config_seed,
        top_k=top_k,
        protocol=protocol,
        seed=args.seed,
        conditions=conditions,
        args=args,
    )
    return model, rows


def run(args: argparse.Namespace) -> None:
    protocol = validate_args(args)
    config = build_config(args)
    validate_labeled_evaluation(config)
    checkpoint_bundles = load_protocol_checkpoints(args, protocol)
    for bundle in checkpoint_bundles.values():
        validate_checkpoint_config(config, bundle)

    init_seed(args.seed, config["reproducibility"])
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    calibration_data = train_data if args.calibration_split == "train" else valid_data
    conditions = build_conditions(args)
    rows = []

    if protocol == "shared_checkpoint":
        bundle = checkpoint_bundles[1]
        model = load_frozen_model(config, train_data.dataset, bundle)
        statistics = fit_router_spectrum(
            model,
            calibration_data,
            maximum_samples=args.max_calibration_samples,
            center=args.center_spectrum,
        )
        for top_k in (1, 4):
            rows.extend(
                evaluate_conditions(
                    model=model,
                    test_data=test_data,
                    statistics=statistics,
                    checkpoint_path=bundle.path,
                    checkpoint_training_top_k=bundle.training_top_k,
                    checkpoint_saved_config_seed=bundle.saved_config_seed,
                    top_k=top_k,
                    protocol=protocol,
                    seed=args.seed,
                    conditions=conditions,
                    args=args,
                )
            )
    else:
        for top_k in (1, 4):
            model, condition_rows = evaluate_checkpoint(
                bundle=checkpoint_bundles[top_k],
                top_k=top_k,
                config=config,
                train_data=train_data,
                calibration_data=calibration_data,
                test_data=test_data,
                protocol=protocol,
                conditions=conditions,
                args=args,
            )
            rows.extend(condition_rows)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    delta_rows = pair_delta_scale(rows)
    write_results(args.output_dir, rows, delta_rows, args, config, protocol)


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
