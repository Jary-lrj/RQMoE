"""Write controlled-intervention result artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .project import Config


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value) if isinstance(value, (list, dict)) else value
                    for key, value in row.items()
                }
            )


def write_results(
    output_dir: Path,
    rows: Sequence[Mapping[str, Any]],
    delta_rows: Sequence[Mapping[str, Any]],
    args: argparse.Namespace,
    config: Config,
    protocol: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "condition_results.jsonl", rows)
    write_csv(output_dir / "condition_results.csv", rows)
    write_csv(output_dir / "delta_scale.csv", delta_rows)
    with (output_dir / "run_config.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "protocol": protocol,
                "config": str(args.config.resolve()),
                "calibration_split": args.calibration_split,
                "maximum_calibration_samples": args.max_calibration_samples,
                "retained_rank": args.retained_rank,
                "gammas": args.gammas,
                "flatten_strengths": args.flatten_strengths,
                "center_spectrum": args.center_spectrum,
                "spectral_floor_ratio": args.spectral_floor_ratio,
                "max_flatten_gain": args.max_flatten_gain,
                "runtime_split_seed": args.seed,
                "yaml_config_seed": config["seed"],
                "shared_checkpoint": (
                    str(args.shared_checkpoint.resolve()) if args.shared_checkpoint is not None else None
                ),
                "checkpoint_k1": (
                    str(args.checkpoint_k1.resolve()) if args.checkpoint_k1 is not None else None
                ),
                "checkpoint_k4": (
                    str(args.checkpoint_k4.resolve()) if args.checkpoint_k4 is not None else None
                ),
            },
            handle,
            indent=2,
        )
