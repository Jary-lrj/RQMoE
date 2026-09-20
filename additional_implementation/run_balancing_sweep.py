"""Run the full alpha grid plus a Loss-Free controlled baseline."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import List

from .balancing_configuration import DEFAULT_CONFIG
from .balancing_report import generate_report
from .project import REPOSITORY_ROOT


DEFAULT_ALPHAS = [0.0, 1e-3, 1e-2, 1e-1, 1.0, 10.0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--dataset", default="beauty")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=REPOSITORY_ROOT / "exps" / "running" / "dataset",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46])
    parser.add_argument("--alphas", type=float, nargs="+", default=DEFAULT_ALPHAS)
    parser.add_argument("--gpu-id", default="2")
    parser.add_argument("--num-experts", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--loss-free-update-rate", type=float, default=1e-3)
    parser.add_argument(
        "--include-loss-free",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--show-progress", action="store_true")
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Retrain conditions whose result.json already exists.",
    )
    args = parser.parse_args()
    if any(alpha < 0 for alpha in args.alphas):
        parser.error("All alpha values must be non-negative.")
    if len(set(args.alphas)) != len(args.alphas):
        parser.error("--alphas contains duplicates.")
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("--seeds contains duplicates.")
    return args


def _condition_specs(args: argparse.Namespace) -> List[tuple[str, float]]:
    specifications = [("auxiliary", alpha) for alpha in args.alphas]
    if args.include_loss_free:
        specifications.append(("loss_free", 0.0))
    return specifications


def _name(method: str, alpha: float, update_rate: float) -> str:
    return (
        f"loss_free_u_{update_rate:g}"
        if method == "loss_free"
        else f"alpha_{alpha:g}"
    )


def _command(
    args: argparse.Namespace,
    seed: int,
    seed_dir: Path,
    method: str,
    alpha: float,
) -> list[str]:
    command = [
        sys.executable,
        "-B",
        "-m",
        "additional_implementation.run_balancing_condition",
        "--config",
        str(args.config.expanduser().resolve()),
        "--dataset",
        args.dataset,
        "--data-path",
        str(args.data_path.expanduser().resolve()),
        "--output-dir",
        str(seed_dir),
        "--seed",
        str(seed),
        "--gpu-id",
        str(args.gpu_id),
        "--num-experts",
        str(args.num_experts),
        "--top-k",
        str(args.top_k),
        "--method",
        method,
        "--alpha",
        str(alpha),
        "--loss-free-update-rate",
        str(args.loss_free_update_rate),
    ]
    if args.show_progress:
        command.append("--show-progress")
    return command


def run(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "dataset": args.dataset,
        "seeds": args.seeds,
        "alphas": args.alphas,
        "include_loss_free": args.include_loss_free,
        "loss_free_update_rate": args.loss_free_update_rate,
        "num_experts": args.num_experts,
        "top_k": args.top_k,
        "gpu_id": str(args.gpu_id),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )

    for seed in args.seeds:
        seed_dir = output_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        for method, alpha in _condition_specs(args):
            name = _name(method, alpha, args.loss_free_update_rate)
            result_path = seed_dir / name / "result.json"
            if result_path.is_file() and not args.rerun:
                print(f"skip seed={seed} condition={name}", flush=True)
                continue
            log_path = seed_dir / f"{name}.log"
            print(f"start seed={seed} condition={name}", flush=True)
            with log_path.open("w", encoding="utf-8") as log_stream:
                subprocess.run(
                    _command(args, seed, seed_dir, method, alpha),
                    cwd=REPOSITORY_ROOT,
                    stdout=log_stream,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            print(f"done seed={seed} condition={name}", flush=True)

    generate_report(output_dir)
    print(f"report={output_dir / 'RESULTS.md'}", flush=True)
    print(f"figure={output_dir / 'balancing_sweep.html'}", flush=True)


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
