import argparse
import csv
import gc
import json
import time
from logging import getLogger
from pathlib import Path
from typing import Any, Dict, Iterable, List

import torch

from smoe import SparseMoE

try:
    from recbole.model.context_aware_recommender import AutoInt, DCNV2, DeepFM
    from recbole.model.init import xavier_normal_initialization

    HAS_RECBOLE = True
except ModuleNotFoundError:
    AutoInt = DCNV2 = DeepFM = object
    HAS_RECBOLE = False

    def xavier_normal_initialization(module):
        return module


ROOT = Path(__file__).resolve().parent

DATASET_CONFIGS = {
    "Avazu": ROOT / "exps" / "running" / "avazu.yaml",
    "avazu": ROOT / "exps" / "running" / "avazu.yaml",
    "Beauty": ROOT / "exps" / "running" / "beauty.yaml",
    "MovieLens-1M": ROOT / "exps" / "running" / "ml-1m.yaml",
    "Movielens-1M": ROOT / "exps" / "running" / "ml-1m.yaml",
    "ml-1m": ROOT / "exps" / "running" / "ml-1m.yaml",
}


def cfg_get(config: Any, key: str, default: Any = None) -> Any:
    try:
        return config[key]
    except KeyError:
        return default


def expert_hidden_sizes(config: Any) -> List[int]:
    hidden = list(cfg_get(config, "moe_hidden_size", []) or [])
    if hidden:
        return hidden
    mlp_hidden_size = list(config["mlp_hidden_size"])
    return mlp_hidden_size[:-1]


def build_smoe(config: Any, input_size: int, output_size: int, dropout: float) -> SparseMoE:
    return SparseMoE(
        input_size=input_size,
        output_size=output_size,
        num_experts=int(config["moe_num_experts"]),
        hidden_sizes=expert_hidden_sizes(config),
        top_k_eval=int(config["moe_top_k"]),
        dropout=dropout,
        activation=cfg_get(config, "moe_activation", "relu"),
        noisy_gating=bool(cfg_get(config, "moe_noisy_gating", False)),
    )


MODEL_DEFAULTS = {
    "DeepFM": {},
    "AutoInt": {
        "attention_size": 16,
        "dropout_probs": [0.2, 0.2, 0.2],
        "n_layers": 2,
        "num_heads": 2,
        "has_residual": True,
    },
    "DCNv2": {
        "mixed": False,
        "structure": "parallel",
        "cross_layer_num": 2,
        "expert_num": 4,
        "low_rank": 32,
        "reg_weight": 1e-5,
    },
}


class DeepFMSparseMoE(DeepFM):
    """DeepFM with the deep MLP branch replaced by SparseMoE."""

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        input_size = self.embedding_size * self.num_feature_field
        self.mlp_layers = build_smoe(
            config=config,
            input_size=input_size,
            output_size=self.mlp_hidden_size[-1],
            dropout=self.dropout_prob,
        )
        self.mlp_layers.apply(self._init_weights)


class DCNV2SparseMoE(DCNV2):
    """DCNv2 with the deep MLP branch replaced by SparseMoE."""

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.mlp_layers = build_smoe(
            config=config,
            input_size=self.in_feature_num,
            output_size=self.mlp_hidden_size[-1],
            dropout=self.dropout_prob,
        )
        self.mlp_layers.apply(xavier_normal_initialization)


class AutoIntSparseMoE(AutoInt):
    """AutoInt with the deep MLP branch replaced by SparseMoE."""

    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.mlp_layers = build_smoe(
            config=config,
            input_size=self.embed_output_dim,
            output_size=self.mlp_hidden_size[-1],
            dropout=self.dropout_probs[1],
        )
        self.mlp_layers.apply(self._init_weights)


MODEL_CLASSES = {
    "DeepFM": DeepFMSparseMoE,
    "DCNv2": DCNV2SparseMoE,
    "DCNV2": DCNV2SparseMoE,
    "AutoInt": AutoIntSparseMoE,
}


def normalize_model_name(name: str) -> str:
    if name == "DCNV2":
        return "DCNv2"
    return name


def normalize_dataset_name(name: str) -> str:
    if name == "avazu":
        return "Avazu"
    if name == "Movielens-1M":
        return "MovieLens-1M"
    if name == "ml-1m":
        return "MovieLens-1M"
    return name


def build_config_dict(args: argparse.Namespace, model_name: str, active_experts: int) -> Dict[str, Any]:
    overrides = dict(MODEL_DEFAULTS.get(model_name, {}))
    overrides.update(
        {
            "top_k": active_experts,
            "moe_top_k": active_experts,
            "moe_num_experts": args.num_experts,
            "moe_noisy_gating": args.noisy_gating,
            "moe_activation": args.activation,
            "show_progress": args.show_progress,
        }
    )

    if args.epochs is not None:
        overrides["epochs"] = args.epochs
    if args.seed is not None:
        overrides["seed"] = args.seed
    if args.gpu_id is not None:
        overrides["gpu_id"] = args.gpu_id
    if args.use_gpu is not None:
        overrides["use_gpu"] = args.use_gpu

    return overrides


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def append_csv(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flat = {
        "dataset": row["dataset"],
        "model": row["model"],
        "num_experts": row["num_experts"],
        "active_experts": row["active_experts"],
        "best_valid_score": row["best_valid_score"],
        "elapsed_sec": row["elapsed_sec"],
    }
    for prefix in ("best_valid_result", "test_result"):
        for key, value in row.get(prefix, {}).items():
            flat[f"{prefix}.{key}"] = value

    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(flat)


def run_one(
    args: argparse.Namespace,
    dataset_name: str,
    model_name: str,
    active_experts: int,
    jsonl_path: Path,
    csv_path: Path,
) -> Dict[str, Any]:
    if not HAS_RECBOLE:
        raise ModuleNotFoundError("RecBole is required to run experiments. Install recbole or run with --dry-run only.")

    from recbole.config import Config
    from recbole.data import create_dataset, data_preparation
    from recbole.trainer import Trainer
    from recbole.utils import init_logger, init_seed

    model_name = normalize_model_name(model_name)
    dataset_name = normalize_dataset_name(dataset_name)
    model_class = MODEL_CLASSES[model_name]
    config_path = DATASET_CONFIGS[dataset_name]
    config_dict = build_config_dict(args, model_name, active_experts)

    config = Config(
        model=model_class,
        config_file_list=[str(config_path)],
        config_dict=config_dict,
    )
    init_seed(config["seed"], config["reproducibility"])
    init_logger(config)
    logger = getLogger()
    logger.info(
        "[ScalingFailure] dataset=%s model=%s num_experts=%s active_experts=%s",
        dataset_name,
        model_name,
        args.num_experts,
        active_experts,
    )

    start = time.time()
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    model = model_class(config, train_data.dataset).to(config["device"])
    trainer = Trainer(config, model)

    best_valid_score, best_valid_result = trainer.fit(
        train_data,
        valid_data,
        show_progress=args.show_progress,
    )
    test_result = trainer.evaluate(test_data, show_progress=args.show_progress)

    row = {
        "dataset": dataset_name,
        "model": model_name,
        "model_class": model_class.__name__,
        "config_file": str(config_path),
        "num_experts": args.num_experts,
        "active_experts": active_experts,
        "moe_train_sparse": True,
        "moe_eval_sparse": True,
        "best_valid_score": float(best_valid_score),
        "best_valid_result": best_valid_result,
        "test_result": test_result,
        "elapsed_sec": round(time.time() - start, 4),
    }
    logger.info("[ScalingFailure] result=%s", row)
    append_jsonl(jsonl_path, row)
    append_csv(csv_path, row)

    del trainer, model, train_data, valid_data, test_data, dataset
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SparseMoE scaling-failure experiments on RecBole context models."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["DeepFM", "DCNv2", "AutoInt"],
        choices=["DeepFM", "DCNv2", "DCNV2", "AutoInt"],
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["Avazu"],
        choices=["Avazu", "avazu", "Beauty", "MovieLens-1M", "Movielens-1M", "ml-1m"],
    )
    parser.add_argument("--active-experts", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    parser.add_argument("--num-experts", type=int, default=16)
    parser.add_argument("--activation", type=str, default="relu")
    parser.add_argument("--noisy-gating", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--gpu-id", type=str, default=None)
    parser.add_argument("--use-gpu", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--show-progress", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "exps" / "running" / "scaling_failure")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    bad_k = [k for k in args.active_experts if k < 1 or k > args.num_experts]
    if bad_k:
        raise ValueError(f"active experts must be in [1, {args.num_experts}], got {bad_k}")

    for dataset_name in args.datasets:
        normalized = normalize_dataset_name(dataset_name)
        if normalized not in DATASET_CONFIGS:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        if not DATASET_CONFIGS[normalized].exists():
            raise FileNotFoundError(DATASET_CONFIGS[normalized])


def experiment_grid(args: argparse.Namespace) -> Iterable[tuple]:
    for dataset_name in args.datasets:
        for model_name in args.models:
            for active_experts in args.active_experts:
                yield normalize_dataset_name(dataset_name), normalize_model_name(model_name), active_experts


def main() -> None:
    args = parse_args()
    validate_args(args)

    jsonl_path = args.output_dir / "results.jsonl"
    csv_path = args.output_dir / "results.csv"
    grid = list(experiment_grid(args))

    if args.dry_run:
        for dataset_name, model_name, active_experts in grid:
            print(
                f"dataset={dataset_name} model={model_name} "
                f"num_experts={args.num_experts} active_experts={active_experts}"
            )
        return

    for dataset_name, model_name, active_experts in grid:
        run_one(
            args=args,
            dataset_name=dataset_name,
            model_name=model_name,
            active_experts=active_experts,
            jsonl_path=jsonl_path,
            csv_path=csv_path,
        )


if __name__ == "__main__":
    main()
