import argparse
import csv
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, Iterable, List, Tuple

cache_root = Path(tempfile.gettempdir())
mpl_config_dir = cache_root / "matplotlib"
xdg_cache_dir = cache_root / "fontconfig-cache"
mpl_config_dir.mkdir(parents=True, exist_ok=True)
xdg_cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))
os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache_dir))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "exps" / "running" / "scaling_failure" / "results.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "exps" / "running" / "scaling_failure"


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)

    records = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no}: {path}") from exc
    return records


def metric_value(result: Dict[str, Any], metric: str) -> float:
    metric_lower = metric.lower()
    for key, value in result.items():
        if key.lower() == metric_lower:
            return float(value)
    raise KeyError(f"Metric {metric!r} not found in result keys: {list(result.keys())}")


def record_top_k(record: Dict[str, Any]) -> int:
    if "active_experts" in record:
        return int(record["active_experts"])
    if "top_k" in record:
        return int(record["top_k"])
    if "moe_top_k" in record:
        return int(record["moe_top_k"])
    raise KeyError("No top-k field found. Expected active_experts, top_k, or moe_top_k.")


def grouped_auc(records: Iterable[Dict[str, Any]], result_field: str, metric: str) -> Dict[Tuple[str, str, int], List[float]]:
    values = defaultdict(list)
    for record in records:
        if result_field not in record:
            continue
        dataset = str(record["dataset"])
        model = str(record["model"])
        top_k = record_top_k(record)
        auc = metric_value(record[result_field], metric)
        values[(dataset, model, top_k)].append(auc)
    return values


def build_delta_rows(
    values: Dict[Tuple[str, str, int], List[float]],
) -> List[Dict[str, Any]]:
    means = {key: mean(v) for key, v in values.items()}
    rows = []
    series = defaultdict(list)

    for dataset, model, top_k in sorted(means):
        series[(dataset, model)].append(top_k)

    for dataset, model in sorted(series):
        previous_top_k = None
        previous_auc = None
        for top_k in sorted(series[(dataset, model)]):
            current_auc = means[(dataset, model, top_k)]
            raw_values = values[(dataset, model, top_k)]
            if previous_auc is None:
                delta_from_top_k = None
                delta_from_auc = None
                auc_delta = 0.0
            else:
                delta_from_top_k = previous_top_k
                delta_from_auc = previous_auc
                auc_delta = current_auc - previous_auc

            rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "top_k": top_k,
                    "auc": current_auc,
                    "auc_std": stdev(raw_values) if len(raw_values) > 1 else 0.0,
                    "runs": len(raw_values),
                    "delta_from_top_k": delta_from_top_k,
                    "delta_from_auc": delta_from_auc,
                    "auc_delta": auc_delta,
                }
            )
            previous_top_k = top_k
            previous_auc = current_auc

    return rows


def write_delta_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "model",
        "top_k",
        "auc",
        "auc_std",
        "runs",
        "delta_from_top_k",
        "delta_from_auc",
        "auc_delta",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_delta_auc(rows: List[Dict[str, Any]], output_path: Path, title: str) -> None:
    if not rows:
        raise ValueError("No plottable rows.")

    datasets = sorted({row["dataset"] for row in rows})
    models = sorted({row["model"] for row in rows})

    fig_width = max(6, 5 * len(datasets))
    fig, axes = plt.subplots(1, len(datasets), figsize=(fig_width, 4.5), squeeze=False, sharey=True)
    axes = axes[0]

    for ax, dataset in zip(axes, datasets):
        dataset_rows = [row for row in rows if row["dataset"] == dataset]
        for model in models:
            model_rows = sorted(
                [row for row in dataset_rows if row["model"] == model],
                key=lambda row: row["top_k"],
            )
            if not model_rows:
                continue
            x = [row["top_k"] for row in model_rows]
            y = [row["auc_delta"] for row in model_rows]
            ax.plot(x, y, marker="o", linewidth=2, label=model)

        ax.axhline(0.0, color="black", linewidth=1, linestyle="--", alpha=0.5)
        ax.set_title(dataset)
        ax.set_xlabel("Active experts (top-k)")
        ax.set_xticks(sorted({row["top_k"] for row in dataset_rows}))
        ax.grid(True, axis="y", alpha=0.25)

    axes[0].set_ylabel("AUC change from previous top-k")
    handles, labels = axes[-1].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=min(len(labels), 3), frameon=False)

    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot adjacent AUC deltas by SparseMoE top-k from scaling_failure JSONL."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--result-field", type=str, default="test_result", choices=["test_result", "best_valid_result"])
    parser.add_argument("--metric", type=str, default="AUC")
    parser.add_argument("--png-name", type=str, default="auc_delta_by_topk.png")
    parser.add_argument("--csv-name", type=str, default="auc_delta_by_topk.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_jsonl(args.input)
    values = grouped_auc(records, result_field=args.result_field, metric=args.metric)
    rows = build_delta_rows(values)

    csv_path = args.output_dir / args.csv_name
    png_path = args.output_dir / args.png_name
    write_delta_csv(csv_path, rows)
    plot_delta_auc(
        rows,
        output_path=png_path,
        title=f"{args.metric.upper()} change from previous top-k ({args.result_field})",
    )
    print(f"Wrote {csv_path}")
    print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
