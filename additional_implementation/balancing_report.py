"""Aggregate balancing-sweep runs into tables and compact figures."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, Iterable, List

import plotly.graph_objects as go
from plotly.subplots import make_subplots


METRICS = (
    "test_auc",
    "expert_load_entropy",
    "expert_load_normalized_entropy",
    "expert_load_cv",
    "expert_max_load_ratio",
)

PAIRED_METRICS = (
    "test_auc",
    "expert_load_normalized_entropy",
    "expert_load_cv",
    "expert_max_load_ratio",
)


def _condition_sort_key(row: Dict[str, Any]) -> tuple[int, float]:
    if row["balancing_method"] == "loss_free":
        return (1, math.inf)
    return (0, float(row["alpha"]))


def _condition_label(row: Dict[str, Any]) -> str:
    if row["balancing_method"] == "loss_free":
        return f"LF-sign control\n(u={row['loss_free_update_rate']:g})"
    alpha = float(row["alpha"])
    return "α=0" if alpha == 0 else f"α={alpha:g}"


def _compact_condition_label(row: Dict[str, Any]) -> str:
    if row["balancing_method"] == "loss_free":
        return "LF-sign"
    labels = {
        0.0: "0",
        1e-3: "10⁻³",
        1e-2: "10⁻²",
        1e-1: "10⁻¹",
        1.0: "1",
        10.0: "10",
    }
    alpha = float(row["alpha"])
    return labels.get(alpha, f"{alpha:g}")


def load_results(output_dir: Path) -> List[Dict[str, Any]]:
    paths = sorted(output_dir.glob("seed_*/**/result.json"))
    if not paths:
        raise FileNotFoundError(f"No per-condition results found under {output_dir}")
    rows: List[Dict[str, Any]] = []
    for path in paths:
        with path.open("r", encoding="utf-8") as stream:
            row = json.load(stream)
        row["result_path"] = str(path.resolve())
        rows.append(row)
    return sorted(rows, key=lambda row: (_condition_sort_key(row), int(row["seed"])))


def validate_results(output_dir: Path, rows: List[Dict[str, Any]]) -> None:
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing sweep manifest: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)

    expected_conditions = {
        f"alpha_{float(alpha):g}" for alpha in manifest["alphas"]
    }
    if manifest["include_loss_free"]:
        expected_conditions.add(
            f"loss_free_u_{float(manifest['loss_free_update_rate']):g}"
        )
    expected = {
        (int(seed), condition)
        for seed in manifest["seeds"]
        for condition in expected_conditions
    }
    observed = [(int(row["seed"]), str(row["condition"])) for row in rows]
    if len(observed) != len(set(observed)):
        raise ValueError("Duplicate seed/condition records found in sweep results.")
    observed_set = set(observed)
    if observed_set != expected:
        missing = sorted(expected - observed_set)
        unexpected = sorted(observed_set - expected)
        raise ValueError(
            f"Incomplete sweep results; missing={missing}, unexpected={unexpected}."
        )


def summarize(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: dict[str, list[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["condition"]].append(row)

    summaries: List[Dict[str, Any]] = []
    for condition_rows in grouped.values():
        first = condition_rows[0]
        summary: Dict[str, Any] = {
            "condition": first["condition"],
            "label": _condition_label(first),
            "balancing_method": first["balancing_method"],
            "alpha": first["alpha"],
            "loss_free_update_rate": first["loss_free_update_rate"],
            "num_runs": len(condition_rows),
            "seeds": [int(row["seed"]) for row in condition_rows],
        }
        for metric in METRICS:
            values = [float(row[metric]) for row in condition_rows]
            summary[f"{metric}_mean"] = mean(values)
            summary[f"{metric}_std"] = stdev(values) if len(values) > 1 else 0.0
        summaries.append(summary)
    return sorted(summaries, key=_condition_sort_key)


def pair_against_alpha_zero(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    baselines = {
        int(row["seed"]): row
        for row in rows
        if row["balancing_method"] == "auxiliary" and float(row["alpha"]) == 0.0
    }
    seeds = {int(row["seed"]) for row in rows}
    if set(baselines) != seeds:
        raise ValueError("Each seed must contain exactly one alpha=0 baseline.")

    paired: List[Dict[str, Any]] = []
    for row in rows:
        if row["balancing_method"] == "auxiliary" and float(row["alpha"]) == 0.0:
            continue
        baseline = baselines[int(row["seed"])]
        paired_row: Dict[str, Any] = {
            "condition": row["condition"],
            "label": _condition_label(row),
            "balancing_method": row["balancing_method"],
            "alpha": row["alpha"],
            "loss_free_update_rate": row["loss_free_update_rate"],
            "seed": int(row["seed"]),
        }
        for metric in PAIRED_METRICS:
            paired_row[f"delta_{metric}"] = float(row[metric]) - float(
                baseline[metric]
            )
        paired.append(paired_row)
    return sorted(paired, key=lambda row: (_condition_sort_key(row), row["seed"]))


def summarize_paired(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: dict[str, list[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["condition"]].append(row)

    summaries: List[Dict[str, Any]] = []
    for condition_rows in grouped.values():
        first = condition_rows[0]
        summary: Dict[str, Any] = {
            "condition": first["condition"],
            "label": first["label"],
            "balancing_method": first["balancing_method"],
            "alpha": first["alpha"],
            "loss_free_update_rate": first["loss_free_update_rate"],
            "num_pairs": len(condition_rows),
        }
        for metric in PAIRED_METRICS:
            key = f"delta_{metric}"
            values = [float(row[key]) for row in condition_rows]
            summary[f"{key}_mean"] = mean(values)
            summary[f"{key}_std"] = stdev(values) if len(values) > 1 else 0.0
        summaries.append(summary)
    return sorted(summaries, key=_condition_sort_key)


def write_run_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    fields = [
        "condition",
        "balancing_method",
        "alpha",
        "loss_free_update_rate",
        "dataset",
        "seed",
        "num_experts",
        "top_k",
        "best_epoch",
        "test_auc",
        "test_logloss",
        "expert_load_entropy",
        "expert_load_normalized_entropy",
        "expert_load_cv",
        "expert_load_cv_squared",
        "expert_max_load_ratio",
        "expert_max_violation",
        "expert_load_counts",
        "expert_load_probabilities",
        "loss_free_bias",
        "loss_free_update_count",
        "elapsed_seconds",
        "checkpoint",
        "result_path",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(row[key]) if isinstance(row.get(key), list) else row.get(key)
                    for key in fields
                }
            )


def write_summary_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: json.dumps(value) if isinstance(value, list) else value
                    for key, value in row.items()
                }
            )


def write_rows_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_figure(summaries: List[Dict[str, Any]], path: Path) -> None:
    auxiliary = [
        row for row in summaries if row["balancing_method"] == "auxiliary"
    ]
    loss_free = [
        row for row in summaries if row["balancing_method"] == "loss_free"
    ]
    labels = [_compact_condition_label(row) for row in summaries]
    auxiliary_labels = [_compact_condition_label(row) for row in auxiliary]

    auc_color = "#4C78A8"
    entropy_color = "#E45756"
    figure = make_subplots(specs=[[{"secondary_y": True}]])
    figure.add_trace(
        go.Scatter(
            x=auxiliary_labels,
            y=[row["test_auc_mean"] for row in auxiliary],
            error_y={
                "type": "data",
                "array": [row["test_auc_std"] for row in auxiliary],
                "visible": True,
                "thickness": 1,
                "width": 3,
            },
            mode="lines+markers",
            line={"width": 1.8, "color": auc_color},
            marker={"size": 7, "symbol": "circle", "color": auc_color},
            name="Test AUC",
            hovertemplate="α=%{x}<br>Test AUC: %{y:.6f}<extra></extra>",
        ),
        secondary_y=False,
    )
    figure.add_trace(
        go.Scatter(
            x=auxiliary_labels,
            y=[row["expert_load_normalized_entropy_mean"] for row in auxiliary],
            error_y={
                "type": "data",
                "array": [
                    row["expert_load_normalized_entropy_std"] for row in auxiliary
                ],
                "visible": True,
                "thickness": 1,
                "width": 3,
            },
            mode="lines+markers",
            line={"width": 1.8, "color": entropy_color},
            marker={"size": 7, "symbol": "square", "color": entropy_color},
            name="Expert Load Entropy",
            hovertemplate="α=%{x}<br>H/log N: %{y:.6f}<extra></extra>",
        ),
        secondary_y=True,
    )
    for row in loss_free:
        label = _compact_condition_label(row)
        figure.add_trace(
            go.Scatter(
                x=[label],
                y=[row["test_auc_mean"]],
                error_y={
                    "type": "data",
                    "array": [row["test_auc_std"]],
                    "visible": True,
                    "thickness": 1,
                    "width": 3,
                },
                mode="markers",
                marker={"size": 9, "symbol": "diamond", "color": auc_color},
                name="LF-sign Test AUC",
                showlegend=False,
                hovertemplate="LF-sign<br>Test AUC: %{y:.6f}<extra></extra>",
            ),
            secondary_y=False,
        )
        figure.add_trace(
            go.Scatter(
                x=[label],
                y=[row["expert_load_normalized_entropy_mean"]],
                error_y={
                    "type": "data",
                    "array": [row["expert_load_normalized_entropy_std"]],
                    "visible": True,
                    "thickness": 1,
                    "width": 3,
                },
                mode="markers",
                marker={"size": 9, "symbol": "diamond", "color": entropy_color},
                name="LF-sign entropy",
                showlegend=False,
                hovertemplate="LF-sign<br>H/log N: %{y:.6f}<extra></extra>",
            ),
            secondary_y=True,
        )
    if loss_free:
        figure.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker={"size": 8, "symbol": "diamond", "color": "#666666"},
                name="LF-sign control",
                hoverinfo="skip",
            ),
            secondary_y=False,
        )

    auc_lower = min(row["test_auc_mean"] - row["test_auc_std"] for row in summaries)
    auc_upper = max(row["test_auc_mean"] + row["test_auc_std"] for row in summaries)
    auc_padding = (auc_upper - auc_lower) * 0.12
    entropy_lower = min(
        row["expert_load_normalized_entropy_mean"]
        - row["expert_load_normalized_entropy_std"]
        for row in summaries
    )
    entropy_upper = max(
        row["expert_load_normalized_entropy_mean"]
        + row["expert_load_normalized_entropy_std"]
        for row in summaries
    )
    entropy_padding = (entropy_upper - entropy_lower) * 0.08

    figure.update_xaxes(
        title_text="α / control",
        categoryorder="array",
        categoryarray=labels,
        showline=True,
        linecolor="#555555",
        gridcolor="white",
        tickfont={"size": 11},
        title_font={"size": 11},
    )
    figure.update_yaxes(
        title_text="Test AUC",
        range=[auc_lower - auc_padding, auc_upper + auc_padding],
        tickformat=".3f",
        color=auc_color,
        showline=True,
        linecolor=auc_color,
        gridcolor="white",
        tickfont={"size": 10},
        title_font={"size": 11},
        secondary_y=False,
    )
    figure.update_yaxes(
        title_text="Expert Load Entropy (H/log N)",
        range=[entropy_lower - entropy_padding, entropy_upper + entropy_padding],
        tickformat=".2f",
        color=entropy_color,
        showline=True,
        linecolor=entropy_color,
        showgrid=False,
        tickfont={"size": 10},
        title_font={"size": 11},
        secondary_y=True,
    )
    figure.update_layout(
        template="plotly_white",
        width=690,
        height=226,
        margin={"l": 58, "r": 66, "t": 36, "b": 42},
        paper_bgcolor="white",
        plot_bgcolor="#EBEBEB",
        font={"family": "Arial, sans-serif", "size": 11, "color": "#222222"},
        legend={
            "orientation": "h",
            "x": 0.5,
            "xanchor": "center",
            "y": 1.01,
            "yanchor": "bottom",
            "font": {"size": 9},
        },
        hovermode="x unified",
    )
    figure.write_html(
        path,
        include_plotlyjs=True,
        config={
            "displaylogo": False,
            "toImageButtonOptions": {
                "format": "svg",
                "filename": "rqmoe_alpha_auc_entropy",
                "width": 690,
                "height": 226,
                "scale": 1,
            },
        },
    )


def _mean_std(row: Dict[str, Any], metric: str, digits: int = 6) -> str:
    return (
        f"{row[f'{metric}_mean']:.{digits}f} ± "
        f"{row[f'{metric}_std']:.{digits}f}"
    )


def write_markdown(
    runs: List[Dict[str, Any]],
    summaries: List[Dict[str, Any]],
    paired_summaries: List[Dict[str, Any]],
    path: Path,
) -> None:
    first = runs[0]
    lines = [
        "# Auxiliary Load-Balancing Sweep",
        "",
        (
            f"Dataset: `{first['dataset']}`; experts: {first['num_experts']}; "
            f"Top-k: {first['top_k']}; seeds: "
            f"{', '.join(str(seed) for seed in sorted({row['seed'] for row in runs}))}."
        ),
        "",
        "The auxiliary conditions use the dense pre-Top-k softmax probability in "
        "the Switch loss. The adapted Loss-Free sign-bias control uses `α=0`, "
        "adds the learned bias only for expert selection, and applies the "
        "Algorithm-1 sign load-error update with `u=1e-3`. All conditions "
        "retain the legacy Top-1 mixture-weight renormalization.",
        "",
        "Load statistics aggregate actual expert assignments over the complete test "
        "split. Entropy is normalized by `log(N)` below; CV uses population standard "
        "deviation divided by mean load; max-load ratio is maximum load divided by "
        "mean load.",
        "",
        "## Mean ± sample standard deviation",
        "",
        "| Condition | Runs | AUC | Load entropy / log N | Load CV | Max-load ratio |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            "| {label} | {runs} | {auc} | {entropy} | {cv} | {maximum} |".format(
                label=row["label"].replace("\n", " "),
                runs=row["num_runs"],
                auc=_mean_std(row, "test_auc"),
                entropy=_mean_std(row, "expert_load_normalized_entropy"),
                cv=_mean_std(row, "expert_load_cv"),
                maximum=_mean_std(row, "expert_max_load_ratio"),
            )
        )
    lines.extend(
        [
            "",
            "## Paired changes relative to alpha=0",
            "",
            "| Condition | ΔAUC | ΔH/log N | ΔCV | Δmax/mean |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in paired_summaries:
        lines.append(
            "| {label} | {auc:+.6f} ± {auc_std:.6f} | "
            "{entropy:+.6f} ± {entropy_std:.6f} | "
            "{cv:+.6f} ± {cv_std:.6f} | "
            "{maximum:+.6f} ± {maximum_std:.6f} |".format(
                label=row["label"].replace("\n", " "),
                auc=row["delta_test_auc_mean"],
                auc_std=row["delta_test_auc_std"],
                entropy=row["delta_expert_load_normalized_entropy_mean"],
                entropy_std=row["delta_expert_load_normalized_entropy_std"],
                cv=row["delta_expert_load_cv_mean"],
                cv_std=row["delta_expert_load_cv_std"],
                maximum=row["delta_expert_max_load_ratio_mean"],
                maximum_std=row["delta_expert_max_load_ratio_std"],
            )
        )
    lines.extend(
        [
            "",
            "## Per-run audit table",
            "",
            "| Condition | Seed | Epoch | AUC | LogLoss | Raw H | H/log N | CV | Max/mean | Counts |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in runs:
        lines.append(
            "| {label} | {seed} | {epoch} | {auc:.6f} | {logloss:.6f} | "
            "{entropy:.6f} | {normalized:.6f} | {cv:.6f} | {maximum:.6f} | `{counts}` |".format(
                label=_condition_label(row).replace("\n", " "),
                seed=row["seed"],
                epoch=row["best_epoch"],
                auc=row["test_auc"],
                logloss=row["test_logloss"],
                entropy=row["expert_load_entropy"],
                normalized=row["expert_load_normalized_entropy"],
                cv=row["expert_load_cv"],
                maximum=row["expert_max_load_ratio"],
                counts=", ".join(str(value) for value in row["expert_load_counts"]),
            )
        )
    lines.extend(
        [
            "",
            "Artifacts: `runs.csv`, `summary.csv`, `paired_deltas.csv`, "
            "`paired_summary.csv`, and the self-contained "
            "`balancing_sweep.html` compact dual-axis AUC/entropy figure. CV and "
            "max-load ratio remain reported in the tables above and in the CSV files.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def generate_report(output_dir: Path) -> None:
    output_dir = output_dir.expanduser().resolve()
    runs = load_results(output_dir)
    validate_results(output_dir, runs)
    summaries = summarize(runs)
    paired = pair_against_alpha_zero(runs)
    paired_summaries = summarize_paired(paired)
    write_run_csv(runs, output_dir / "runs.csv")
    write_summary_csv(summaries, output_dir / "summary.csv")
    write_rows_csv(paired, output_dir / "paired_deltas.csv")
    write_rows_csv(paired_summaries, output_dir / "paired_summary.csv")
    write_figure(summaries, output_dir / "balancing_sweep.html")
    write_markdown(runs, summaries, paired_summaries, output_dir / "RESULTS.md")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    generate_report(args.output_dir)


if __name__ == "__main__":
    main()
