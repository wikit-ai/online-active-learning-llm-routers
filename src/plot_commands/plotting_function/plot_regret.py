from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from plot_commands.plotting_function.utils import (
    compute_mean_stderr,  # type: ignore
    get_k_from_run,
    load_json_by_benchmark,
)

from logging_config import logger

plt.style.use("ggplot")

_CHECKPOINT_PCTS = [0.0] + [round(p * 0.1, 1) for p in range(1, 11)]


def load_regret_data(
    results_dir: str, embedding_model: str
) -> dict[str, list[list[dict[str, Any]]]]:
    """Load all regret_results json files, organized by benchmark."""
    return load_json_by_benchmark(
        results_dir, embedding_model, "model_switch_regret", "regret_results_*.json"
    )


def compute_regret_stats(
    runs: list[list[dict[str, Any]]],
) -> dict[float, dict[str, dict[str, float]]]:
    """Aggregate delta metrics across seeds per checkpoint."""
    per_pct: dict[float, dict[str, list[float]]] = {
        pct: {"delta_perf": [], "delta_dto": []} for pct in _CHECKPOINT_PCTS
    }

    for run_log in runs:
        k = get_k_from_run(run_log)
        for entry in run_log:
            if entry.get("type") != "checkpoint":
                continue
            pct = entry["checkpoint_pct"]
            if pct not in per_pct:
                continue
            dp = entry.get(f"delta_perf_k{k}")
            dd = entry.get(f"delta_dto_k{k}")
            if dp is not None:
                per_pct[pct]["delta_perf"].append(dp)
            if dd is not None:
                per_pct[pct]["delta_dto"].append(dd)

    stats: dict[float, dict[str, dict[str, float]]] = {}
    for pct in _CHECKPOINT_PCTS:
        stats[pct] = {}
        for metric in ["delta_perf", "delta_dto"]:
            vals = per_pct[pct][metric]
            if not vals:
                stats[pct][metric] = {"mean": 0.0, "stderr": 0.0}
            else:
                mean, stderr = compute_mean_stderr(vals)
                stats[pct][metric] = {"mean": mean, "stderr": stderr}
        stats[pct]["n"] = {
            "mean": float(len(per_pct[pct]["delta_perf"])),
            "stderr": 0.0,
        }
    return stats


_METRIC_CFG: dict[str, dict[str, str]] = {
    "delta_perf": {
        "label": r"$\Delta$ Performance",
        "color": "#009E73",
        "marker": "o",
    },
    "delta_dto": {
        "label": r"$\Delta$ DTO",
        "color": "#E69F00",
        "marker": "s",
    },
}


def plot_regret_over_time(
    stats_per_benchmark: dict[str, dict[float, dict[str, dict[str, float]]]],
    n_runs_per_benchmark: dict[str, int],
    output_dir: str = "plots_benchmarks/model_switch_regret",
) -> None:
    """2×2 grid: one cell per benchmark, two lines (delta AUC_P, delta AUC_DTO)."""
    benchmarks = sorted(stats_per_benchmark.keys())
    n = len(benchmarks)
    if n == 0:
        logger.info("No data to plot.")
        return

    ncols = min(n, 2)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(  # type: ignore
        nrows, ncols, figsize=(6 * ncols, 3 * nrows), squeeze=False
    )

    x_vals = [pct * 100 for pct in _CHECKPOINT_PCTS]
    x_labels = [f"{int(p)}%" for p in x_vals]

    for idx, benchmark in enumerate(benchmarks):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]
        stats = stats_per_benchmark[benchmark]
        n_runs = n_runs_per_benchmark[benchmark]

        for metric, cfg in _METRIC_CFG.items():
            means = [
                stats.get(pct, {}).get(metric, {}).get("mean", 0.0)
                for pct in _CHECKPOINT_PCTS
            ]
            stderrs = [
                stats.get(pct, {}).get(metric, {}).get("stderr", 0.0)
                for pct in _CHECKPOINT_PCTS
            ]
            ax.plot(
                x_vals,
                means,
                label=cfg["label"],
                color=cfg["color"],
                marker=cfg["marker"],
                markersize=5,
                linewidth=2,
            )
            lower = [m - s for m, s in zip(means, stderrs)]
            upper = [m + s for m, s in zip(means, stderrs)]
            ax.fill_between(x_vals, lower, upper, color=cfg["color"], alpha=0.15)

        ax.axhline(0, color="black", linestyle="--", alpha=0.5, linewidth=1)
        ax.axvline(0, color="red", linestyle=":", alpha=0.7, linewidth=1)
        ax.set_ylim(-0.02, 0.02)
        ax.set_xticks(x_vals)
        ax.set_xticklabels(x_labels, fontsize=8)
        ax.set_xlabel("Post-switch stream", fontsize=9)
        ax.set_ylabel("Regret (Continual - Optimal)", fontsize=9)
        ax.set_title(f"{benchmark}", fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3, linestyle="--")

    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    handles = [
        plt.Line2D(  # type: ignore
            [0],
            [0],
            color=cfg["color"],
            marker=cfg["marker"],
            markersize=6,
            linewidth=2,
            label=cfg["label"],
        )
        for cfg in _METRIC_CFG.values()
    ]
    handles.append(
        plt.Line2D(  # type: ignore
            [0],
            [0],
            color="black",
            linestyle="--",
            alpha=0.5,
            linewidth=1,
            label="No regret",
        )
    )
    handles.append(
        plt.Line2D(  # type: ignore
            [0],
            [0],
            color="red",
            linestyle=":",
            alpha=0.7,
            linewidth=1,
            label="Switch",
        )
    )
    fig.legend(  # type: ignore
        handles=handles,
        loc="lower center",
        ncol=len(handles),
        fontsize=9,
        bbox_to_anchor=(0.5, -0.03),
        frameon=True,
    )

    plt.tight_layout(rect=(0, 0.04, 1, 0.96))

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    out_file = out / "regret_over_time.png"
    plt.savefig(out_file, dpi=300, bbox_inches="tight")  # type: ignore
    logger.info(f"Saved -> {out_file}")
    plt.close()
