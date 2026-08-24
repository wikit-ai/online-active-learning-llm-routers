import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from sklearn.metrics import auc as sklearn_auc

from plot_commands.plotting_function.utils import (  # type: ignore
    compute_mean_stderr,  # type: ignore
    get_colors_per_strategy,
)
from logging_config import logger

plt.style.use("ggplot")


def load_streaming_results(json_paths: str | list[str]) -> list[dict[str, int | float]]:
    """
    Load streaming results from one or multiple JSON files.

    Args:
        json_paths: Path to a single JSON file or list of paths containing streaming_results

    Returns:
        List of streaming result dictionaries from all files combined
    """
    if isinstance(json_paths, str):
        json_paths = [json_paths]

    all_results: list[dict[str, int | float]] = []
    for json_path in json_paths:
        with open(json_path, "r") as f:
            data = json.load(f)
        all_results.extend(data["streaming_results"])

    return all_results


def organize_data_by_strategy_and_k(
    streaming_results: list[dict[str, int | float]],
) -> dict[str, dict[int, dict[str, list[float]]]]:
    """
    Organize streaming results by strategy and k_value.
    Groups multiple runs and calculates mean and standard error per step.

    Args:
        streaming_results: List of streaming result dictionaries

    Returns:
        Dictionary: {strategy: {k_value: {'steps': [...], 'mean_performance': [...], 'std_error': [...]}}}
    """
    raw_data: dict[str, dict[int, dict[int, list[float]]]] = {}

    for result in streaming_results:
        strategy = result["strategy"]
        k_value = result["k_value"]
        step = result["step"]
        performance = result["test_performance"]

        if strategy not in raw_data:  # type: ignore
            raw_data[strategy] = {}  # type: ignore

        if k_value not in raw_data[strategy]:  # type: ignore
            raw_data[strategy][k_value] = {}  # type: ignore

        if step not in raw_data[strategy][k_value]:  # type: ignore
            raw_data[strategy][k_value][step] = []  # type: ignore

        raw_data[strategy][k_value][step].append(performance)  # type: ignore

    data: dict[str, dict[int, dict[str, list[float]]]] = {}

    for strategy, strategy_data in raw_data.items():
        data[strategy] = {}

        for k_value, k_data in strategy_data.items():
            steps = sorted(k_data.keys())
            mean_performance: list[float] = []
            std_error: list[float] = []

            for step in steps:
                mean_perf, std_err = compute_mean_stderr(k_data[step])
                mean_performance.append(mean_perf)
                std_error.append(std_err)

            data[strategy][k_value] = {  # type: ignore
                "steps": steps,
                "mean_performance": mean_performance,
                "std_error": std_error,
            }

    return data


def _compute_auc(steps: np.ndarray, values: np.ndarray) -> float:
    """Return AUC with steps normalised to [0, 1]."""
    if len(steps) < 2:
        return 0.0
    x_norm = (steps - steps[0]) / (steps[-1] - steps[0])
    return float(sklearn_auc(x_norm, values))


def _compute_auc_per_seed(
    json_paths: str | list[str],
) -> dict[str, dict[int, dict[str, Any]]]:
    """
    Compute AUC per seed (per json file) for each (strategy, k_value),
    then return mean and stderr across seeds.

    Returns:
        {strategy: {k_value: {'mean': float, 'stderr': float}}}
    """
    if isinstance(json_paths, str):
        json_paths = [json_paths]

    aucs: dict = defaultdict(lambda: defaultdict(list))

    for json_path in json_paths:
        with open(json_path, "r") as f:
            data = json.load(f)

        grouped: dict = defaultdict(list)
        for result in data["streaming_results"]:
            key = (result["strategy"], result["k_value"])
            grouped[key].append((result["step"], result["test_performance"]))

        for (strategy, k_value), records in grouped.items():
            records.sort(key=lambda x: x[0])
            steps = np.array([r[0] for r in records])
            performances = np.array([r[1] for r in records])
            aucs[strategy][k_value].append(_compute_auc(steps, performances))

    result: dict[str, dict[int, dict[str, float | list[float]]]] = {}
    for strategy, k_data in aucs.items():
        result[strategy] = {}
        for k_value, auc_list in k_data.items():
            mean, stderr = compute_mean_stderr(auc_list)
            result[strategy][k_value] = {
                "mean": mean,
                "stderr": stderr,
                "values": auc_list,
            }

    return result


def _plot_best_envelope(
    ax: Axes,
    series: dict[Any, tuple[np.ndarray, np.ndarray]],
    seed_aucs: dict[Any, list[float]],
    label: str,
) -> None:
    """Draw the pointwise max over `series` (mean curves) on their common steps.

    The legend AUC is the mean ± stderr of the per-seed best AUCs when available,
    otherwise the AUC of the envelope itself.

    Args:
        ax: Axes to draw on
        series: {key: (steps, mean_values)} for each curve to envelope
        seed_aucs: {key: per-seed AUC list} used for the legend statistics
        label: Legend prefix, e.g. "best across k"
    """
    if len(series) < 2:
        return

    common_steps: list[float] = sorted(
        set.intersection(*[set(s.tolist()) for s, _ in series.values()])
    )
    if not common_steps:
        return

    best_steps = np.array(common_steps, dtype=float)
    best_means = np.array(
        [
            max(
                float(m[list(s).index(st)])
                for s, m in series.values()
                if st in s.tolist()
            )
            for st in common_steps
        ]
    )

    n_seeds = min((len(v) for v in seed_aucs.values() if v), default=0)
    if n_seeds > 0:
        best_per_seed = [
            max(v[i] for v in seed_aucs.values() if i < len(v)) for i in range(n_seeds)
        ]
        auc_mean, auc_stderr = compute_mean_stderr(best_per_seed)
        legend = f"{label} (AUC={auc_mean:.3f}±{auc_stderr:.3f})"
    else:
        legend = f"{label} (AUC={_compute_auc(best_steps, best_means):.3f})"

    ax.plot(  # type: ignore
        best_steps,
        best_means,
        color="crimson",
        linewidth=2.5,
        linestyle="--",
        label=legend,
        alpha=0.4,
    )


def plot_performance_vs_step_per_strategy(
    json_paths: str | list[str],
    output_dir: str = "plots_benchmarks/performance_vs_step",
    figsize: tuple[int, int] = (12, 8),
):
    """
    Create line plots showing performance vs step for each strategy.
    Each plot shows multiple lines (one per k_value) plus a dashed "best across k" line.
    Each line is annotated at its right end with its AUC (steps normalised to [0, 1]).
    Handles multiple runs by computing mean and standard error.

    Args:
        json_paths: Path to a single JSON file or list of paths containing streaming_results
        output_dir: Directory to save plots
        figsize: Figure size (width, height)
    """
    streaming_results = load_streaming_results(json_paths)
    data = organize_data_by_strategy_and_k(streaming_results)
    auc_per_seed = _compute_auc_per_seed(json_paths)

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    strategies = sorted(data.keys())
    all_k_values: set[int] = set()
    for strategy_data in data.values():
        all_k_values.update(strategy_data.keys())
    k_values = sorted(all_k_values)

    colors_k: list[str] = plt.cm.viridis(np.linspace(0, 1, len(k_values)))  # type: ignore

    for strategy in strategies:
        fig, ax = plt.subplots(figsize=figsize)  # type: ignore

        k_arrays: dict[int, tuple[np.ndarray, np.ndarray]] = {}

        for i, k_value in enumerate(k_values):
            if k_value not in data[strategy]:
                continue

            steps = np.array(data[strategy][k_value]["steps"])
            mean_performance = np.array(data[strategy][k_value]["mean_performance"])
            std_error = np.array(data[strategy][k_value]["std_error"])

            k_arrays[k_value] = (steps, mean_performance)
            auc_mean = auc_per_seed.get(strategy, {}).get(k_value, {}).get("mean", 0.0)
            auc_stderr = (
                auc_per_seed.get(strategy, {}).get(k_value, {}).get("stderr", 0.0)
            )
            auc_label = f"k={k_value} (AUC={auc_mean:.3f}±{auc_stderr:.3f})"

            ax.plot(  # type: ignore
                steps,
                mean_performance,
                color=colors_k[i],
                marker="o",
                markersize=4,
                linewidth=2,
                label=auc_label,
                alpha=0.8,
            )

            ax.fill_between(  # type: ignore
                steps,
                mean_performance - std_error,
                mean_performance + std_error,
                color=colors_k[i],
                alpha=0.2,
            )

        _plot_best_envelope(
            ax,
            k_arrays,
            {
                k: auc_per_seed.get(strategy, {}).get(k, {}).get("values", [])
                for k in k_arrays
            },
            "best across k",
        )

        ax.set_xlabel("Step", fontsize=14, fontweight="bold")  # type: ignore
        ax.set_ylabel("Test Performance", fontsize=14, fontweight="bold")  # type: ignore
        ax.set_title(  # type: ignore
            f"Test Performance vs Step - {strategy.capitalize()}",
            fontsize=16,
            fontweight="bold",
            pad=20,
        )
        ax.grid(True, alpha=0.3, linestyle="--")  # type: ignore
        ax.legend(  # type: ignore
            bbox_to_anchor=(1.05, 1),
            loc="upper left",
            fontsize=10,
            framealpha=0.9,
        )

        plt.tight_layout()

        output_file = output_path / f"performance_vs_step_{strategy}.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")  # type: ignore
        plt.close(fig)

    logger.info(f"Generated {len(strategies)} plots in {output_path}")


def plot_performance_vs_step_all_strategies(
    json_paths: str | list[str],
    k_value: int,
    output_dir: str = "plots_benchmarks/performance_vs_step",
    figsize: tuple[int, int] = (14, 8),
):
    """
    Create a single plot comparing all strategies for a specific k_value.
    Handles multiple runs by computing mean and standard error.

    Args:
        json_paths: Path to a single JSON file or list of paths containing streaming_results
        k_value: The k_value to plot
        output_dir: Directory to save plots
        figsize: Figure size (width, height)
    """
    streaming_results = load_streaming_results(json_paths)
    data = organize_data_by_strategy_and_k(streaming_results)
    auc_per_seed = _compute_auc_per_seed(json_paths)

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    strategies = sorted(data.keys())
    strategy_colors = get_colors_per_strategy(strategies)
    strategy_color_map = dict(zip(strategies, strategy_colors))

    fig, ax = plt.subplots(figsize=figsize)  # type: ignore

    strategy_arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for strategy in strategies:
        if k_value not in data[strategy]:
            continue

        steps = np.array(data[strategy][k_value]["steps"])
        mean_performance = np.array(data[strategy][k_value]["mean_performance"])
        std_error = np.array(data[strategy][k_value]["std_error"])

        strategy_arrays[strategy] = (steps, mean_performance)
        auc_mean = auc_per_seed.get(strategy, {}).get(k_value, {}).get("mean", 0.0)
        auc_stderr = auc_per_seed.get(strategy, {}).get(k_value, {}).get("stderr", 0.0)

        ax.plot(  # type: ignore
            steps,
            mean_performance,
            color=strategy_color_map[strategy],
            marker="o",
            markersize=4,
            linewidth=2,
            label=f"{strategy} (AUC={auc_mean:.3f}±{auc_stderr:.3f})",
            alpha=0.8,
        )

        ax.fill_between(  # type: ignore
            steps,
            mean_performance - std_error,
            mean_performance + std_error,
            color=strategy_color_map[strategy],
            alpha=0.2,
        )

    _plot_best_envelope(
        ax,
        strategy_arrays,
        {
            s: auc_per_seed.get(s, {}).get(k_value, {}).get("values", [])
            for s in strategy_arrays
        },
        "best across strategies",
    )

    ax.set_xlabel("Step", fontsize=14, fontweight="bold")  # type: ignore
    ax.set_ylabel("Test Performance", fontsize=14, fontweight="bold")  # type: ignore
    ax.set_title(  # type: ignore
        f"Test Performance vs Step - All Strategies (k={k_value})",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )
    ax.grid(True, alpha=0.3, linestyle="--")  # type: ignore
    ax.legend(  # type: ignore
        bbox_to_anchor=(1.05, 1),
        loc="upper left",
        fontsize=9,
        framealpha=0.9,
    )

    plt.tight_layout()

    output_file = output_path / f"performance_vs_step_all_strategies_k{k_value}.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")  # type: ignore
    plt.close(fig)

    logger.info(f"Generated plot: {output_file}")
