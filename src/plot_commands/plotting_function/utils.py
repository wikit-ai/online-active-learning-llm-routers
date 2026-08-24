import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from plot_commands.constants import (
    STRATEGY_EXCLUDE,
    STRATEGY_ORDER,
)

from logging_config import logger


def get_colors_per_strategy(strategies: list[str]) -> list[str]:
    colors: list[str] = []
    for strategy in strategies:
        if strategy.startswith("rand_"):
            colors.append("#0072B2")
        elif strategy.startswith("inferred_sparse"):
            colors.append("#9C27B0")
        elif strategy.startswith("unc_t3"):
            colors.append("#E69F00")
        elif strategy.startswith("act_repr_min"):
            colors.append("#009E73")
        elif strategy.startswith("repr_vmf"):
            colors.append("#56B4E9")
        elif strategy.startswith("passive"):
            colors.append("#C2185B")
        elif strategy.startswith("oracle_sparse"):
            colors.append("#4A0072")
        elif strategy.startswith("oracle_ds"):
            colors.append("#000000")
        else:
            colors.append("gray")
    return colors


def get_markers_per_strategy(strategies: list[str]) -> list[str]:
    markers: list[str] = []
    for strategy in strategies:
        if strategy.startswith("rand_"):
            markers.append("o")
        elif strategy.startswith("act_unc"):
            markers.append("s")
        elif strategy.startswith("act_repr"):
            markers.append("^")
        elif strategy.startswith("act_learn"):
            markers.append("D")
        elif strategy == "static":
            markers.append("*")
        elif strategy.startswith("oracle"):
            markers.append("P")
        else:
            markers.append(".")
    return markers


def filter_strategies_by_prefix(
    data: dict[str, Any], prefixes: list[str] | None
) -> dict[str, Any]:
    """Keep only strategies whose name starts with one of the given prefixes."""
    if prefixes is None:
        return data
    return {
        strategy: metrics
        for strategy, metrics in data.items()
        if any(strategy.startswith(p) for p in prefixes)
    }


def print_statistics_table(
    stats: dict[str, dict[str, float | int]], k_value: int
) -> None:
    """Print a formatted table of statistics sorted by AUC performance."""
    has_dto = any("auc_dto_mean" in data for data in stats.values())

    logger.info(f"\n{'=' * 145}")
    logger.info(f"K_VALUE = {k_value}")
    logger.info("=" * 145)

    if has_dto:
        logger.info(
            f"{'Strategy':<30} {'AUC Perf (Mean±SE)':<25} {'AUC DTO (Mean±SE)':<25} "
            f"{'Train Cost Final (Mean±SE)':<30} {'N Samples':<10}"
        )
        logger.info("=" * 145)
    else:
        logger.info(
            f"{'Strategy':<30} {'AUC Perf (Mean±SE)':<25} "
            f"{'Train Cost Final (Mean±SE)':<30} {'N Samples':<10}"
        )
        logger.info("=" * 110)

    for strategy, data in sorted(
        stats.items(), key=lambda x: x[1]["auc_performance_mean"], reverse=True
    ):
        perf_str = (
            f"{data['auc_performance_mean']:.4f} ± {data['auc_performance_stderr']:.4f}"
        )
        cost_str = f"{data['train_cost_final_mean']:.2f} ± {data['train_cost_final_stderr']:.2f}"
        if has_dto and "auc_dto_mean" in data:
            dto_str = f"{data['auc_dto_mean']:.4f} ± {data['auc_dto_stderr']:.4f}"
            logger.info(
                f"{strategy:<30} {perf_str:<25} {dto_str:<25} {cost_str:<30} {data['n_samples']:<10}"
            )
        elif has_dto:
            logger.info(
                f"{strategy:<30} {perf_str:<25} {'N/A':<25} {cost_str:<30} {data['n_samples']:<10}"
            )
        else:
            logger.info(
                f"{strategy:<30} {perf_str:<25} {cost_str:<30} {data['n_samples']:<10}"
            )

    logger.info("=" * (145 if has_dto else 110))


def get_best_k_stats(results_dir: str) -> dict[str, Any]:
    """Load data and keep only the best k per strategy (maximises mean AUC)."""
    from plot_commands.plotting_function.plot_auc_vs_cost import (
        compute_statistics,
        load_auc_from_steps_per_k,
    )

    raw = load_auc_from_steps_per_k(results_dir)
    stats_per_k = {k: compute_statistics(strategies) for k, strategies in raw.items()}

    all_strategies: set[str] = set()
    for stats in stats_per_k.values():
        all_strategies.update(stats.keys())

    best_stats: dict[str, Any] = {}
    for strategy in sorted(all_strategies):
        best_k = max(
            (k for k, stats in stats_per_k.items() if strategy in stats),
            key=lambda k: stats_per_k[k][strategy]["auc_performance_mean"],
        )
        best_stats[strategy] = stats_per_k[best_k][strategy]

    return best_stats


def apply_standard_filters(
    best_stats: dict[str, Any],
    strategy_prefixes: list[str] | None = None,
    extra_exclude: set[str] | None = None,
) -> dict[str, Any]:
    """Apply standard exclusions and optional prefix filtering to best_stats."""
    exclude = STRATEGY_EXCLUDE | (extra_exclude or set())
    best_stats = {s: v for s, v in best_stats.items() if s not in exclude}
    if strategy_prefixes:
        best_stats = filter_strategies_by_prefix(best_stats, strategy_prefixes)
    return best_stats


def order_strategies(best_stats: dict) -> list[str]:
    """Return strategies in canonical order, with unknowns appended alphabetically."""
    ordered = [s for s in STRATEGY_ORDER if s in best_stats]
    remainder = sorted(s for s in best_stats if s not in STRATEGY_ORDER)
    return ordered + remainder


def setup_paper_rcparams() -> None:
    """Apply publication-quality matplotlib rcParams."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 9,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
        }
    )


def compute_mean_stderr(values: list | np.ndarray) -> tuple[float, float]:
    """Compute mean and standard error for a list of values."""
    arr = np.asarray(values, dtype=float)
    mean = float(np.mean(arr))
    stderr = float(np.std(arr, ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0
    return mean, stderr


def plot_bar_per_k(
    stats_per_k: dict[int, dict[str, dict[str, float | int]]],
    mean_key: str,
    stderr_key: str,
    benchmark: str,
    n_seeds: int,
    ylabel: str,
    title_prefix: str,
    filename_prefix: str,
    output_dir: str = "plots_benchmarks",
    figsize: tuple[int, int] = (12, 8),
) -> None:
    """Generic bar plot per k_value, sorted by mean_key descending."""
    output_path = Path(output_dir) / benchmark
    output_path.mkdir(exist_ok=True, parents=True)

    for k_value, stats in stats_per_k.items():
        sorted_strategies = sorted(
            stats.items(), key=lambda x: x[1][mean_key], reverse=True
        )

        strategies = [s[0] for s in sorted_strategies]
        means = [float(s[1][mean_key]) for s in sorted_strategies]
        stderrs = [float(s[1][stderr_key]) for s in sorted_strategies]

        colors = get_colors_per_strategy(strategies)

        fig, ax = plt.subplots(figsize=figsize)
        x_pos = np.arange(len(strategies))
        ax.bar(
            x_pos,
            means,
            yerr=stderrs,
            capsize=5,
            color=colors,
            alpha=0.7,
            edgecolor="black",
            linewidth=1.5,
        )

        ax.set_xlabel("Strategy", fontsize=14, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=14, fontweight="bold")
        ax.set_title(
            f"{title_prefix} (k={k_value}, {benchmark.capitalize()})\n"
            f"(Error bars show standard error across {n_seeds} seeds)",
            fontsize=16,
            fontweight="bold",
            pad=20,
        )
        ax.set_xticks(x_pos)
        ax.set_xticklabels(strategies, rotation=45, ha="right", fontsize=8)
        ax.grid(True, alpha=0.3, linestyle="--", axis="y")

        for i, (mean, stderr) in enumerate(zip(means, stderrs)):
            ax.text(
                i,
                mean + stderr + 0.01,
                f"{mean:.3f}",
                ha="center",
                va="bottom",
                fontsize=7,
                fontweight="bold",
            )

        plt.tight_layout()
        output_file = output_path / f"{filename_prefix}_k{k_value}_{benchmark}.png"
        plt.savefig(output_file, dpi=300, bbox_inches="tight")
        plt.close(fig)


def load_json_by_benchmark(
    results_dir: str,
    embedding_model: str,
    experiment_subdir: str,
    file_pattern: str,
) -> dict[str, list[list[dict[str, Any]]]]:
    """Load JSON files organized by benchmark from the standard directory layout.

    Expected path:
        <results_dir>/<experiment_subdir>/<embedding_model>/<benchmark>/<seed>/run_*/<file_pattern>

    Returns:
        {benchmark: [[entry, ...], ...]}  one inner list per file found
    """
    root = Path(results_dir) / experiment_subdir / embedding_model
    if not root.exists():
        root = Path(results_dir)

    data: dict[str, list[list[dict[str, Any]]]] = {}
    for json_file in root.rglob(file_pattern):
        parts = json_file.relative_to(root).parts
        benchmark = parts[0] if len(parts) >= 2 else "unknown"
        with open(json_file, "r", encoding="utf-8") as f:
            run_log: list[dict[str, Any]] = json.load(f)
        data.setdefault(benchmark, []).append(run_log)
    return data


def get_k_from_run(run_log: list[dict[str, Any]]) -> int:
    """Infer the first k value from a run's metadata entry."""
    for entry in run_log:
        if entry.get("type") == "metadata" and "k" in entry:
            return int(entry["k"][0])
    return 1
