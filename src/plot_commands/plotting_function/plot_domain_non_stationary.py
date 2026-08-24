from collections import defaultdict
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from plot_commands.plotting_function.utils import (
    compute_mean_stderr,  # type: ignore
    get_colors_per_strategy,
)

from logging_config import logger

plt.style.use("ggplot")


def compute_period_statistics(
    period_data: dict[str, dict[str, dict[str, list[float]]]],
) -> dict[str, dict[str, dict[str, float | int]]]:
    """
    Compute mean and standard error for period-specific metrics for a single k_value slice.

    Args:
        period_data: {strategy: {period: {metric: [values across seeds]}}}

    Returns:
        dict: {strategy: {period: {metric_mean: value, metric_stderr: value, n_samples: int}}}
    """
    stats: dict[str, dict[str, dict[str, float | int]]] = {}

    for strategy, periods in period_data.items():
        stats[strategy] = {}
        for period_name, metrics in periods.items():
            stats[strategy][period_name] = {}
            for metric_name, values in metrics.items():
                mean, stderr = compute_mean_stderr(values)
                stats[strategy][period_name][f"{metric_name}_mean"] = mean
                stats[strategy][period_name][f"{metric_name}_stderr"] = stderr
            stats[strategy][period_name]["n_samples"] = len(metrics["auc_performance"])

    return stats


def load_domain_period_metrics_with_k(
    results_dir: str,
) -> dict[int, dict[str, dict[str, dict[str, list[float]]]]]:
    """
    Load domain-period metrics from domain_period_metrics_*.json files.

    The saved structure from DomainShiftExperiment is:
        {strategy: {k_value: {domain_name: {metric: value}}}}

    Returns:
        dict: {k_value: {strategy: {domain_name: {metric: [values across seeds]}}}}
    """
    results_path = Path(results_dir)
    period_data: dict[int, dict[str, dict[str, dict[str, list[float]]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    )

    period_files = list(results_path.rglob("domain_period_metrics_*.json"))

    if not period_files:
        logger.info(f"Warning: No domain_period_metrics files found in {results_dir}")
        return {}

    logger.info(f"Found {len(period_files)} domain period metrics files")

    for period_file in period_files:
        logger.info(f"Loading: {period_file}")
        with open(period_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        for strategy, k_values in data.items():
            for k_value_str, domains in k_values.items():
                k_value = int(k_value_str)
                for domain_name, metrics in domains.items():
                    period_data[k_value][strategy][domain_name][
                        "auc_performance"
                    ].append(metrics["auc_performance"])
                    period_data[k_value][strategy][domain_name]["auc_dto"].append(
                        metrics["auc_dto"]
                    )
                    period_data[k_value][strategy][domain_name][
                        "final_performance"
                    ].append(metrics["final_performance"])
                    period_data[k_value][strategy][domain_name]["final_cost"].append(
                        metrics["final_cost"]
                    )
                    period_data[k_value][strategy][domain_name]["final_dto"].append(
                        metrics["final_dto"]
                    )

    return dict(period_data)


def print_domain_period_statistics_table(
    period_stats: dict[str, dict[str, dict[str, float | int]]], k_value: int
) -> None:
    """Print a formatted table of domain-period statistics for a given k_value."""
    all_domains: list[str] = []
    for strategy_data in period_stats.values():
        for d in strategy_data:
            if d not in all_domains:
                all_domains.append(d)

    logger.info(f"\n{'=' * 160}")
    logger.info(f"DOMAIN-PERIOD STATISTICS — K_VALUE = {k_value}")
    logger.info("=" * 160)

    for strategy in sorted(period_stats.keys()):
        logger.info(f"\n{strategy}:")
        logger.info("-" * 160)
        logger.info(
            f"  {'Domain':<20} {'AUC Perf (Mean±SE)':<25} {'AUC DTO (Mean±SE)':<25} "
            f"{'Final Perf (Mean±SE)':<25} {'Final Cost (Mean±SE)':<25} {'N':<5}"
        )
        logger.info("-" * 160)

        for domain in all_domains:
            if domain in period_stats[strategy]:
                data = period_stats[strategy][domain]
                auc_perf_str = f"{data['auc_performance_mean']:.4f} ± {data['auc_performance_stderr']:.4f}"
                auc_dto_str = (
                    f"{data['auc_dto_mean']:.4f} ± {data['auc_dto_stderr']:.4f}"
                )
                final_perf_str = f"{data['final_performance_mean']:.4f} ± {data['final_performance_stderr']:.4f}"
                final_cost_str = (
                    f"{data['final_cost_mean']:.2f} ± {data['final_cost_stderr']:.2f}"
                )
                logger.info(
                    f"  {domain:<20} {auc_perf_str:<25} {auc_dto_str:<25} "
                    f"{final_perf_str:<25} {final_cost_str:<25} {data['n_samples']:<5}"
                )

    logger.info("=" * 160)


def plot_domain_period_comparison_per_k(
    period_stats_per_k: dict[int, dict[str, dict[str, dict[str, float | int]]]],
    benchmark: str,
    n_seeds: int,
    output_dir: str = "plots_benchmarks/domain_period_comparison",
    figsize: tuple[int, int] = (18, 8),
) -> None:
    """
    Create domain-period comparison bar plots for each k_value.

    One bar group per strategy; one bar per domain within the group.
    Domains are shown in the order they appear in the data.

    Args:
        period_stats_per_k: {k_value: {strategy: {domain: {metric_mean/stderr: value}}}}
        benchmark: Benchmark name
        n_seeds: Number of seeds (used in title)
        output_dir: Directory to save plots
        figsize: Figure size (width, height)
    """
    output_path = Path(output_dir) / benchmark
    output_path.mkdir(exist_ok=True, parents=True)

    metrics = [
        ("auc_performance", "AUC Performance"),
        ("auc_dto", "AUC DTO"),
    ]

    for k_value, period_stats in period_stats_per_k.items():
        all_domains: list[str] = []
        for strategy_data in period_stats.values():
            for d in strategy_data:
                if d not in all_domains:
                    all_domains.append(d)

        n_domains = len(all_domains)
        bar_width = 0.8 / n_domains
        domain_alphas = [
            0.9 - i * (0.4 / max(n_domains - 1, 1)) for i in range(n_domains)
        ]

        fig, axes = plt.subplots(1, 2, figsize=figsize)  # type: ignore
        strategies = list(period_stats.keys())

        for col, (metric, metric_label) in enumerate(metrics):
            ax = axes[col]
            ascending = metric == "auc_dto"

            key = f"{metric}_mean"
            sorted_strategies = sorted(
                strategies,
                key=lambda s: (
                    sum(
                        period_stats[s][d][key]
                        for d in all_domains
                        if d in period_stats[s]
                    )
                    / max(sum(1 for d in all_domains if d in period_stats[s]), 1)
                ),
                reverse=not ascending,
            )

            x = np.arange(len(sorted_strategies))

            for d_idx, domain in enumerate(all_domains):
                means: list[float] = []
                stderrs: list[float] = []
                for strategy in sorted_strategies:
                    if domain in period_stats[strategy]:
                        means.append(period_stats[strategy][domain][f"{metric}_mean"])  # type: ignore
                        stderrs.append(period_stats[strategy][domain][f"{metric}_stderr"])  # type: ignore
                    else:
                        means.append(0.0)
                        stderrs.append(0.0)

                offset = (d_idx - (n_domains - 1) / 2) * bar_width
                bars = ax.bar(  # type: ignore
                    x + offset,
                    means,
                    width=bar_width,
                    yerr=stderrs,
                    label=domain,
                    capsize=3,
                    alpha=domain_alphas[d_idx],
                    error_kw=dict(elinewidth=1.2, capthick=1.2),
                )

                for bar, strategy in zip(bars, sorted_strategies):
                    color = get_colors_per_strategy([strategy])[0]
                    bar.set_facecolor(color)
                    bar.set_alpha(domain_alphas[d_idx])
                    bar.set_edgecolor(color)

            ax.set_xticks(x)  # type: ignore
            ax.set_xticklabels(sorted_strategies, rotation=25, ha="right", fontsize=9)  # type: ignore
            ax.set_xlabel("Strategy", fontsize=12, fontweight="bold")  # type: ignore
            ax.set_ylabel(metric_label, fontsize=12, fontweight="bold")  # type: ignore

            sort_desc = "high -> low" if not ascending else "low -> high"
            ax.set_title(  # type: ignore
                f"{metric_label}\n(sorted by avg across all domains: {sort_desc})",
                fontsize=13,
                fontweight="bold",
                pad=15,
            )
            ax.grid(True, axis="y", alpha=0.3, linestyle="--")  # type: ignore

            legend_handles = [
                Patch(facecolor="grey", alpha=domain_alphas[i], label=d)
                for i, d in enumerate(all_domains)
            ]
            ax.legend(handles=legend_handles, loc="upper right", fontsize=9, framealpha=0.9)  # type: ignore

        fig.suptitle(  # type: ignore
            f"Domain-Period Comparison (k={k_value}, {benchmark})\n"
            f"Domains: {' -> '.join(all_domains)}  |  Error bars = SE across {n_seeds} seeds",
            fontsize=14,
            fontweight="bold",
            y=0.98,
        )

        plt.tight_layout(rect=(0, 0, 1, 0.96))

        output_file = (
            output_path / f"domain_period_comparison_k{k_value}_{benchmark}.png"
        )
        plt.savefig(output_file, dpi=300, bbox_inches="tight")  # type: ignore
        plt.close(fig)
        logger.info(f"Domain period comparison plot saved to: {output_file}")
