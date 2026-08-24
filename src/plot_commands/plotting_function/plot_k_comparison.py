from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.style.use("ggplot")


def plot_k_comparison_barplot(
    stats_per_k: dict[int, dict[str, dict[str, float | int]]],
    benchmark: str,
    n_seeds: int,
    output_dir: str = "plots_benchmarks/k_comparison",
    figsize: tuple[int, int] = (16, 10),
):
    """
    Create a grouped bar plot showing AUC performance and AUC DTO for all k_values per strategy.

    Args:
        stats_per_k: Dictionary with k_value -> strategy -> statistics
        benchmark: Benchmark name
        n_seeds: Number of seeds
        output_dir: Directory to save plots
        figsize: Figure size (width, height)
    """
    output_path = Path(output_dir) / benchmark
    output_path.mkdir(exist_ok=True, parents=True)

    all_strategies: set[str] = set()
    for k_stats in stats_per_k.values():
        all_strategies.update(k_stats.keys())

    k_values = sorted(stats_per_k.keys())
    first_k = k_values[0]

    strategies_with_n = [
        (strat, stats_per_k[first_k][strat]["train_cost_final_mean"])
        for strat in all_strategies
        if strat in stats_per_k[first_k]
    ]
    strategies_with_n.sort(key=lambda x: x[1])
    strategies = [strat for strat, _ in strategies_with_n]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)  # type: ignore

    x = np.arange(len(strategies))
    width = 0.8 / len(k_values)

    colors_k = plt.cm.viridis(np.linspace(0, 1, len(k_values)))  # type: ignore

    for i, k_value in enumerate(k_values):
        means_perf: list[float] = []
        stderrs_perf: list[float] = []

        for strategy in strategies:
            if strategy in stats_per_k[k_value]:
                means_perf.append(
                    stats_per_k[k_value][strategy]["auc_performance_mean"]
                )
                stderrs_perf.append(
                    stats_per_k[k_value][strategy]["auc_performance_stderr"]
                )
            else:
                means_perf.append(0)
                stderrs_perf.append(0)

        offset = width * (i - len(k_values) / 2 + 0.5)
        ax1.bar(  # type: ignore
            x + offset,
            means_perf,
            width,
            yerr=stderrs_perf,
            label=f"k={k_value}",
            capsize=3,
            color=colors_k[i],
            alpha=0.8,
            edgecolor="black",
            linewidth=1,
        )

    ax1.set_xlabel("Strategy", fontsize=12, fontweight="bold")  # type: ignore
    ax1.set_ylabel("AUC Performance", fontsize=12, fontweight="bold")  # type: ignore
    ax1.set_title(  # type: ignore
        f"AUC Performance Comparison Across K Values ({benchmark.capitalize()})\n(Error bars show standard error across {n_seeds} seeds)",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax1.set_xticks(x)  # type: ignore
    strategies_with_samples = [
        f"{strat} (n={stats_per_k[k_value][strat]['train_cost_final_mean']:.1f})"  # type: ignore
        for strat in strategies
    ]
    ax1.set_xticklabels(strategies_with_samples, rotation=45, ha="right", fontsize=9)  # type: ignore
    ax1.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)  # type: ignore
    ax1.grid(True, alpha=0.3, linestyle="--", axis="y")  # type: ignore

    for i, k_value in enumerate(k_values):
        means_dto: list[float] = []
        stderrs_dto: list[float] = []

        for strategy in strategies:
            if (
                strategy in stats_per_k[k_value]
                and "auc_dto_mean" in stats_per_k[k_value][strategy]
            ):
                means_dto.append(stats_per_k[k_value][strategy]["auc_dto_mean"])
                stderrs_dto.append(stats_per_k[k_value][strategy]["auc_dto_stderr"])
            else:
                means_dto.append(0)
                stderrs_dto.append(0)

        offset = width * (i - len(k_values) / 2 + 0.5)
        ax2.bar(  # type: ignore
            x + offset,
            means_dto,
            width,
            yerr=stderrs_dto,
            label=f"k={k_value}",
            capsize=3,
            color=colors_k[i],
            alpha=0.8,
            edgecolor="black",
            linewidth=1,
        )

    ax2.set_xlabel("Strategy", fontsize=12, fontweight="bold")  # type: ignore
    ax2.set_ylabel("AUC DTO", fontsize=12, fontweight="bold")  # type: ignore
    ax2.set_title(  # type: ignore
        f"AUC DTO Comparison Across K Values ({benchmark.capitalize()})\n(Error bars show standard error across {n_seeds} seeds)",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax2.set_xticks(x)  # type: ignore
    ax2.set_xticklabels(strategies_with_samples, rotation=45, ha="right", fontsize=9)  # type: ignore
    ax2.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)  # type: ignore
    ax2.grid(True, alpha=0.3, linestyle="--", axis="y")  # type: ignore

    plt.tight_layout()

    output_file = output_path / f"k_comparison_barplot_{benchmark}.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")  # type: ignore
