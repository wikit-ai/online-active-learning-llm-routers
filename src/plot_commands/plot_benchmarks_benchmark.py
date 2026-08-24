"""Plot AUC Performance vs Corpus Annotated for all 4 benchmarks in a 2x2 grid.

Supports both stationary and domain_shift experiments via --experiment-type.
Replaces the former plot_benchmarks_benchmark_stationary.py and
plot_benchmarks_benchmark_non_stationary.py.
"""

import argparse
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from plot_commands.constants import (
    BENCHMARKS,
    BENCHMARK_LABELS,
    FAMILY_COLORS,
    FAMILY_LINESTYLES,
    STRATEGY_NAMES,
    STRATEGY_ORDER,
    get_results_dir,
)
from plot_commands.plotting_function.plot_auc_vs_cost import (
    VARIANT_LABELS,
    get_strategy_family,
    get_variant_sort_key,
)
from plot_commands.plotting_function.utils import (
    apply_standard_filters,
    get_best_k_stats,
    get_colors_per_strategy,
    get_markers_per_strategy,
    order_strategies,
)

plt.style.use("ggplot")


def plot_all_benchmarks(
    penalty: float,
    experiment_type: str = "stationary",
    output_dir: str | None = None,
    strategy_prefixes: list[str] | None = None,
):
    if output_dir is None:
        output_dir = f"plots_benchmarks/{experiment_type}"

    fig, axes = plt.subplots(2, 2, figsize=(18, 10), squeeze=False)
    axes_flat = axes.flatten()

    all_handles: dict[str, object] = {}

    for idx, benchmark in enumerate(BENCHMARKS):
        ax = axes_flat[idx]
        results_dir = get_results_dir(benchmark, penalty, experiment_type)

        print(f"Loading {benchmark} from {results_dir}...")
        try:
            best_stats = get_best_k_stats(results_dir)
        except ValueError as e:
            ax.text(
                0.5, 0.5, f"No data\n{e}",
                ha="center", va="center", transform=ax.transAxes,
            )
            ax.set_title(BENCHMARK_LABELS[benchmark], fontsize=14, fontweight="bold")
            continue

        best_stats = apply_standard_filters(best_stats, strategy_prefixes)
        strategies = order_strategies(best_stats)

        x_vals = [best_stats[s]["train_cost_final_mean"] for s in strategies]
        y_vals = [best_stats[s]["auc_performance_mean"] for s in strategies]
        x_errs = [best_stats[s]["train_cost_final_stderr"] for s in strategies]
        y_errs = [best_stats[s]["auc_performance_stderr"] for s in strategies]

        colors = get_colors_per_strategy(strategies)
        markers = get_markers_per_strategy(strategies)

        for i, strategy in enumerate(strategies):
            handle = ax.errorbar(
                x_vals[i], y_vals[i],
                xerr=x_errs[i], yerr=y_errs[i],
                fmt=markers[i], color=colors[i],
                markersize=13, capsize=5, capthick=2,
                label=strategy, alpha=0.8, linewidth=2,
            )
            if strategy not in all_handles:
                all_handles[strategy] = handle

            ax.annotate(
                STRATEGY_NAMES.get(strategy, strategy),
                (x_vals[i], y_vals[i]),
                xytext=(10, 10), textcoords="offset points",
                fontsize=12, fontweight="bold", alpha=0.95,
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor=colors[i], alpha=0.2, edgecolor="none",
                ),
            )

        ax.set_xlabel("Size Corpus Annotated", fontsize=13, fontweight="bold")
        ax.set_ylabel("AUC Performance", fontsize=13, fontweight="bold")
        ax.tick_params(axis="x", labelsize=12)
        ax.tick_params(axis="y", labelsize=12)
        ax.set_title(BENCHMARK_LABELS[benchmark], fontsize=16, fontweight="bold", pad=8)
        ax.grid(True, alpha=0.3, linestyle="--")

    fig.tight_layout()

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    out_file = output_path / f"all_benchmarks_auc_perf_vs_cost_penalty{penalty}.png"
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    print(f"\nSaved figure to {out_file}")
    plt.close(fig)


def plot_all_benchmarks_line(
    penalty: float,
    experiment_type: str = "stationary",
    output_dir: str | None = None,
    strategy_prefixes: list[str] | None = None,
):
    """2x2 line plot: one line per strategy family, points = hyperparameter variants."""
    if output_dir is None:
        output_dir = f"plots_benchmarks/{experiment_type}"

    family_order: list[str] = []
    seen_families: set[str] = set()
    for s in STRATEGY_ORDER:
        family, _ = get_strategy_family(s)
        if family not in seen_families:
            family_order.append(family)
            seen_families.add(family)

    fig, axes = plt.subplots(2, 2, figsize=(18, 10), squeeze=False)
    axes_flat = axes.flatten()

    all_handles: dict[str, object] = {}

    for idx, benchmark in enumerate(BENCHMARKS):
        ax = axes_flat[idx]
        results_dir = get_results_dir(benchmark, penalty, experiment_type)

        print(f"Loading {benchmark} from {results_dir}...")
        try:
            best_stats = get_best_k_stats(results_dir)
        except ValueError as e:
            ax.text(
                0.5, 0.5, f"No data\n{e}",
                ha="center", va="center", transform=ax.transAxes,
            )
            ax.set_title(BENCHMARK_LABELS[benchmark], fontsize=14, fontweight="bold")
            continue

        best_stats = apply_standard_filters(best_stats, strategy_prefixes)

        families: dict[str, list[str]] = defaultdict(list)
        for strategy in best_stats:
            family, _ = get_strategy_family(strategy)
            if family in seen_families:
                families[family].append(strategy)

        for family in family_order:
            if family not in families:
                continue

            variants = sorted(families[family], key=get_variant_sort_key)
            color = FAMILY_COLORS.get(family, get_colors_per_strategy([family])[0])
            linestyle = FAMILY_LINESTYLES.get(family, "-")
            family_label = STRATEGY_NAMES.get(family, family)

            x_arr = np.array([best_stats[s]["train_cost_final_mean"] for s in variants])
            y_arr = np.array([best_stats[s]["auc_performance_mean"] for s in variants])
            y_err = np.array([best_stats[s]["auc_performance_stderr"] for s in variants])

            (line,) = ax.plot(
                x_arr, y_arr,
                color=color, linewidth=2, linestyle=linestyle,
                marker="o", markersize=8, label=family_label,
                alpha=0.85, zorder=3,
            )
            if family not in all_handles:
                all_handles[family] = line

            ax.fill_between(
                x_arr, y_arr - y_err, y_arr + y_err,
                color=color, alpha=0.12, zorder=2,
            )

            for strategy, x, y in zip(variants, x_arr, y_arr):
                point_label = VARIANT_LABELS.get(
                    strategy, get_strategy_family(strategy)[1]
                )
                ax.annotate(
                    point_label, (x, y),
                    xytext=(6, 6), textcoords="offset points",
                    fontsize=9, fontweight="bold", color=color, alpha=0.95,
                )

        ax.set_xlabel("Size Corpus Annotated", fontsize=13, fontweight="bold")
        ax.set_ylabel("AUC Performance", fontsize=13, fontweight="bold")
        ax.tick_params(axis="x", labelsize=12)
        ax.tick_params(axis="y", labelsize=12)
        ax.set_title(BENCHMARK_LABELS[benchmark], fontsize=16, fontweight="bold", pad=8)
        ax.grid(True, alpha=0.3, linestyle="--")

    handles = [all_handles[f] for f in family_order if f in all_handles]
    labels = [STRATEGY_NAMES.get(f, f) for f in family_order if f in all_handles]
    fig.legend(
        handles, labels,
        loc="center left", ncol=1, fontsize=11,
        bbox_to_anchor=(1.01, 0.5), frameon=True, title="Strategy family",
    )

    fig.tight_layout(rect=(0, 0, 1, 1))

    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    out_file = output_path / f"all_benchmarks_line_families_penalty{penalty}.png"
    fig.savefig(out_file, dpi=300, bbox_inches="tight")
    print(f"\nSaved -> {out_file}")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot AUC Performance vs Corpus Annotated for all 4 benchmarks in a 2x2 grid."
    )
    parser.add_argument(
        "--penalty", type=float, required=True,
        help="Cost penalty value (used to locate results directory)",
    )
    parser.add_argument(
        "--experiment-type", type=str,
        choices=["stationary", "domain_shift"],
        default="stationary",
        help="Experiment type (default: stationary)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Directory to save the figure (default: plots_benchmarks/<experiment_type>)",
    )
    parser.add_argument(
        "--strategy-prefixes", type=str, nargs="+", default=None,
        help="Only include strategies starting with these prefixes",
    )

    args = parser.parse_args()

    plot_all_benchmarks(
        penalty=args.penalty,
        experiment_type=args.experiment_type,
        output_dir=args.output_dir,
        strategy_prefixes=args.strategy_prefixes,
    )

    plot_all_benchmarks_line(
        penalty=args.penalty,
        experiment_type=args.experiment_type,
        output_dir=args.output_dir,
        strategy_prefixes=args.strategy_prefixes,
    )

    print("\nDone!")
