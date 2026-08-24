import argparse
from pathlib import Path
from typing import Any


from plot_commands.plotting_function.plot_auc_vs_cost import (
    compute_statistics,
    load_all_data_with_k,
    plot_auc_dto_vs_cost_per_k,
    plot_auc_vs_cost_best_k,
    plot_auc_vs_cost_per_k,
)
from plot_commands.plotting_function.plot_final_performance import (
    compute_final_performance_statistics,
    load_final_global_performance_with_k,
    plot_final_global_performance_per_k,
)
from plot_commands.plotting_function.plot_k_comparison import plot_k_comparison_barplot
from plot_commands.plotting_function.plot_macro_auc_avg import (
    compute_macro_statistics,
    load_macro_avg_data_with_k,
    plot_macro_auc_per_k,
)
from plot_commands.plotting_function.plot_latex_table import generate_latex_table  # type: ignore
from plot_commands.plotting_function.plot_performance_vs_step import (
    plot_performance_vs_step_per_strategy,
)


def filter_strategies_by_prefix(
    data: dict[str, Any], prefixes: list[str] | None
) -> dict[str, Any]:
    """
    Filter strategies by their prefixes.

    Args:
        data: Dictionary of strategy data
        prefixes: List of prefixes to include (e.g., ['rand_', 'act_unc'])

    Returns:
        dict: Filtered data containing only strategies matching the prefixes
    """
    if prefixes is None:
        return data

    filtered_data: dict[str, Any] = {}
    for strategy, metrics in data.items():
        if any(strategy.startswith(prefix) for prefix in prefixes):
            filtered_data[strategy] = metrics

    return filtered_data


def print_statistics_table(stats: dict[str, dict[str, float | int]], k_value: int):
    """Print a formatted table of statistics."""
    # Check if any strategy has DTO data
    has_dto = any("auc_dto_mean" in data for data in stats.values())

    print(f"\n{'=' * 145}")
    print(f"K_VALUE = {k_value}")
    print("=" * 145)

    if has_dto:
        print(
            f"{'Strategy':<30} {'AUC Perf (Mean±SE)':<25} {'AUC DTO (Mean±SE)':<25} {'Train Cost Final (Mean±SE)':<30} {'N Samples':<10}"
        )
        print("=" * 145)
    else:
        print(
            f"{'Strategy':<30} {'AUC Perf (Mean±SE)':<25} {'Train Cost Final (Mean±SE)':<30} {'N Samples':<10}"
        )
        print("=" * 110)

    # Sort by AUC performance (descending)
    sorted_strategies = sorted(
        stats.items(), key=lambda x: x[1]["auc_performance_mean"], reverse=True
    )

    for strategy, data in sorted_strategies:
        perf_str = (
            f"{data['auc_performance_mean']:.4f} ± {data['auc_performance_stderr']:.4f}"
        )
        cost_str = f"{data['train_cost_final_mean']:.2f} ± {data['train_cost_final_stderr']:.2f}"

        if has_dto and "auc_dto_mean" in data:
            dto_str = f"{data['auc_dto_mean']:.4f} ± {data['auc_dto_stderr']:.4f}"
            print(
                f"{strategy:<30} {perf_str:<25} {dto_str:<25} {cost_str:<30} {data['n_samples']:<10}"
            )
        elif has_dto:
            print(
                f"{strategy:<30} {perf_str:<25} {'N/A':<25} {cost_str:<30} {data['n_samples']:<10}"
            )
        else:
            print(
                f"{strategy:<30} {perf_str:<25} {cost_str:<30} {data['n_samples']:<10}"
            )

    if has_dto:
        print("=" * 145)
    else:
        print("=" * 110)


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Plot AUC Performance vs Training Cost for different benchmarks with k_value support"
    )
    parser.add_argument(
        "--penalty",
        type=float,
        help="Which cost penalty to plot",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        choices=["sprout", "routerbench", "fusionbench", "embedllm"],
        default="sprout",
        help="Which benchmark to use (sprout or routerbench). Default: sprout",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Custom results directory (overrides benchmark default)",
    )
    parser.add_argument(
        "--strategy-prefixes",
        type=str,
        nargs="+",
        default=None,
        help="List of strategy prefixes to include (e.g., 'rand_' 'act_unc' 'oracle'). If not specified, all strategies are included.",
    )

    args = parser.parse_args()

    # Configure paths based on benchmark
    if args.results_dir:
        results_dir = args.results_dir
    else:
        if args.benchmark == "sprout":
            results_dir = f"results/{args.penalty}/stationary/snowflake-arctic-embed-m-v2.0/Sprout"
        elif args.benchmark == "routerbench":
            results_dir = f"results/{args.penalty}/stationary/snowflake-arctic-embed-m-v2.0/RouterBench"
        elif args.benchmark == "embedllm":
            results_dir = f"results/{args.penalty}/stationary/snowflake-arctic-embed-m-v2.0/EmbedLLM"
        elif args.benchmark == "fusionbench":
            results_dir = f"results/{args.penalty}/stationary/snowflake-arctic-embed-m-v2.0/FusionBench"

        else:
            raise ValueError(f"Unknown benchmark: {args.benchmark}")

    print(f"Using benchmark: {args.benchmark}")
    print()

    data_per_k = load_all_data_with_k(results_dir)

    # Filter strategies by prefix if specified
    if args.strategy_prefixes:
        print(f"\nFiltering strategies by prefixes: {args.strategy_prefixes}")
        for k_value in data_per_k:
            data_per_k[k_value] = filter_strategies_by_prefix(
                data_per_k[k_value], args.strategy_prefixes
            )
            print(
                f"K={k_value} - Strategies after filtering: {list(data_per_k[k_value].keys())}"
            )

    stats_per_k: dict[int, dict[str, dict[str, float | int]]] = {}
    for k_value, data in data_per_k.items():
        stats_per_k[k_value] = compute_statistics(data)

    n_seeds = int(
        max(
            (s["n_samples"] for stats in stats_per_k.values() for s in stats.values()),
            default=0,
        )
        if stats_per_k
        else 0
    )

    for k_value, stats in stats_per_k.items():
        print_statistics_table(stats, k_value)

    plot_auc_vs_cost_per_k(stats_per_k, args.benchmark, n_seeds)

    plot_auc_dto_vs_cost_per_k(stats_per_k, args.benchmark, n_seeds)

    plot_auc_vs_cost_best_k(results_dir, args.benchmark, n_seeds)

    final_perf_data_per_k = load_final_global_performance_with_k(results_dir)

    if args.strategy_prefixes:
        for k_value in final_perf_data_per_k:
            final_perf_data_per_k[k_value] = filter_strategies_by_prefix(
                final_perf_data_per_k[k_value], args.strategy_prefixes
            )

    final_stats_per_k: dict[int, dict[str, dict[str, float]]] = {}
    for k_value, data in final_perf_data_per_k.items():
        final_stats_per_k[k_value] = compute_final_performance_statistics(data)

    plot_final_global_performance_per_k(final_stats_per_k, args.benchmark, n_seeds)

    generate_latex_table(
        stats_per_k,
        final_stats_per_k,
        args.benchmark,
        n_seeds,
        output_dir="plots_benchmarks/stationary",
    )

    results_path = Path(results_dir)
    perf_files = list(results_path.rglob("perf_per_dataset_*.json"))

    if not perf_files:
        print(f"Warning: No perf_per_dataset files found in {results_dir}")
    else:
        macro_data_per_k: dict[int, dict[str, Any]] = load_macro_avg_data_with_k(
            perf_files
        )

        if args.strategy_prefixes:
            for k_value in macro_data_per_k:
                macro_data_per_k[k_value] = filter_strategies_by_prefix(
                    macro_data_per_k[k_value], args.strategy_prefixes
                )

        macro_stats_per_k: dict[int, dict[str, dict[str, float | int]]] = {}
        for k_value, data in macro_data_per_k.items():
            macro_stats_per_k[k_value] = compute_macro_statistics(data)

        plot_macro_auc_per_k(macro_stats_per_k, args.benchmark, n_seeds)
    plot_k_comparison_barplot(stats_per_k, args.benchmark, n_seeds)

    # Generate performance vs step plots from all run files combined
    print("\nGenerating performance vs step plots...")
    run_files = list(results_path.rglob("run_*.json"))

    if run_files:
        print(f"Processing {len(run_files)} run files...")
        try:
            plot_performance_vs_step_per_strategy(
                json_paths=[str(f) for f in run_files],
                output_dir=f"plots_benchmarks/performance_vs_step/{args.benchmark}",
            )
        except Exception as e:
            print(f"Warning: Failed to generate plots: {e}")
    else:
        print("Warning: No run files found for performance vs step plots")

    print("\nDone!")
