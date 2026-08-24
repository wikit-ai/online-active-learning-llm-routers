import argparse
from pathlib import Path
from typing import Any

from logging_config import logger
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
from plot_commands.plotting_function.plot_domain_non_stationary import (
    compute_period_statistics,
    load_domain_period_metrics_with_k,
    plot_domain_period_comparison_per_k,
    print_domain_period_statistics_table,
)
from plot_commands.plotting_function.plot_latex_table import generate_latex_table  # type: ignore
from plot_commands.plotting_function.plot_performance_vs_step import (
    plot_performance_vs_step_per_strategy,
)


def filter_strategies_by_prefix(
    data: dict[str, Any], prefixes: list[str] | None
) -> dict[str, Any]:
    if prefixes is None:
        return data
    return {
        strategy: metrics
        for strategy, metrics in data.items()
        if any(strategy.startswith(p) for p in prefixes)
    }


def print_statistics_table(stats: dict[str, dict[str, float | int]], k_value: int):
    """Print a formatted table of global statistics."""
    has_dto = any("auc_dto_mean" in data for data in stats.values())

    print(f"\n{'=' * 145}")
    print(f"K_VALUE = {k_value}")
    print("=" * 145)

    if has_dto:
        print(
            f"{'Strategy':<30} {'AUC Perf (Mean±SE)':<25} {'AUC DTO (Mean±SE)':<25} "
            f"{'Train Cost Final (Mean±SE)':<30} {'N Samples':<10}"
        )
        print("=" * 145)
    else:
        print(
            f"{'Strategy':<30} {'AUC Perf (Mean±SE)':<25} {'Train Cost Final (Mean±SE)':<30} {'N Samples':<10}"
        )
        print("=" * 110)

    for strategy, data in sorted(
        stats.items(), key=lambda x: x[1]["auc_performance_mean"], reverse=True
    ):
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

    print("=" * (145 if has_dto else 110))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot AUC Performance vs Training Cost for domain-shift experiments"
    )
    parser.add_argument(
        "--penalty", type=float, required=True, help="Cost penalty value"
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        choices=["sprout", "routerbench", "embedllm", "fusionbench"],
        default="sprout",
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
    )

    args = parser.parse_args()

    if args.results_dir:
        results_dir = args.results_dir
    else:
        benchmark_map = {
            "sprout": "Sprout",
            "routerbench": "RouterBench",
            "embedllm": "EmbedLLM",
            "fusionbench": "FusionBench",
        }
        results_dir = (
            f"results/{args.penalty}/domain_shift/"
            f"snowflake-arctic-embed-m-v2.0/{benchmark_map[args.benchmark]}"
        )

    print(f"Using benchmark: {args.benchmark}")
    print(f"Results dir: {results_dir}")
    print()

    # --- Global AUC vs Cost ---
    data_per_k = load_all_data_with_k(results_dir)

    if args.strategy_prefixes:
        print(f"Filtering strategies by prefixes: {args.strategy_prefixes}")
        for k_value in data_per_k:
            data_per_k[k_value] = filter_strategies_by_prefix(
                data_per_k[k_value], args.strategy_prefixes
            )

    stats_per_k: dict[int, dict[str, dict[str, float | int]]] = {
        k_value: compute_statistics(data) for k_value, data in data_per_k.items()
    }

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

    plot_auc_vs_cost_per_k(
        stats_per_k,
        args.benchmark,
        n_seeds,
        output_dir="plots_benchmarks/domain_shift/perf_vs_cost",
    )
    plot_auc_dto_vs_cost_per_k(
        stats_per_k,
        args.benchmark,
        n_seeds,
        output_dir="plots_benchmarks/domain_shift/dto_vs_cost",
    )
    plot_auc_vs_cost_best_k(
        results_dir,
        args.benchmark,
        n_seeds,
        output_dir="plots_benchmarks/domain_shift/perf_vs_cost",
    )

    # --- Final performance ---
    final_perf_data_per_k = load_final_global_performance_with_k(results_dir)
    if args.strategy_prefixes:
        for k_value in final_perf_data_per_k:
            final_perf_data_per_k[k_value] = filter_strategies_by_prefix(
                final_perf_data_per_k[k_value], args.strategy_prefixes
            )

    final_stats_per_k: dict[int, dict[str, dict[str, float]]] = {
        k_value: compute_final_performance_statistics(data)
        for k_value, data in final_perf_data_per_k.items()
    }
    plot_final_global_performance_per_k(
        final_stats_per_k,
        args.benchmark,
        n_seeds,
        output_dir="plots_benchmarks/domain_shift/final_performance",
    )

    generate_latex_table(
        stats_per_k,
        final_stats_per_k,
        args.benchmark,
        n_seeds,
        output_dir="plots_benchmarks/domain_shift",
    )

    # --- Domain-period analysis ---
    print("\n" + "=" * 80)
    print("Loading domain-period metrics...")
    domain_period_data_per_k = load_domain_period_metrics_with_k(results_dir)

    if domain_period_data_per_k:
        if args.strategy_prefixes:
            for k_value in domain_period_data_per_k:
                domain_period_data_per_k[k_value] = filter_strategies_by_prefix(
                    domain_period_data_per_k[k_value], args.strategy_prefixes
                )

        domain_period_stats_per_k: dict[
            int, dict[str, dict[str, dict[str, float | int]]]
        ] = {
            k_value: compute_period_statistics(domain_data)
            for k_value, domain_data in domain_period_data_per_k.items()
        }

        for k_value, domain_period_stats in domain_period_stats_per_k.items():
            print_domain_period_statistics_table(domain_period_stats, k_value)

        plot_domain_period_comparison_per_k(
            domain_period_stats_per_k,
            args.benchmark,
            n_seeds,
            output_dir="plots_benchmarks/domain_shift/domain_period_comparison",
        )

    else:
        print("Warning: No domain-period data found. Skipping domain comparison.")

    # --- Performance vs step ---
    print("\nGenerating performance vs step plots...")
    results_path = Path(results_dir)
    run_files = list(results_path.rglob("run_*.json"))

    if run_files:
        print(f"Processing {len(run_files)} run files...")
        try:
            plot_performance_vs_step_per_strategy(
                json_paths=[str(f) for f in run_files],
                output_dir=f"plots_benchmarks/domain_shift/performance_vs_step/{args.benchmark}",
            )
        except Exception as e:
            logger.error(f"Warning: Failed to generate plots: {e}")
    else:
        print("Warning: No run files found for performance vs step plots")

    print("\nDone!")
