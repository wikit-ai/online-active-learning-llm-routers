import argparse
from pathlib import Path
from typing import Any

from plot_commands.constants import get_results_dir
from plot_commands.plotting_function.plot_auc_vs_cost import (
    compute_statistics,
    load_all_data_with_k,
    plot_auc_dto_vs_cost_per_k,
    plot_auc_vs_cost_best_k,
    plot_auc_vs_cost_line_families,
    plot_auc_vs_cost_per_k,
)
from plot_commands.plotting_function.plot_domain_non_stationary import (
    compute_period_statistics,
    load_domain_period_metrics_with_k,
    plot_domain_period_comparison_per_k,
    print_domain_period_statistics_table,
)
from plot_commands.plotting_function.plot_final_performance import (
    compute_final_performance_statistics,
    load_final_global_performance_with_k,
    plot_final_global_performance_per_k,
)
from plot_commands.plotting_function.plot_k_comparison import plot_k_comparison_barplot
from plot_commands.plotting_function.plot_latex_table import generate_latex_table  # type: ignore
from plot_commands.plotting_function.plot_macro_auc_avg import (
    compute_macro_statistics,
    load_macro_avg_data_with_k,
    plot_macro_auc_per_k,
)
from plot_commands.plotting_function.plot_performance_vs_step import (
    plot_performance_vs_step_per_strategy,
)
from plot_commands.plotting_function.utils import (
    filter_strategies_by_prefix,
    print_statistics_table,
)
from logging_config import logger


def _compute_n_seeds(stats_per_k: dict[int, dict[str, dict[str, float | int]]]) -> int:
    """Return the largest sample count found across all k values and strategies."""
    return int(
        max(
            (s["n_samples"] for stats in stats_per_k.values() for s in stats.values()),
            default=0,
        )
        if stats_per_k
        else 0
    )


def _filter_all_k(
    data_per_k: dict[int, Any], prefixes: list[str] | None
) -> dict[int, Any]:
    """Apply prefix filtering to every k value, in place. No-op without prefixes."""
    if not prefixes:
        return data_per_k
    logger.info(f"Filtering strategies by prefixes: {prefixes}")
    for k_value in data_per_k:
        data_per_k[k_value] = filter_strategies_by_prefix(data_per_k[k_value], prefixes)
    return data_per_k


def run_plots(
    results_dir: str,
    benchmark: str,
    experiment_type: str,
    strategy_prefixes: list[str] | None = None,
) -> None:
    """Generate every figure and the LaTeX table for one benchmark run.

    Covers AUC vs cost, final performance, performance vs step, plus the
    experiment-specific extras (macro AUC and k comparison for stationary,
    domain-period analysis for domain_shift).

    Args:
        results_dir: Directory holding the run JSON files
        benchmark: Benchmark name, used in titles and output paths
        experiment_type: "stationary" or "domain_shift"
        strategy_prefixes: Keep only strategies starting with these prefixes
    """
    output_base = f"plots_benchmarks/{experiment_type}"

    data_per_k = _filter_all_k(load_all_data_with_k(results_dir), strategy_prefixes)

    stats_per_k: dict[int, dict[str, dict[str, float | int]]] = {
        k: compute_statistics(data) for k, data in data_per_k.items()
    }
    n_seeds = _compute_n_seeds(stats_per_k)

    for k_value, stats in stats_per_k.items():
        print_statistics_table(stats, k_value)

    plot_auc_vs_cost_per_k(
        stats_per_k,
        benchmark,
        n_seeds,
        output_dir=f"{output_base}/perf_vs_cost",
    )
    plot_auc_dto_vs_cost_per_k(
        stats_per_k,
        benchmark,
        n_seeds,
        output_dir=f"{output_base}/dto_vs_cost",
    )
    plot_auc_vs_cost_best_k(
        results_dir,
        benchmark,
        n_seeds,
        output_dir=f"{output_base}/perf_vs_cost",
    )

    if experiment_type == "stationary":
        plot_auc_vs_cost_line_families(stats_per_k, benchmark, n_seeds)

    final_perf_data_per_k = _filter_all_k(
        load_final_global_performance_with_k(results_dir), strategy_prefixes
    )
    final_stats_per_k: dict[int, dict[str, dict[str, float]]] = {
        k: compute_final_performance_statistics(data)
        for k, data in final_perf_data_per_k.items()
    }
    plot_final_global_performance_per_k(
        final_stats_per_k,
        benchmark,
        n_seeds,
        output_dir=f"{output_base}/final_performance",
    )

    generate_latex_table(
        stats_per_k,
        final_stats_per_k,
        benchmark,
        output_dir=output_base,
        raw_per_k=data_per_k,
        raw_final_per_k=final_perf_data_per_k,
    )

    if experiment_type == "stationary":
        results_path = Path(results_dir)
        perf_files = list(results_path.rglob("perf_per_dataset_*.json"))

        if perf_files:
            macro_data_per_k: dict[int, dict[str, Any]] = _filter_all_k(
                load_macro_avg_data_with_k(perf_files), strategy_prefixes
            )
            macro_stats_per_k: dict[int, dict[str, dict[str, float | int]]] = {
                k: compute_macro_statistics(data)
                for k, data in macro_data_per_k.items()
            }
            plot_macro_auc_per_k(macro_stats_per_k, benchmark, n_seeds)
        else:
            logger.info(f"Warning: No perf_per_dataset files found in {results_dir}")

        plot_k_comparison_barplot(stats_per_k, benchmark, n_seeds)

    # --- Domain-shift only: domain-period analysis ---
    if experiment_type == "domain_shift":
        logger.info("\n" + "=" * 80)
        logger.info("Loading domain-period metrics...")
        domain_period_data_per_k = load_domain_period_metrics_with_k(results_dir)

        if domain_period_data_per_k:
            domain_period_data_per_k = _filter_all_k(
                domain_period_data_per_k, strategy_prefixes
            )
            domain_period_stats_per_k = {
                k: compute_period_statistics(data)
                for k, data in domain_period_data_per_k.items()
            }
            for k_value, dps in domain_period_stats_per_k.items():
                print_domain_period_statistics_table(dps, k_value)

            plot_domain_period_comparison_per_k(
                domain_period_stats_per_k,
                benchmark,
                n_seeds,
                output_dir=f"{output_base}/domain_period_comparison",
            )
        else:
            logger.info(
                "Warning: No domain-period data found. Skipping domain comparison."
            )

    logger.info("\nGenerating performance vs step plots...")
    results_path = Path(results_dir)
    run_files = list(results_path.rglob("run_*.json"))

    if run_files:
        logger.info(f"Processing {len(run_files)} run files...")
        try:
            plot_performance_vs_step_per_strategy(
                json_paths=[str(f) for f in run_files],
                output_dir=f"{output_base}/performance_vs_step/{benchmark}",
            )
        except Exception as e:
            logger.info(f"Warning: Failed to generate plots: {e}")
    else:
        logger.info("Warning: No run files found for performance vs step plots")

    logger.info("\nDone!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot AUC Performance vs Training Cost for a benchmark"
    )
    parser.add_argument(
        "--penalty",
        type=float,
        required=True,
        help="Cost penalty value",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        choices=["sprout", "routerbench", "fusionbench", "embedllm"],
        default="sprout",
    )
    parser.add_argument(
        "--experiment-type",
        type=str,
        choices=["stationary", "domain_shift"],
        default="stationary",
        help="Experiment type (default: stationary)",
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
        help="Only include strategies starting with these prefixes",
    )

    args = parser.parse_args()

    if args.results_dir:
        results_dir = args.results_dir
    else:
        results_dir = get_results_dir(
            args.benchmark, args.penalty, args.experiment_type
        )

    logger.info(f"Using benchmark: {args.benchmark}")
    logger.info(f"Experiment type: {args.experiment_type}")
    logger.info(f"Results dir: {results_dir}")

    run_plots(
        results_dir=results_dir,
        benchmark=args.benchmark,
        experiment_type=args.experiment_type,
        strategy_prefixes=args.strategy_prefixes,
    )
