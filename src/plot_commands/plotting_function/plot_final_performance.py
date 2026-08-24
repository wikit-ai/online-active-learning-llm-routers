from collections import defaultdict
from pathlib import Path
from typing import Any
import json

from plot_commands.plotting_function.utils import compute_mean_stderr, plot_bar_per_k  # type: ignore
from logging_config import logger


def load_final_global_performance_with_k(results_dir: str) -> dict[int, dict[str, Any]]:
    """
    Load final global performance (last step test_performance) for each strategy and k_value from run files.

    Args:
        results_dir: Path to results directory

    Returns:
        dict: {k_value: {strategy: [final_test_performance_values]}} across different seeds
    """
    results_path = Path(results_dir)
    final_perf_data: dict[int, dict[str, Any]] = defaultdict(lambda: defaultdict(list))  # type: ignore

    run_files = list(results_path.rglob("run_*.json"))

    if not run_files:
        raise ValueError(f"No run files found in {results_dir}")

    logger.info(f"Found {len(run_files)} run files")

    for run_file in run_files:
        with open(run_file, "r") as f:
            run_data = json.load(f)

        strategy_final: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)  # type: ignore
        for result in run_data.get("streaming_results", []):
            k_value = result["k_value"]
            strategy = result["strategy"]
            step = result["step"]
            test_perf = result["test_performance"]

            if strategy not in strategy_final[k_value]:
                strategy_final[k_value][strategy] = {
                    "step": step,
                    "test_performance": test_perf,
                }
            elif step > strategy_final[k_value][strategy]["step"]:
                strategy_final[k_value][strategy] = {
                    "step": step,
                    "test_performance": test_perf,
                }

        for k_value, strategies in strategy_final.items():
            for strategy, data in strategies.items():  # type: ignore
                final_perf_data[k_value][strategy].append(data["test_performance"])  # type: ignore

    return dict(final_perf_data)  # type: ignore


def compute_final_performance_statistics(
    final_perf_data: dict[str, Any],
) -> dict[str, dict[str, float]]:
    """Compute mean and standard error for final performance per strategy."""
    stats: dict[str, dict[str, float]] = {}
    for strategy, perf_values in final_perf_data.items():
        mean, stderr = compute_mean_stderr(perf_values)
        stats[strategy] = {  # type: ignore
            "final_perf_mean": mean,
            "final_perf_stderr": stderr,
            "n_samples": len(perf_values),
        }
    return stats


def plot_final_global_performance_per_k(
    final_stats_per_k: dict[int, dict[str, dict[str, float]]],
    benchmark: str,
    n_seeds: int,
    output_dir: str = "plots_benchmarks",
    figsize: tuple[int, int] = (12, 8),
):
    """Bar plots showing final global performance for each strategy, one per k_value."""
    plot_bar_per_k(
        final_stats_per_k,
        mean_key="final_perf_mean",
        stderr_key="final_perf_stderr",
        benchmark=benchmark,
        n_seeds=n_seeds,
        ylabel="Final Test Performance",
        title_prefix="Final Global Performance",
        filename_prefix="final_global_performance",
        output_dir=output_dir,
        figsize=figsize,
    )
