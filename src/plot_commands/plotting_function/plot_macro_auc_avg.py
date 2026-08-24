from collections import defaultdict
from pathlib import Path
from typing import Any
import json

import numpy as np

from plot_commands.plotting_function.utils import compute_mean_stderr, plot_bar_per_k  # type: ignore


def load_macro_avg_data_with_k(
    perf_per_dataset_files: list[Path],
) -> dict[int, dict[str, Any]]:
    """
    Load performance per dataset files and compute macro-averaged AUC for each strategy and k_value.

    Args:
        perf_per_dataset_files: List of paths to perf_per_dataset_*.json files

    Returns:
        dict: {k_value: {strategy: [macro_avg_auc_values]}} across different runs
    """
    macro_data: dict[int, dict[str, Any]] = defaultdict(lambda: defaultdict(list))  # type: ignore

    for perf_file in perf_per_dataset_files:

        with open(perf_file, "r") as f:
            perf_data = json.load(f)

        for strategy, k_values_dict in perf_data.items():
            for k_value_str, datasets_perf in k_values_dict.items():
                k_value = int(k_value_str)
                final_perfs: list[float] = []
                for _, dataset_data in datasets_perf.items():
                    if isinstance(dataset_data, dict) and "perf" in dataset_data:
                        perf_values: list[float] = dataset_data["perf"]  # type: ignore
                        if perf_values:
                            final_perfs.append(perf_values[-1])  # type: ignore

                if final_perfs:
                    macro_avg_auc = np.mean(final_perfs)
                    macro_data[k_value][strategy].append(macro_avg_auc)  # type: ignore

    return dict(macro_data)  # type: ignore


def compute_macro_statistics(
    macro_data: dict[str, Any],
) -> dict[str, dict[str, float | int]]:
    """Compute mean and standard error for macro-averaged AUC per strategy."""
    stats: dict[str, dict[str, float | int]] = {}
    for strategy, auc_values in macro_data.items():
        mean, stderr = compute_mean_stderr(auc_values)
        stats[strategy] = {  # type: ignore
            "macro_auc_mean": mean,
            "macro_auc_stderr": stderr,
            "n_samples": len(auc_values),
        }
    return stats


def plot_macro_auc_per_k(
    macro_stats_per_k: dict[int, dict[str, dict[str, float | int]]],
    benchmark: str,
    n_seeds: int,
    output_dir: str = "plots_benchmarks/macro_auc_avg",
    figsize: tuple[int, int] = (12, 8),
):
    """Bar plots comparing macro-averaged AUC across strategies, one per k_value."""
    plot_bar_per_k(
        macro_stats_per_k,
        mean_key="macro_auc_mean",
        stderr_key="macro_auc_stderr",
        benchmark=benchmark,
        n_seeds=n_seeds,
        ylabel="Macro-Averaged AUC",
        title_prefix="Macro-Averaged AUC Performance Across Datasets",
        filename_prefix="macro_auc_comparison",
        output_dir=output_dir,
        figsize=figsize,
    )
