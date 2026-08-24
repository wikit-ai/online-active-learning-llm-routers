"""Sensitivity Analysis over cold-start length n_0.

Below n_0, all incoming queries are annotated unconditionally.
"""

from datasets_management.non_stationary_dataset import DatasetManagement
from experiment.main_stationary import run_main_experiment  # type: ignore
from experiment.results_tracker import ResultsTracker
from sampling_strategies import InferredSparsePerformanceStrategy

DEFAULT_COLD_START_VALUES = [25, 50, 75, 115, 150, 200]


def run_cold_start_ablation(
    seed: int,
    benchmark: str,
    embedding_model: str,
    budget: int,
    ds_loader: DatasetManagement,
    k_values: list[int],
    output_dir: str = "results/secondary_experiments/cold_start",
    cold_start_values: list[int] | None = None,
    cost_penalty: float = 0.25,
) -> ResultsTracker:
    """Run InferredSparsePerformanceStrategy with varying cold_start_n values."""
    cold_start_values = cold_start_values or DEFAULT_COLD_START_VALUES

    strategies = {
        f"cold_start_{n}": InferredSparsePerformanceStrategy(
            cold_start_n=n, budget=budget, seed=seed
        )
        for n in cold_start_values
    }

    return run_main_experiment(
        budget=budget,
        seed=seed,
        benchmark=benchmark,
        embedding_model=embedding_model,
        ds_loader=ds_loader,
        strategies=strategies,  # type: ignore
        k=k_values,
        output_dir=output_dir,
        cost_penalty=cost_penalty,
    )
