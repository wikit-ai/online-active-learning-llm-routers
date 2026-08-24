"""
f_theta is retrained every delta new annotations.
"""

from datasets_management.non_stationary_dataset import DatasetManagement
from experiment.main_stationary import run_main_experiment  # type: ignore
from experiment.results_tracker import ResultsTracker
from sampling_strategies import InferredSparsePerformanceStrategy

DEFAULT_RETRAIN_INTERVAL_VALUES = [1, 5, 10, 20, 50]
DEFAULT_COLD_START_N = 115


def run_retrain_interval_ablation(
    seed: int,
    benchmark: str,
    embedding_model: str,
    budget: int,
    ds_loader: DatasetManagement,
    k_values: list[int],
    output_dir: str = "results/secondary_experiments/retrain_interval",
    retrain_interval_values: list[int] | None = None,
    cold_start_n: int = DEFAULT_COLD_START_N,
    cost_penalty: float = 0.0,
) -> ResultsTracker:
    """Run InferredSparsePerformanceStrategy with varying trigger_every (retrain interval) values."""
    retrain_interval_values = retrain_interval_values or DEFAULT_RETRAIN_INTERVAL_VALUES

    strategies = {
        f"retrain_every_{delta}": InferredSparsePerformanceStrategy(
            cold_start_n=cold_start_n,
            budget=budget,
            seed=seed,
            training_settings={"trigger_every": delta},
        )
        for delta in retrain_interval_values
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
