import json
import os
from typing import Dict, List, Literal

from tqdm import tqdm
from sklearn.metrics import auc
import numpy as np

from datasets_management.stationary_dataset import DatasetManagement
from experiment.base_experiment import BaseExperiment
from experiment.results_tracker import ResultsTracker
from experiment.utils.metrics import calculate_weighted_dto
from logging_config import logger
from sampling_strategies import (
    HeuristicsStrategy,
    RandomStrategy,
    PassiveStrategy,
    OracleCoverageStrategy,
    OracleSparsePerformanceStrategy,
    InferredSparsePerformanceStrategy,
)


class MainExperiment(BaseExperiment):
    """Main experiment using real embeddings and model performance."""

    def __init__(
        self,
        budget: int,
        seed: int,
        embedding_model: str,
        benchmark: str,
        ds_loader: DatasetManagement,
        strategies: dict[
            str,
            InferredSparsePerformanceStrategy
            | HeuristicsStrategy
            | PassiveStrategy
            | RandomStrategy
            | OracleCoverageStrategy
            | OracleSparsePerformanceStrategy,
        ],
        k: list[int],
        output_dir: str = "results",
        max_step: int | Literal["all"] = "all",
        cost_penalty: float = 0.0,
        stationary: bool = True,
    ):
        super().__init__(
            seed=seed,
            embedding_model=embedding_model,
            benchmark=benchmark,
            ds_loader=ds_loader,
            strategies=strategies,  # type: ignore
            output_dir=output_dir,
            max_step=max_step,
            cost_penalty=cost_penalty,
            k=k,
        )
        self.budget = budget
        self.stationary = stationary
        self.ds_first_item_per_ds = {}
        self.k = k

    def preprocess_dataset(self) -> None:
        """No preprocessing needed for main experiment."""
        pass

    def get_experiment_name(self) -> str:
        """Get the experiment name.

        Returns:
            Experiment name in format: embedding_model/benchmark/seed/run_timestamp
        """
        str_add = "stationary"
        return f"{str_add}/{self.embedding_model}/{self.benchmark}/{self.seed}/run_{self.timestamp}"

    def run_streaming_loop(self) -> Dict[str, List[int]]:
        """Run the main streaming experiment loop with dataset tracking.

        Returns:
            Dictionary mapping strategy names to selected step indices
        """
        samples_selected = {name: 0 for name in self.strategies.keys()}
        previous_metrics = {
            name: {
                k_value: {"avg_perf": 0.0, "avg_cost": 0.0, "avg_mse": 0.0, "dto": 0.0}
                for k_value in self.k
            }
            for name in self.strategies.keys()
        }

        length_stream: int = len(self.ds_loader.get_training_dataset())
        dict_selected: dict[str, list[int]] = {
            strategy: [] for strategy in self.strategies
        }
        for step, item in enumerate(  # type: ignore
            tqdm(
                self.ds_loader.get_training_generator(),  # type: ignore
                desc=self.get_progress_description(),
                total=length_stream,
            )
        ):

            if step <= length_stream:
                for strategy_name, strategy in self.strategies.items():
                    evaluator = self.evaluators[strategy_name]
                    current_corpus = evaluator.train_df

                    should_select = strategy.should_select(  # type: ignore
                        item=item,  # type: ignore
                        current_train_ds=current_corpus,  # type: ignore
                    )
                    # If should select and (training corpus is not empty OR budget is not exhausted)
                    if should_select and (
                        current_corpus is None or len(current_corpus) < self.budget
                    ):
                        dict_selected[strategy_name].append(step)
                        results_k = evaluator.evaluate_on_test(train_item=item, k=self.k)  # type: ignore

                        for k_value in results_k.keys():
                            dto = calculate_weighted_dto(
                                oracle_cost=self.o_cost,
                                oracle_perf=self.o_perf,
                                test_cost=results_k[k_value]["avg_cost"],
                                test_perf=results_k[k_value]["avg_perf"],
                            )
                            results_k[k_value]["dto"] = float(dto)

                        samples_selected[strategy_name] += 1
                        previous_metrics[strategy_name] = results_k
                    # if should not select, no evaluation on test dataset: will be same results as previous iteration
                    else:
                        evaluator.redundant_step_no_annotation()
                        results_k = previous_metrics[strategy_name]

                    for k_value in results_k.keys():
                        self.tracker.log_step(
                            step=step,
                            k_value=k_value,
                            strategy=strategy_name,
                            test_performance=results_k[k_value]["avg_perf"],
                            test_cost=results_k[k_value]["avg_cost"],
                            test_mse=results_k[k_value]["avg_mse"],
                            train_cost=evaluator.cost_train,
                            samples_selected=samples_selected[strategy_name],
                            dto=results_k[k_value]["dto"],
                        )

        return dict_selected

    def calculate_per_dataset_auc(
        self,
    ) -> dict[str, dict[int, dict[str, dict[str, float]]]]:
        """Calculate AUC for performance and DTO per dataset for each strategy and k value.

        Returns:
            Nested dictionary: {strategy: {k_value: {dataset: {"auc_perf": float, "auc_dto": float}}}}
        """
        auc_per_dataset: dict[str, dict[int, dict[str, dict[str, float]]]] = {}

        for strategy_name in self.strategies:
            evaluator = self.evaluators[strategy_name]
            auc_per_dataset[strategy_name] = {}

            for k_value, datasets_metrics in evaluator.perf_per_dataset.items():
                auc_per_dataset[strategy_name][k_value] = {}

                for dataset_name, metrics in datasets_metrics.items():
                    perf_values = np.array(metrics["perf"])
                    cost_values = np.array(metrics["cost"])

                    if len(perf_values) > 1:
                        steps_normalized = np.linspace(0, 1, len(perf_values))
                        auc_perf = auc(steps_normalized, perf_values)

                        dto_values = np.array(
                            [
                                calculate_weighted_dto(
                                    oracle_cost=self.o_cost,
                                    oracle_perf=self.o_perf,
                                    test_cost=cost,
                                    test_perf=perf,
                                )
                                for perf, cost in zip(perf_values, cost_values)
                            ]
                        )

                        auc_dto = auc(steps_normalized, dto_values)

                        auc_per_dataset[strategy_name][k_value][dataset_name] = {
                            "auc_performance": float(auc_perf),
                            "auc_dto": float(auc_dto),
                        }
                    else:
                        auc_per_dataset[strategy_name][k_value][dataset_name] = {
                            "auc_performance": 0.0,
                            "auc_dto": 0.0,
                        }

        return auc_per_dataset

    def save_additional_data(self) -> None:
        """Save main experiment specific data.

        Args:
            dict_selected: Dictionary mapping strategy names to selected step indices
        """
        train_ds = self.ds_loader.get_training_dataset()
        experiment_name = self.get_experiment_name()

        seed_dir = f"{self.output_dir}/{experiment_name}"
        os.makedirs(seed_dir, exist_ok=True)
        seed_datasets_file = f"{seed_dir}/seed_datasets.json"
        if os.path.exists(seed_datasets_file):
            with open(seed_datasets_file, "r", encoding="utf-8") as f:
                dict_datasets = json.load(f)
        else:
            dict_datasets = {}

        if self.benchmark not in dict_datasets:
            dict_datasets[self.benchmark] = {}

        dict_datasets[self.benchmark][self.seed] = list(train_ds["dataset"])  # type: ignore

        with open(seed_datasets_file, "w", encoding="utf-8") as f:
            json.dump(dict_datasets, f, indent=2)

        perf_per_dataset_file = os.path.join(
            seed_dir, f"perf_per_dataset_{self.timestamp}.json"
        )
        perf_per_dataset_data = {
            strategy_name: self.evaluators[strategy_name].perf_per_dataset
            for strategy_name in self.strategies.keys()
        }

        with open(perf_per_dataset_file, "w", encoding="utf-8") as f:
            json.dump(perf_per_dataset_data, f, indent=2)

        logger.info(f"Per-dataset performance saved to: {perf_per_dataset_file}")

        auc_global_file = os.path.join(seed_dir, f"auc_global_{self.timestamp}.json")
        auc_global_metrics = self.tracker.calculate_auc_metrics()

        with open(auc_global_file, "w", encoding="utf-8") as f:
            json.dump(auc_global_metrics, f, indent=2)

        logger.info(f"Global AUC metrics saved to: {auc_global_file}")

        auc_per_dataset_file: str = os.path.join(
            seed_dir, f"auc_per_dataset_{self.timestamp}.json"
        )
        auc_per_dataset_data = self.calculate_per_dataset_auc()

        with open(auc_per_dataset_file, "w", encoding="utf-8") as f:
            json.dump(auc_per_dataset_data, f, indent=2)

        logger.info(f"Per-dataset AUC metrics saved to: {auc_per_dataset_file}")


def run_main_experiment(
    budget: int,
    seed: int,
    embedding_model: str,
    benchmark: str,
    ds_loader: DatasetManagement,
    strategies: dict[
        str,
        InferredSparsePerformanceStrategy
        | HeuristicsStrategy
        | PassiveStrategy
        | RandomStrategy
        | OracleCoverageStrategy
        | OracleSparsePerformanceStrategy,
    ],
    k: list[int],
    output_dir: str = "results",
    max_step: int | Literal["all"] = "all",
    cost_penalty: float = 0.0,
    stationary: bool = True,
) -> ResultsTracker:
    """Run streaming comparison experiment: Active Learning vs Random (Traditional).

    Args:
        seed: Random seed for reproducibility
        embedding_model: Name of the embedding model to use
        benchmark: Name of the benchmark dataset
        ds_loader: Dataset management object
        strategies: List of strategies to compare
        output_dir: Directory to save results
        max_step: Maximum number of steps or "all"
        cost_penalty: Cost penalty factor

    Returns:
        ResultsTracker with experiment results
    """
    experiment = MainExperiment(
        budget=budget,
        seed=seed,
        embedding_model=embedding_model,
        benchmark=benchmark,
        ds_loader=ds_loader,
        strategies=strategies,
        output_dir=output_dir,
        max_step=max_step,
        cost_penalty=cost_penalty,
        stationary=stationary,
        k=k,
    )
    return experiment.run()
