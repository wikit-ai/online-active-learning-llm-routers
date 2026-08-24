import json
import os
from typing import Dict, List, Literal

from tqdm import tqdm
from sklearn.metrics import auc
import numpy as np

from datasets_management.non_stationary_dataset import DomainShiftDatasetManagement
from experiment.base_experiment import BaseExperiment

from experiment.results_tracker import ResultsTracker
from logging_config import logger
from sampling_strategies import (
    HeuristicsStrategy,
    RandomStrategy,
    PassiveStrategy,
    OracleCoverageStrategy,
    OracleSparsePerformanceStrategy,
    InferredSparsePerformanceStrategy,
)
from experiment.utils.metrics import calculate_weighted_dto


class DomainShiftExperiment(BaseExperiment):
    """Experiment that streams through all topic domains sequentially.

    Each domain of TRAIN_SIZE_PER_DOMAIN steps is tracked as its own period.
    Period boundaries are derived automatically from DomainShiftDatasetManagement.
    """

    def __init__(
        self,
        budget: int,
        seed: int,
        embedding_model: str,
        benchmark: str,
        ds_loader: DomainShiftDatasetManagement,
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
    ):
        super().__init__(
            seed=seed,
            embedding_model=embedding_model,
            benchmark=benchmark,
            ds_loader=ds_loader,
            strategies=strategies,  # type: ignore
            k=k,
            output_dir=output_dir,
            max_step=max_step,
            cost_penalty=cost_penalty,
        )
        self.budget = budget
        self.ds_loader: DomainShiftDatasetManagement = ds_loader  # type: ignore

        n = DomainShiftDatasetManagement.TRAIN_SIZE_PER_DOMAIN
        domain_names: list[str] = ds_loader.domain_names
        self.domain_boundaries: list[tuple[int, int, str]] = [
            (i * n, (i + 1) * n - 1, name) for i, name in enumerate(domain_names)
        ]

        self.domain_period_metrics: dict[
            str, dict[int, dict[str, dict[str, list[float]]]]
        ] = {
            strategy_name: {
                k_value: {
                    name: {"steps": [], "performance": [], "cost": [], "dto": []}
                    for _, _, name in self.domain_boundaries
                }
                for k_value in k
            }
            for strategy_name in strategies.keys()
        }

    def preprocess_dataset(self) -> None:
        pass

    def get_experiment_name(self) -> str:
        abbrev = "_".join(d[:4] for d in self.ds_loader.domain_names)
        return (
            f"domain_shift/{self.embedding_model}/{self.benchmark}/{self.seed}/"
            f"{abbrev}/run_{self.timestamp}"
        )

    def _get_domain_for_step(self, step: int) -> str | None:
        for start, end, name in self.domain_boundaries:
            if start <= step <= end:
                return name
        return None

    def run_streaming_loop(self) -> Dict[str, List[int]]:
        """Run streaming loop with per-domain period tracking."""
        samples_selected = {name: 0 for name in self.strategies.keys()}
        previous_metrics = {
            name: {
                k_value: {"avg_perf": 0.0, "avg_cost": 0.0, "avg_mse": 0.0, "dto": 0.0}
                for k_value in self.k
            }
            for name in self.strategies.keys()
        }

        length_stream: int = len(self.ds_loader.get_training_dataset())
        dict_selected: dict[str, list[int]] = {s: [] for s in self.strategies}

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

                    should_select = strategy.should_select(
                        item=item,  # type: ignore
                        current_train_ds=current_corpus,  # type: ignore
                    )

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

                    domain = self._get_domain_for_step(step)
                    if domain is not None:
                        for k_value in results_k.keys():
                            self.domain_period_metrics[strategy_name][k_value][domain][
                                "steps"
                            ].append(step)
                            self.domain_period_metrics[strategy_name][k_value][domain][
                                "performance"
                            ].append(results_k[k_value]["avg_perf"])
                            self.domain_period_metrics[strategy_name][k_value][domain][
                                "cost"
                            ].append(results_k[k_value]["avg_cost"])
                            self.domain_period_metrics[strategy_name][k_value][domain][
                                "dto"
                            ].append(results_k[k_value]["dto"])

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

    def calculate_period_metrics(
        self,
    ) -> dict[str, dict[int, dict[str, dict[str, float]]]]:
        """Calculate AUC and final values per domain period."""
        period_results: dict[str, dict[int, dict[str, dict[str, float]]]] = {}

        for strategy_name, k_periods in self.domain_period_metrics.items():
            period_results[strategy_name] = {}

            for k_value, domains in k_periods.items():
                period_results[strategy_name][k_value] = {}

                for domain_name, metrics in domains.items():
                    if len(metrics["steps"]) > 1:
                        steps = np.array(metrics["steps"])
                        performance = np.array(metrics["performance"])
                        cost = np.array(metrics["cost"])
                        dto_values = np.array(metrics["dto"])

                        steps_normalized = (steps - steps[0]) / (steps[-1] - steps[0])
                        auc_perf = auc(steps_normalized, performance)
                        auc_dto_val = auc(steps_normalized, dto_values)

                        period_results[strategy_name][k_value][domain_name] = {  # type: ignore
                            "auc_performance": float(auc_perf),
                            "auc_dto": float(auc_dto_val),
                            "final_performance": float(performance[-1]),
                            "final_cost": float(cost[-1]),
                            "final_dto": float(dto_values[-1]),
                            "num_steps": len(steps),
                            "step_range": f"{int(steps[0])}-{int(steps[-1])}",
                        }
                    else:
                        period_results[strategy_name][k_value][domain_name] = {  # type: ignore
                            "auc_performance": 0.0,
                            "auc_dto": 0.0,
                            "final_performance": 0.0,
                            "final_cost": 0.0,
                            "final_dto": 0.0,
                            "num_steps": len(metrics["steps"]),
                            "step_range": "N/A",
                        }

        return period_results

    def save_additional_data(self) -> None:
        """Save domain-shift specific data (global AUC, per-dataset AUC, per-domain metrics)."""
        experiment_name = self.get_experiment_name()
        seed_dir = f"{self.output_dir}/{experiment_name}"
        os.makedirs(seed_dir, exist_ok=True)

        auc_global_file = os.path.join(seed_dir, f"auc_global_{self.timestamp}.json")
        with open(auc_global_file, "w", encoding="utf-8") as f:
            json.dump(self.tracker.calculate_auc_metrics(), f, indent=2)
        logger.info(f"Global AUC metrics saved to: {auc_global_file}")

        auc_per_dataset_file = os.path.join(
            seed_dir, f"auc_per_dataset_{self.timestamp}.json"
        )
        with open(auc_per_dataset_file, "w", encoding="utf-8") as f:
            json.dump(self.calculate_per_dataset_auc(), f, indent=2)
        logger.info(f"Per-dataset AUC metrics saved to: {auc_per_dataset_file}")

        domain_metrics_file = os.path.join(
            seed_dir, f"domain_period_metrics_{self.timestamp}.json"
        )
        with open(domain_metrics_file, "w", encoding="utf-8") as f:
            json.dump(self.calculate_period_metrics(), f, indent=2)
        logger.info(f"Per-domain period metrics saved to: {domain_metrics_file}")


def run_domain_shift_experiment(
    budget: int,
    seed: int,
    embedding_model: str,
    benchmark: str,
    ds_loader: DomainShiftDatasetManagement,
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
) -> ResultsTracker:
    """Run a sequential domain-shift streaming experiment."""
    experiment = DomainShiftExperiment(
        budget=budget,
        seed=seed,
        embedding_model=embedding_model,
        benchmark=benchmark,
        ds_loader=ds_loader,
        strategies=strategies,
        k=k,
        output_dir=output_dir,
        max_step=max_step,
        cost_penalty=cost_penalty,
    )
    return experiment.run()
