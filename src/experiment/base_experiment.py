from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Literal

import numpy as np

from datasets_management.non_stationary_dataset import DatasetManagement
from logging_config import logger
from experiment.evaluation import MainEvaluator
from experiment.results_tracker import ResultsTracker
from experiment.routers.knn_router import KNNRouter
from experiment.utils.metrics import extract_oracle_performance_numpy

from sampling_strategies import (
    HeuristicsStrategy,
    RandomStrategy,
    PassiveStrategy,
    OracleCoverageStrategy,
    OracleSparsePerformanceStrategy,
    InferredSparsePerformanceStrategy,
)


class BaseExperiment(ABC):
    """Abstract base class for streaming experiments.

    Provides common infrastructure for running experiments with different strategies
    and handles consistent logging, evaluation, and result tracking.
    """

    def __init__(
        self,
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
        output_dir: str,
        k: list[int],
        max_step: int | Literal["all"] = "all",
        cost_penalty: float = 0.0,
    ):
        """Initialize base experiment.

        Args:
            seed: Random seed for reproducibility
            embedding_model: Name of the embedding model
            benchmark: Name of the benchmark dataset
            ds_loader: Dataset management object
            strategies: Dictionary of strategies to compare
            output_dir: Directory to save results
            max_step: Maximum number of steps or "all"
            cost_penalty: Cost penalty factor
        """
        self.seed = seed
        self.embedding_model = embedding_model
        self.benchmark = benchmark
        self.ds_loader = ds_loader
        self.strategies = strategies
        self.output_dir = output_dir
        self.max_step = max_step
        self.cost_penalty = cost_penalty
        self.k = k

        if cost_penalty > 0.0:
            self.output_dir = f"{output_dir}/{str(cost_penalty)}"

        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        self.evaluators: Dict[str, MainEvaluator] = {}
        self.o_cost: float = 0.0
        self.o_perf: float = 0.0

    @abstractmethod
    def preprocess_dataset(self) -> None:
        """Preprocess the dataset before experiment (e.g., add drift, simulations)."""
        pass

    @abstractmethod
    def get_experiment_name(self) -> str:
        """Get the experiment name for tracking and saving results."""
        pass

    @abstractmethod
    def save_additional_data(self) -> None:
        """Save any additional experiment-specific data.

        Args:
            dict_selected: Dictionary mapping strategy names to selected step indices
        """
        pass

    def setup_evaluators(self) -> None:
        """Setup evaluators for all strategies."""
        self.evaluators = {
            name: MainEvaluator(
                router=KNNRouter(),
                test_ds=self.test_ds,
                cost_penalty=self.cost_penalty,
            )
            for name in self.strategies.keys()
        }

    def setup_tracker(self, experiment_name: str) -> None:
        """Setup results tracker.

        Args:
            experiment_name: Name of the experiment
        """
        self.tracker = ResultsTracker(experiment_name, output_dir=self.output_dir)

        al_params = [
            {s: self.strategies[s].get_params()}
            for s in self.strategies
            if "actlearn" in s
        ]

        self.tracker.set_metadata(
            seed=self.seed,
            embedding_model=self.embedding_model,
            benchmark=self.benchmark,
            train_size=len(self.ds_loader.get_training_dataset()),
            test_size=len(self.ds_loader.get_test_dataset()),
            timestamp=self.timestamp,
            al_params=al_params,
        )

    def compute_oracle_metrics(
        self,
    ) -> None:
        """Compute oracle performance and cost metrics.

        Args:
            perf_feature_name: Name of the performance feature column
            cost_feature_name: Name of the cost feature column
        """
        o_cost, o_perf = extract_oracle_performance_numpy(
            np.array(self.test_ds["models_performance"]),  # type: ignore
            np.array(self.test_ds["cost"]),  # type: ignore
        )
        self.o_cost = float(np.mean(o_cost))
        self.o_perf = float(np.mean(o_perf))  # type: ignore

    @abstractmethod
    def run_streaming_loop(self) -> Dict[str, List[int]]:
        """Run the main streaming experiment loop.

        Returns:
            Dictionary mapping strategy names to selected step indices
        """
        pass

    def get_progress_description(self) -> str:
        """Get the progress bar description.

        Returns:
            Description string for the progress bar
        """
        return "Processing stream"

    def run(self) -> ResultsTracker:
        """Run the complete experiment pipeline.

        Returns:
            ResultsTracker with experiment results
        """
        self.preprocess_dataset()
        self.test_ds = self.ds_loader.get_test_dataset()
        self.setup_evaluators()

        experiment_name = self.get_experiment_name()
        self.setup_tracker(experiment_name)

        logger.info(f"\nRunning streaming experiment: {experiment_name}")
        logger.info(f"Strategies: {list(self.strategies.keys())}")
        logger.info(
            f"Benchmark: {self.benchmark}, Embedding: {self.embedding_model}, Seed: {self.seed}\n"
        )

        self.compute_oracle_metrics()

        dict_selected = self.run_streaming_loop()

        if self.tracker:
            self.tracker.set_dict_selected(dict_selected)
            self.tracker.save()
            self.tracker.print_summary()

        self.save_additional_data()

        return self.tracker
