import json
from pathlib import Path
from typing import Any
import polars as pl
from sklearn.metrics import auc

from logging_config import logger


class ResultsTracker:
    """Tracks and logs experiment results for streaming active learning comparison."""

    def __init__(
        self,
        experiment_name: str,
        output_dir: str = "results",
    ):
        self.experiment_name = experiment_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.results: dict[
            str, dict[str, list[dict[str, float | str | int]] | list[int]]
        ] = {
            "metadata": {},
            "streaming_results": [],  # type: ignore
            "dict_selected": {},
        }

    def set_metadata(self, **kwargs: Any) -> None:
        """Set experiment metadata (seed, model, benchmark, etc.)."""
        self.results["metadata"].update(kwargs)

    def set_dict_selected(self, dict_selected: dict[str, list[int]]) -> None:
        """Set the dictionary of selected samples for each strategy.

        Args:
            dict_selected: Dictionary mapping strategy names to lists of selected step indices
        """
        self.results["dict_selected"] = dict_selected  # type: ignore

    def log_step(
        self,
        step: int,
        k_value: int,
        strategy: str,
        test_performance: float,
        test_cost: float,
        test_mse: float,
        train_cost: float,
        samples_selected: int,
        dto: float | None = None,
        annotation_cost: float | None = None,
    ) -> None:
        """Log results for a single streaming step.

        Args:
            step: The current step number in the stream
            strategy: Name of the selection strategy
            test_performance: Average test performance
            test_cost: Average test cost
            test_mse: Average test mean squared error
            train_cost: Cumulative training cost
            samples_selected: Number of samples selected so far
            dto: Distance to Oracle metric (optional)
            annotation_cost: Cumulative cost of annotating models (optional)
        """
        entry: dict[str, float | int | str | None] = {
            "step": step,
            "k_value": k_value,
            "strategy": strategy,
            "test_performance": test_performance,
            "test_cost": test_cost,
            "test_mse": test_mse,
            "train_cost": train_cost,
            "samples_selected": samples_selected,
            "dto": dto,
        }
        if annotation_cost is not None:
            entry["annotation_cost"] = annotation_cost
        self.results["streaming_results"].append(entry)  # type: ignore

    def save(self) -> None:
        """Save results to JSON and CSV files."""
        base_path = self.output_dir / self.experiment_name
        base_path.parent.mkdir(parents=True, exist_ok=True)

        json_path = base_path.with_suffix(".json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2)

    def calculate_auc_metrics(self) -> dict[int, dict[str, dict[str, float]]]:
        """Calculate AUC for performance and DTO for each strategy and k_value.

        Returns:
            Dictionary mapping k_value -> strategy name -> AUC metrics
        """
        df = pl.DataFrame(self.results["streaming_results"])
        auc_metrics: dict[int, dict[str, dict[str, float]]] = {}

        for k_value in df["k_value"].unique():
            auc_metrics[k_value] = {}
            for strategy in df["strategy"].unique():
                strategy_df = df.filter(
                    (pl.col("k_value") == k_value) & (pl.col("strategy") == strategy)
                ).sort("step")

                steps = strategy_df["step"].to_numpy()
                performance = strategy_df["test_performance"].to_numpy()
                dto_values = strategy_df["dto"].to_numpy()

                if len(steps) > 1:
                    steps_normalized = (
                        (steps - steps.min()) / (steps.max() - steps.min())
                        if steps.max() > steps.min()
                        else steps
                    )
                    auc_performance = auc(steps_normalized, performance)
                    auc_dto = auc(steps_normalized, dto_values)
                else:
                    auc_performance = 0.0
                    auc_dto = 0.0

                auc_metrics[k_value][strategy] = {
                    "auc_performance": float(auc_performance),
                    "auc_dto": float(auc_dto),
                }

        return auc_metrics

    def get_summary(self) -> dict[int, dict[str, Any]]:
        """Get summary statistics for each strategy grouped by k_value.

        Returns:
            Dictionary mapping k_value -> strategy name -> metrics
        """
        df = pl.DataFrame(self.results["streaming_results"])

        summary_df = df.group_by(["k_value", "strategy"]).agg(
            pl.col("test_performance").last().alias("final_test_performance"),
            pl.col("test_cost").last().alias("final_test_cost"),
            pl.col("test_mse").last().alias("final_test_mse"),
            pl.col("train_cost").last().alias("total_train_cost"),
            pl.col("samples_selected").last().alias("samples_selected"),
            pl.col("test_performance").mean().alias("avg_test_performance"),
            pl.col("test_cost").mean().alias("avg_test_cost"),
            pl.col("test_mse").mean().alias("avg_test_mse"),
            pl.col("dto").last().alias("final_dto"),
            pl.col("dto").mean().alias("avg_dto"),
        )

        summary: dict[int, dict[str, Any]] = {}
        for row in summary_df.to_dicts():
            k_value = row["k_value"]
            strategy = row["strategy"]

            if k_value not in summary:
                summary[k_value] = {}

            summary[k_value][strategy] = {
                k: v for k, v in row.items() if k not in ["k_value", "strategy"]
            }

        auc_metrics = self.calculate_auc_metrics()
        for k_value, strategies in auc_metrics.items():
            if k_value in summary:
                for strategy, metrics in strategies.items():
                    if strategy in summary[k_value]:
                        summary[k_value][strategy].update(metrics)

        return summary

    def print_summary(self) -> None:
        """Print a summary comparison of strategies grouped by k_value."""
        summary = self.get_summary()

        if not summary:
            logger.info("No results to summarize")
            return

        logger.info("\n" + "=" * 80)
        logger.info(f"EXPERIMENT SUMMARY: {self.experiment_name}")
        logger.info("=" * 80)

        for key, value in self.results["metadata"].items():
            logger.info(f"{key}: {value}")

        for k_value, strategies in summary.items():
            logger.info(f"\n{'=' * 80}")
            logger.info(f"K_VALUE = {k_value}")
            logger.info(f"{'=' * 80}")
            logger.info("\nSTRATEGY COMPARISON:")
            logger.info("-" * 80)

            for strategy, stats in strategies.items():
                logger.info(f"\n{strategy.upper()}:")
                logger.info(
                    f"  Final Test Performance: {stats['final_test_performance']:.4f}"
                )
                logger.info(f"  Final Test Cost: {stats['final_test_cost']:.4f}")
                logger.info(f"  Total Training Cost: {stats['total_train_cost']:.4f}")
                logger.info(f"  Samples Selected: {stats['samples_selected']}")
                logger.info(
                    f"  Avg Test Performance: {stats['avg_test_performance']:.4f}"
                )
                logger.info(f"  Avg Test Cost: {stats['avg_test_cost']:.4f}")
                if stats.get("auc_performance") is not None:
                    logger.info(f"  AUC Performance: {stats['auc_performance']:.4f}")
                if stats.get("final_dto") is not None:
                    logger.info(f"  Final DTO: {stats['final_dto']:.4f}")
                    logger.info(f"  Avg DTO: {stats['avg_dto']:.4f}")
                if stats.get("auc_dto") is not None:
                    logger.info(f"  AUC DTO: {stats['auc_dto']:.4f}")

        logger.info("=" * 80 + "\n")
