from typing import Any, Optional

import numpy as np
import polars as pl
from datasets import Dataset  # type: ignore
from numpy.typing import NDArray
from sklearn.metrics.pairwise import cosine_similarity  # type: ignore

from experiment.routers.knn_router import KNNRouter


class MainEvaluator:
    """
    Evaluates router performance on test datasets with cost tracking.

    This class manages the evaluation of a KNN router by tracking performance
    and cost metrics across multiple datasets over time.
    """

    def __init__(
        self,
        router: KNNRouter,
        test_ds: Dataset,
        cost_penalty: float = 0.0,
    ) -> None:
        """
        Initialize the evaluator.

        Args:
            router: KNN router to evaluate
            test_ds: Test dataset containing samples with 'dataset' column
            cost_penalty: Penalty factor for cost in routing decisions
        """
        self.column_names: list[str] = test_ds.column_names
        self.train_df: Optional[pl.DataFrame] = None
        self.test_ds: Dataset = test_ds
        self.test_ds_array: NDArray[Any] = np.array(self.test_ds["dataset"])  # type: ignore

        self.unique_dataset_name: list[str] = np.unique(self.test_ds_array).tolist()
        self.perf_per_dataset: dict[int, dict[str, dict[str, list[float]]]] = {}
        self.dataset_masks: dict[str, NDArray[np.bool_]] = {
            name: self.test_ds_array == name for name in self.unique_dataset_name
        }
        self.router: KNNRouter = router
        self.cost_penalty: float = cost_penalty

        self.cost_train: int = 0

        self._cached_test_emb: NDArray[np.float64] = np.array(test_ds["embeddings"]).squeeze(1)  # type: ignore
        n_test = self._cached_test_emb.shape[0]
        self._train_emb_list: list[NDArray[np.float64]] = []
        self._train_perf_list: list[NDArray[np.float64]] = []
        self._cached_train_emb: Optional[NDArray[np.float64]] = None
        self._cached_train_perf: Optional[NDArray[np.float64]] = None
        self._sim_matrix_buf: NDArray[np.float64] = np.empty(
            (n_test, 0), dtype=np.float64
        )
        self._sim_col_idx: int = 0
        self._cached_sim_matrix: Optional[NDArray[np.float64]] = None

    def redundant_step_no_annotation(self) -> None:
        """
        Repeat last performance and cost values for all datasets.

        Used when no new annotation is added in a step, to maintain
        consistent tracking across all evaluation steps. If no previous
        values exist, initializes with 0.0.
        """
        for k_value in self.perf_per_dataset:
            for dataset_name in self.unique_dataset_name:
                dataset_metrics = self.perf_per_dataset[k_value][dataset_name]
                perf_list = dataset_metrics["perf"]
                cost_list = dataset_metrics["cost"]

                if perf_list:
                    perf_list.append(perf_list[-1])
                    cost_list.append(cost_list[-1])
                else:
                    perf_list.append(0.0)
                    cost_list.append(0.0)

    def evaluate_on_test(
        self, train_item: Optional[dict[str, Any]], k: list[int]
    ) -> dict[int, dict[str, float]]:
        """
        Evaluate router performance on the test dataset after adding a training item.

        Args:
            train_item: New training sample to add, or None to evaluate without adding
            k: List of k values for KNN evaluation

        Returns:
            Dictionary mapping k values to performance metrics:
                - avg_perf: Average performance across test set
                - avg_cost: Average cost across test set
                - avg_mse: Average mean squared error

        """

        if train_item:
            if self.train_df is None:
                self.train_df = pl.DataFrame({k: [v] for k, v in train_item.items()})
            else:
                new_row = pl.DataFrame({k: [v] for k, v in train_item.items()})
                self.train_df = pl.concat([self.train_df, new_row], rechunk=False)

            new_emb = np.array(train_item["embeddings"]).reshape(1, -1)
            new_perf = np.array(train_item["models_performance"]).reshape(1, -1)
            self._train_emb_list.append(new_emb.squeeze(0))
            self._train_perf_list.append(new_perf.squeeze(0))
            self._cached_train_emb = np.stack(self._train_emb_list)
            self._cached_train_perf = np.stack(self._train_perf_list)

            new_sim_col = cosine_similarity(self._cached_test_emb, new_emb)
            if self._sim_col_idx >= self._sim_matrix_buf.shape[1]:
                new_capacity = max(64, self._sim_matrix_buf.shape[1] * 2)
                new_buf = np.empty(
                    (self._cached_test_emb.shape[0], new_capacity), dtype=np.float64
                )
                if self._sim_col_idx > 0:
                    new_buf[:, : self._sim_col_idx] = self._sim_matrix_buf[
                        :, : self._sim_col_idx
                    ]
                self._sim_matrix_buf = new_buf
            self._sim_matrix_buf[:, self._sim_col_idx] = new_sim_col.squeeze()
            self._sim_col_idx += 1
            self._cached_sim_matrix = self._sim_matrix_buf[:, : self._sim_col_idx]

            self.cost_train += 1

        results_avg: dict[int, dict[str, float]] = {}
        if self._cached_sim_matrix is not None:
            results = self.router.inference_test_dataset(
                train_ds=self.train_df,  # type: ignore
                test_ds=self.test_ds,
                cost_penalty=self.cost_penalty,
                k=k,
                sim_matrix=self._cached_sim_matrix,
                train_performances=self._cached_train_perf,
            )
            for k_value in results.keys():
                if k_value not in self.perf_per_dataset.keys():
                    self.perf_per_dataset[k_value] = {
                        i: {"perf": [], "cost": []} for i in self.unique_dataset_name
                    }
                test_performances_arr = np.array(results[k_value]["perf"])
                test_costs_arr = np.array(results[k_value]["cost"])
                mse_values_arr = np.array(results[k_value]["mse"])

                for dataset_name in self.unique_dataset_name:
                    mask = self.dataset_masks[dataset_name]
                    self.perf_per_dataset[k_value][dataset_name]["perf"].append(
                        float(np.mean(test_performances_arr[mask]))
                    )
                    self.perf_per_dataset[k_value][dataset_name]["cost"].append(
                        float(np.mean(test_costs_arr[mask]))
                    )

                results_avg[k_value] = {
                    "avg_perf": float(np.mean(test_performances_arr)),
                    "avg_cost": float(np.mean(test_costs_arr)),
                    "avg_mse": float(np.mean(mse_values_arr)),
                }

            return results_avg
        else:
            return {
                k_value: {
                    "avg_perf": 0.0,
                    "avg_cost": 0.0,
                    "avg_mse": 0.0,
                }
                for k_value in k
            }
