from typing import Optional
from datasets import Dataset  # type: ignore
import numpy as np
import numpy.typing as npt
import polars as pl
from sklearn.metrics.pairwise import cosine_similarity  # type: ignore


class KNNRouter:
    def __init__(
        self,
    ) -> None:
        """Initialize KNNRouter."""
        self._cached_test_costs: npt.NDArray[np.float64] = np.array([])
        self._cached_test_dataset: Optional[Dataset] = None
        self._cached_test_embeddings: npt.NDArray[np.float64] = np.array([])

    def _prepare_embeddings(self, dataset: pl.DataFrame) -> npt.NDArray[np.float64]:
        """Extract and stack training embeddings into a 2D array."""
        return np.vstack(dataset["embeddings"].to_list())

    def _get_top_k_neighbors(
        self,
        query_embeddings: npt.NDArray[np.float64],
        training_embeddings: npt.NDArray[np.float64],
        k: list[int],
    ) -> dict[int, npt.NDArray[np.intp]]:
        """Find the k most similar training examples for each query."""
        similarities = cosine_similarity(query_embeddings, training_embeddings)

        # Different k are done here to avoid computing again cosine similarity
        top_k_indices = {
            k_value: np.argsort(similarities, axis=1)[:, -k_value:][:, ::-1]
            for k_value in k
        }

        return top_k_indices

    def _predict_performances(
        self, training_dataset: pl.DataFrame, top_k_indices: npt.NDArray[np.intp]
    ) -> npt.NDArray[np.float64]:
        """Predict performance by averaging k-nearest neighbors' performances."""
        training_performances = np.vstack(
            training_dataset["models_performance"].to_list()
        )
        neighbor_performances = training_performances[top_k_indices]
        predicted_performances = np.mean(neighbor_performances, axis=1)
        return predicted_performances

    def _predict_performances_from_array(
        self,
        training_performances: npt.NDArray[np.float64],
        top_k_indices: npt.NDArray[np.intp],
    ) -> npt.NDArray[np.float64]:
        """Predict performance using a pre-computed numpy performance matrix."""
        neighbor_performances = training_performances[top_k_indices]
        return np.mean(neighbor_performances, axis=1)

    def _select_best_candidates(
        self,
        predicted_performances: npt.NDArray[np.float64],
        costs: npt.NDArray[np.float64],
        cost_penalty: float,
    ) -> npt.NDArray[np.intp]:
        """Select best candidates by maximizing utility (performance - cost penalty)."""
        utility_scores = predicted_performances - (cost_penalty * costs)
        best_candidate_indices = np.argmax(utility_scores, axis=1)
        return best_candidate_indices

    def inference_test_dataset(
        self,
        test_ds: Dataset,
        train_ds: pl.DataFrame,
        cost_penalty: float,
        k: list[int],
        sim_matrix: Optional[npt.NDArray[np.float64]] = None,
        train_performances: Optional[npt.NDArray[np.float64]] = None,
    ) -> dict[int, dict[str, npt.NDArray[np.float64]]]:
        """Route user queries to optimal candidates based on retrieved performance from similar queries."""
        if self._cached_test_dataset != test_ds:
            self._cached_test_embeddings: npt.NDArray[np.float64] = np.array(
                test_ds["embeddings"]  # type: ignore
            ).squeeze(1)
            self._cached_test_costs: npt.NDArray[np.float64] = np.array(
                test_ds["cost"][0]  # type: ignore
            )
            self._cached_test_dataset = test_ds

        if sim_matrix is not None:
            top_k_indices = {}
            for k_value in k:
                n_train = sim_matrix.shape[1]
                actual_k = min(k_value, n_train)
                part_idx = np.argpartition(sim_matrix, -actual_k, axis=1)[:, -actual_k:]
                rows = np.arange(sim_matrix.shape[0])[:, None]
                sorted_local = np.argsort(sim_matrix[rows, part_idx], axis=1)[:, ::-1]
                top_k_indices[k_value] = part_idx[rows, sorted_local]
        else:
            training_embeddings = self._prepare_embeddings(train_ds)
            top_k_indices = self._get_top_k_neighbors(
                self._cached_test_embeddings, training_embeddings, k
            )

        if train_performances is None:
            train_performances = np.vstack(train_ds["models_performance"].to_list())

        k_results: dict[int, dict[str, npt.NDArray[np.float64]]] = {}
        for k_value in top_k_indices.keys():
            predicted_performances = self._predict_performances_from_array(
                train_performances, top_k_indices[k_value]
            )

            best_candidate_indices = self._select_best_candidates(
                predicted_performances, self._cached_test_costs, cost_penalty
            )

            true_performances = np.array(self._cached_test_dataset["models_performance"])  # type: ignore
            row_indices = np.arange(len(best_candidate_indices))

            selected_true_performances: npt.NDArray[np.float64] = true_performances[  # type: ignore
                row_indices, best_candidate_indices
            ]
            selected_true_costs: npt.NDArray[np.float64] = self._cached_test_costs[
                best_candidate_indices
            ]

            prediction_errors: npt.NDArray[np.float64] = (  # squared_error
                true_performances - predicted_performances
            ) ** 2

            k_results[k_value] = {
                "perf": selected_true_performances,
                "cost": selected_true_costs,
                "mse": prediction_errors,
            }

        return k_results
