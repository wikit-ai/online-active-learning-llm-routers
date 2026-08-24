from typing import Optional, Literal

from numpy.typing import ArrayLike
import numpy as np
import numpy.typing as npt

import polars as pl

from sklearn.metrics.pairwise import cosine_similarity

from statsmodels.stats.weightstats import DescrStatsW  # type: ignore

from active_learning.heuristics.base import BaseHeuristic


class VarianceUncertaintyEvaluation(BaseHeuristic):
    """Evaluates uncertainty of samples using weighted variance for active learning.

    Supports multiple adaptive threshold strategies:
    - 'quantile': Uses a quantile of recent uncertainties (adaptive)
    - 'fixed': Uses a fixed threshold
    """

    def __init__(
        self,
        seed: int,
        strategy: Literal["quantile", "fixed"] = "quantile",
        window_size: int = 20,
        window_quantile: Optional[float] = 0.85,
        fixed_threshold: float = 0.15,
        uncertainty_top_k: int | None = None,
    ):
        """Initialize uncertainty evaluation.

        Args:
            seed: Random seed for reproducibility
            strategy: Threshold strategy - 'quantile' for adaptive or 'fixed' for constant threshold
            window_size: Number of recent uncertainties to keep for quantile calculation
            window_quantile: Quantile value (0-1) to use as threshold when strategy is 'quantile'
            fixed_threshold: Threshold value to use when strategy is 'fixed'
            uncertainty_top_k: If set, only consider top-k models for uncertainty calculation
        """
        super().__init__(window_size=window_size)
        self.seed = seed
        self.strategy = strategy
        self.quantile = window_quantile
        self.fixed_threshold = fixed_threshold
        self.uncertainty_top_k = uncertainty_top_k

    def _compute_threshold(self) -> float:
        """Compute threshold based on configured strategy.

        Raises:
            NotImplementedError: If strategy is not 'quantile'
        """
        if self.strategy == "quantile":
            return float(np.quantile(self._window, self.quantile))  # type: ignore
        else:
            raise NotImplementedError()

    def calculate_weighted_stats(
        self,
        embedding_query: npt.ArrayLike,
        training_dataset: pl.DataFrame,
        training_embeddings: np.ndarray | None = None,
        training_performances: np.ndarray | None = None,
    ):
        """Calculate weighted variance and mean of model performances.

        Weights are based on cosine similarity between query embedding and training embeddings.
        Higher similarity means higher weight in variance/mean calculations.

        Args:
            embedding_query: Query embedding vector
            training_dataset: DataFrame with 'embeddings' and 'models_performance' columns

        Returns:
            Tuple of (weighted_variance, weighted_mean) arrays for each model
        """
        if training_embeddings is None:
            training_embeddings = np.vstack(training_dataset["embeddings"].to_list())
        if training_performances is None:
            training_performances = np.vstack(
                training_dataset["models_performance"].to_list()
            )

        embedding_query = np.asarray(embedding_query).reshape(1, -1)

        cosine_list = cosine_similarity(embedding_query, training_embeddings).flatten()

        weights = cosine_list

        descriptive_stats = DescrStatsW(training_performances, weights=weights)
        weighted_var = descriptive_stats.var
        weighted_mean = descriptive_stats.mean
        if self.uncertainty_top_k is not None:
            top5_idx = np.argsort(weighted_mean)[-self.uncertainty_top_k :][::-1]
            weighted_mean = weighted_mean[top5_idx]
            weighted_var = weighted_var[top5_idx]
        return weighted_var, weighted_mean

    def get_score(
        self,
        embedding_query: npt.ArrayLike,
        training_dataset: pl.DataFrame,
        training_embeddings: np.ndarray | None = None,
        training_performances: np.ndarray | None = None,
    ) -> float:
        """Compute weighted variance uncertainty score for a query.

        Args:
            embedding_query: Query embedding vector
            training_dataset: DataFrame with 'embeddings' and 'models_performance' columns
            training_embeddings: Pre-computed training embeddings matrix
            training_performances: Pre-computed training performances matrix

        Returns:
            Mean weighted variance across all models (or top-k if uncertainty_top_k is set)
        """
        weighted_var_list, _ = self.calculate_weighted_stats(
            embedding_query=embedding_query,
            training_dataset=training_dataset,
            training_embeddings=training_embeddings,
            training_performances=training_performances,
        )
        score = float(np.array(weighted_var_list).mean())
        return score

    def get_params(self) -> dict[str, str | float | None]:
        """Get configuration parameters as a dictionary."""
        d_params: dict[str, str | float | None] = {
            "unc_strategy": "weighted_variance",
            "threshold_strategy": self.strategy,
        }
        if self.strategy == "fixed":
            d_params["fixed_threshold"] = self.fixed_threshold
        elif self.strategy == "quantile":
            d_params["window_size"] = self.window_size
            d_params["window_quantile"] = self.quantile

        return d_params

    def __call__(
        self,
        embedding_query: ArrayLike,
        training_dataset: pl.DataFrame | None,
    ) -> bool:
        """Evaluate if query has high enough uncertainty to be added to training set.

        Args:
            embedding_query: Query embedding to evaluate
            training_dataset: Current training dataset with 'embeddings' and 'models_performance' columns

        Returns:
            True if sample should be selected (high uncertainty = more informative)
        """
        if training_dataset is not None and len(training_dataset) > 5:

            mean_weighted_var = self.get_score(
                embedding_query=embedding_query,
                training_dataset=training_dataset,
            )
            if self.strategy != "fixed":
                self._update_window(float(mean_weighted_var))

            threshold = self._compute_threshold()
            return mean_weighted_var >= threshold
        else:
            return True


class RankingUncertaintyEvaluation(VarianceUncertaintyEvaluation):
    """Evaluates uncertainty using ranking stability (Kendall tau distance) for active learning.

    Extends VarianceUncertaintyEvaluation with Monte Carlo ranking simulation to measure
    how stable the model ranking is given uncertainty in performance estimates.

    Supports the same adaptive threshold strategies as VarianceUncertaintyEvaluation.
    """

    def normalized_kendall_tau_distance(
        self, true_rank: npt.ArrayLike, rank_to_test: npt.ArrayLike
    ):
        """Compute the normalized Kendall tau distance between two rankings.

        Measures the proportion of pairwise disagreements between two rankings.
        A distance of 0 means perfect agreement, 1 means complete disagreement.

        Args:
            true_rank: Reference ranking
            rank_to_test: Ranking to compare against reference

        Returns:
            Normalized distance (float) between 0 and 1

        References:
            - https://en.wikipedia.org/wiki/Kendall_tau_distance
            - https://arxiv.org/pdf/1905.02752
            - https://towardsdatascience.com/comprehensive-guide-to-ranking-evaluation-metrics-7d10382c1025/
        """
        n = len(true_rank)  # type: ignore
        assert len(rank_to_test) == n, "Both lists have to be of equal length"  # type: ignore
        i, j = np.meshgrid(np.arange(n), np.arange(n))

        a = np.argsort(true_rank)
        b = np.argsort(rank_to_test)
        ndisordered = np.logical_or(
            np.logical_and(a[i] < a[j], b[i] > b[j]),
            np.logical_and(a[i] > a[j], b[i] < b[j]),
        ).sum()
        return ndisordered / (n * (n - 1))

    def calculate_tau_distances(
        self,
        embedding_query: npt.ArrayLike,
        training_dataset: pl.DataFrame,
        training_embeddings: np.ndarray | None = None,
        training_performances: np.ndarray | None = None,
    ):
        """Calculate mean Kendall tau distance through Monte Carlo simulation.

        Computes weighted statistics from the training data, then calls `_mc_kendall_taus`
        to estimate ranking instability by sampling from per-model normal distributions.

        Args:
            embedding_query: Query embedding vector
            training_dataset: DataFrame with 'embeddings' and 'models_performance' columns
            training_embeddings: Pre-computed training embeddings matrix
            training_performances: Pre-computed training performances matrix

        Returns:
            Scalar mean Kendall tau distance across all Monte Carlo simulations
        """
        weighted_var, weighted_mean = self.calculate_weighted_stats(
            embedding_query,
            training_dataset,
            training_embeddings,
            training_performances,
        )

        weighted_var = weighted_var + 1e-8  # to avoid zero value

        rng = np.random.default_rng(seed=self.seed)

        knn_rank = np.argsort(-weighted_mean)
        taus = self._mc_kendall_taus(
            rng=rng,
            knn_rank=knn_rank,
            weighted_mean=weighted_mean,
            weighted_var=weighted_var,
        )
        return np.mean(taus)

    def _mc_kendall_taus(
        self,
        rng: np.random.Generator,
        knn_rank: np.ndarray,
        weighted_mean: np.ndarray,
        weighted_var: np.ndarray,
        n_simulations: int = 50,
    ) -> np.ndarray:
        """Run Monte Carlo simulations and compute vectorized Kendall tau distances.

        Samples model performances from normal distributions, then measures ranking
        instability against the reference ranking via Kendall tau distance.

        Args:
            rng: NumPy random generator
            knn_rank: Reference ranking (indices sorted by descending mean performance)
            weighted_mean: Weighted mean performance per model
            weighted_var: Weighted variance per model
            n_simulations: Number of Monte Carlo draws

        Returns:
            Array of normalized Kendall tau distances, shape (n_simulations,)
        """
        n = len(knn_rank)
        simulations = rng.normal(
            loc=weighted_mean,
            scale=np.sqrt(np.maximum(weighted_var, 1e-10)),
            size=(n_simulations, n),
        )
        simulations = np.clip(simulations, 0, 1)

        a = np.argsort(knn_rank)
        b_all = np.argsort(np.argsort(-simulations, axis=1), axis=1)

        a_gt = a[:, None] > a[None, :]
        b_gt = b_all[:, :, None] > b_all[:, None, :]

        pairs = np.logical_xor(a_gt, b_gt).sum(axis=(1, 2))
        return pairs / (n * (n - 1))

    def get_score(
        self,
        embedding_query: npt.ArrayLike,
        training_dataset: pl.DataFrame,
        training_embeddings: np.ndarray | None = None,
        training_performances: np.ndarray | None = None,
    ) -> float:
        """Compute Monte Carlo ranking uncertainty score for a query.

        Args:
            embedding_query: Query embedding vector
            training_dataset: DataFrame with 'embeddings' and 'models_performance' columns
            average: Whether to return mean score (True) or array of scores (False)

        Returns:
            Float uncertainty score
        """
        tau_list = self.calculate_tau_distances(
            embedding_query=embedding_query,
            training_dataset=training_dataset,
            training_embeddings=training_embeddings,
            training_performances=training_performances,
        )
        score = float(tau_list.mean())
        return score

    def get_params(self) -> dict[str, str | float | None]:
        """Get configuration parameters as a dictionary."""
        d_params = super().get_params()
        d_params["unc_strategy"] = "monte_carlo"
        return d_params
