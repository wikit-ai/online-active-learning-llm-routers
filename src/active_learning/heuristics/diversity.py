from typing import Optional, Literal

import numpy as np
import numpy.typing as npt
from numpy.typing import ArrayLike

import polars as pl

from sklearn.metrics.pairwise import cosine_similarity

from active_learning.heuristics.base import BaseHeuristic


class MinCosineEvaluation(BaseHeuristic):
    """Selects samples that are dissimilar to the current training set (diversity heuristic).

    A sample is selected when its aggregated cosine similarity to training embeddings
    is below a threshold, meaning it covers a region not yet well represented.

    Supports two threshold strategies:
    - 'quantile': Adaptive threshold computed as a quantile of recent similarity scores
    - 'fixed': Static threshold defined at construction time
    """

    def __init__(
        self,
        strategy: Literal["quantile", "fixed"] = "quantile",
        type_aggregation: Literal["min", "average"] = "average",
        window_size: int = 20,
        fixed_quantile: Optional[float] = 0.15,
        fixed_threshold: float = 0.2,
    ):
        """Initialize diversity evaluation based on cosine similarity.

        Args:
            strategy: Threshold strategy — 'quantile' adapts from recent scores, 'fixed' is static.
            type_aggregation: How similarities to training points are collapsed —
                'average' uses the mean, 'min' uses the minimum.
            window_size: Number of recent similarity scores retained for adaptive strategies.
            fixed_quantile: Quantile level for 'quantile' strategy (must be in [0, 1]).
            fixed_threshold: Similarity threshold for 'fixed' strategy; samples with score
                below this value are selected.
        """
        super().__init__(window_size=window_size)
        self.strategy = strategy
        self.type_aggregation = type_aggregation
        self.fixed_quantile = fixed_quantile
        self.fixed_threshold = fixed_threshold

        self.min_core_set_similarity = 1

        if strategy == "quantile" and self.fixed_quantile is None:
            raise ValueError("window_quantile must be provided for 'quantile' strategy")

    def _compute_threshold(self) -> float:
        """Compute threshold based on configured strategy."""
        if self.strategy == "quantile":
            if not self._window:
                return self.fixed_threshold
            self.threshold = float(
                np.quantile(a=self._window, q=self.fixed_quantile)  # type: ignore
            )
            return self.threshold
        else:  # 'fixed'
            return self.fixed_threshold

    def get_score(
        self,
        embedding_query: npt.ArrayLike,
        training_dataset: pl.DataFrame,
        training_embeddings: np.ndarray | None = None,
    ) -> float:
        """Compute the aggregated cosine similarity of a query to the training set.

        Args:
            embedding_query: Query embedding to evaluate, shape (d,) or (1, d).
            training_dataset: DataFrame with an 'embeddings' column; used only when
                training_embeddings is None.
            training_embeddings: Pre-stacked training embeddings array of shape (N, d).
                Skips extraction from training_dataset when provided.

        Returns:
            Scalar similarity score: mean or minimum cosine similarity depending on
            type_aggregation. Lower values indicate a more novel query.
        """
        if training_embeddings is None:
            training_embeddings = np.vstack(training_dataset["embeddings"].to_list())
        embedding_query = np.asarray(embedding_query).reshape(1, -1)

        cosine_list = cosine_similarity(embedding_query, training_embeddings)
        if self.type_aggregation == "average":
            score = float(cosine_list.mean())
        elif self.type_aggregation == "min":
            score = float(cosine_list.min())
        else:
            raise NotImplementedError()
        return score

    def get_params(self) -> dict[str, float | str | None]:
        """Return the configuration parameters of this heuristic for logging/serialization."""
        d_params: dict[str, float | str | None] = {
            "threshold_strategy": self.strategy,
        }
        if self.strategy == "fixed":
            d_params["fixed_threshold"] = self.fixed_threshold
        elif self.strategy == "quantile":
            d_params["window_size"] = self.window_size
            d_params["fixed_quantile"] = self.fixed_quantile

        return d_params

    def __call__(
        self,
        embedding_query: npt.ArrayLike,
        training_dataset: pl.DataFrame | None,
    ) -> bool:
        """Evaluate if query is representative enough to be added to training set.

        Args:
            embedding_query: Query embedding to evaluate
            training_dataset: Current training dataset with 'embeddings' column

        Returns:
            True if sample should be selected (low similarity = more representative)
        """
        if training_dataset is None or len(training_dataset) == 0:
            return True

        cosine_agg_value = self.get_score(
            embedding_query=embedding_query, training_dataset=training_dataset
        )
        if self.strategy != "fixed":
            self._update_window(cosine_agg_value)

        threshold = self._compute_threshold()
        return cosine_agg_value <= threshold


class VmFCoverage(BaseHeuristic):
    """Evaluates coverage of samples for active learning using VMF KDE log-likelihood.

    Uses von Mises-Fisher kernel density estimation on the unit sphere.
    If a new sample has a low log-likelihood, it is mostly an outlier.

    Supports multiple threshold strategies:
    - 'fixed': Uses fixed thresholds for LL
    - 'quantile': Uses a quantile of recent scores (adaptive)
    """

    def __init__(
        self,
        strategy: Literal["fixed", "quantile"] = "fixed",
        window_size: int = 20,
        fixed_quantile: Optional[float] = 0.15,
        fixed_ll_threshold: float = 0.0,
        min_samples_for_kde: int = 10,
        kappa: float = 25.0,
        reconstruction_weighting: bool = False,
        k: int = 20,
    ):
        """Initialize vMF coverage evaluation.

        Args:
            strategy: Threshold strategy — 'fixed' uses fixed_ll_threshold, 'quantile'
                adapts from recent log-likelihood scores.
            window_size: Number of recent log-likelihood scores retained for adaptive
                strategies.
            fixed_quantile: Quantile level for 'quantile' strategy (must be in [0, 1]).
            fixed_ll_threshold: Log-likelihood threshold for 'fixed' strategy; samples
                with score below this value are selected.
            min_samples_for_kde: Minimum number of training samples required before KDE
                is computed; queries are always selected below this count.
            kappa: Concentration (bandwidth) parameter of the vMF kernel. Higher values
                produce sharper kernels.
            reconstruction_weighting: Reserved for future use.
            k: Reserved for future use.
        """
        super().__init__(window_size=window_size)

        self.strategy = strategy
        self.fixed_quantile = fixed_quantile
        self.fixed_ll_threshold = fixed_ll_threshold
        self.min_samples_for_kde = min_samples_for_kde
        self.kappa = kappa
        self.threshold = fixed_ll_threshold
        self.reconstruction = reconstruction_weighting
        self.k = k

    def vmf_log_normalizer(self, kappa: np.ndarray, d: int) -> np.ndarray:
        """Compute the log-normalizer of the von Mises-Fisher distribution.

        Uses the Stirling-based approximation from the paper for numerical stability
        in high dimensions.

        Args:
            kappa: Concentration parameter(s)
            d: Dimensionality of the embedding space.

        Returns:
            Log-normalizer values
        """
        a = (d - 1) / 2.0
        b = (d + 1) / 2.0

        sqrt1 = np.sqrt(a**2 + kappa**2)
        sqrt2 = np.sqrt(b**2 + kappa**2)

        return (
            ((d - 1) / 4.0) * np.log(a + sqrt1)
            - 0.5 * sqrt1
            + ((d - 1) / 4.0) * np.log(a + sqrt2)
            - 0.5 * sqrt2
        )

    def vmf_kde_log_likelihood(
        self,
        x_query: np.ndarray,
        x_train: np.ndarray,
        kappa: float,
    ) -> np.ndarray:
        """
        ln p(x) = F_d(kappa) + logsumexp(kappa * x_train @ x_query.T) - ln(N)

        Returns log-likelihood for each query point
        """
        d = x_train.shape[-1]
        n = x_train.shape[0]

        cos_sim = x_train @ x_query.T
        log_kernels = kappa * cos_sim
        a_max = np.max(
            log_kernels, axis=0
        )  # https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/

        vmf_function = np.exp(log_kernels - a_max)

        weighted_sum = np.sum(vmf_function, axis=0)
        lse = np.log(np.maximum(weighted_sum, 1e-300)) + a_max

        log_norm = self.vmf_log_normalizer(np.array([kappa]), d).item()

        return log_norm + lse - np.log(n)

    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Project embeddings onto the unit sphere (L2 normalization along last axis)."""
        return x / np.linalg.norm(x, axis=-1, keepdims=True)

    def _compute_log_likelihood(
        self,
        embedding_query: ArrayLike,
        training_dataset: pl.DataFrame,
        training_embeddings: np.ndarray | None = None,
    ) -> float:
        """Compute the vMF KDE log-likelihood of a query given the training distribution.

        Normalizes both training and query embeddings to the unit sphere before
        calling vmf_kde_log_likelihood.

        Args:
            embedding_query: Query embedding
            training_dataset: DataFrame with an 'embeddings' column; used only when
                training_embeddings is None.
            training_embeddings: Pre-computed training embedding array

        Returns:
            Scalar log-likelihood (averaged over query points if M > 1).
        """
        annotated_embeddings = (
            training_embeddings
            if training_embeddings is not None
            else np.array(
                [np.array(emb) for emb in training_dataset["embeddings"].to_list()]
            )
        )
        embedding_query = np.asarray(embedding_query)

        x_train = self.normalize(annotated_embeddings)
        x_query = self.normalize(embedding_query)

        ll = self.vmf_kde_log_likelihood(x_query, x_train, self.kappa)
        return float(ll.mean())

    def get_score(
        self,
        embedding_query: ArrayLike,
        training_dataset: pl.DataFrame,
        training_embeddings: np.ndarray | None = None,
    ) -> float:
        """Compute vMF KDE log-likelihood score for a query embedding.

        Args:
            embedding_query: Query embedding to evaluate
            training_dataset: DataFrame containing training embeddings

        Returns:
            Log-likelihood score (higher = better covered by training distribution)
        """
        score = self._compute_log_likelihood(
            embedding_query,
            training_dataset,
            training_embeddings,
        )
        return score

    def _compute_thresholds(self) -> float:
        """Compute the log-likelihood threshold based on the configured strategy.

        Returns:
            Threshold value; scores below this indicate under-covered regions.
        """
        if self.strategy == "quantile":
            if not self._window:
                return self.fixed_ll_threshold
            return float(np.quantile(self._window, self.fixed_quantile))  # type: ignore
        else:  # 'fixed'
            return self.fixed_ll_threshold

    def get_params(self) -> dict[str, float | int | str]:
        """Return the configuration parameters of this heuristic for logging/serialization."""
        d_params: dict[str, float | int | str] = {
            "threshold_strategy": self.strategy,
            "min_samples_for_kde": self.min_samples_for_kde,
            "kappa": self.kappa,
        }
        if self.strategy == "fixed":
            d_params["fixed_ll_threshold"] = self.fixed_ll_threshold
        elif self.strategy == "quantile":
            d_params["window_size"] = self.window_size
            d_params["fixed_quantile"] = self.fixed_quantile  # type: ignore

        return d_params

    def __call__(
        self,
        embedding_query: ArrayLike,
        training_dataset: pl.DataFrame | None,
    ) -> bool:
        """Evaluate if query should be added based on coverage criteria.

        Args:
            embedding_query: Query embedding to evaluate
            training_dataset: Current training dataset with 'embeddings' column

        Returns:
            True if sample should be selected (increases coverage)
        """
        if training_dataset is None or len(training_dataset) < self.min_samples_for_kde:
            return True

        ll_thresh = self._compute_thresholds()

        ll_score = self.get_score(embedding_query, training_dataset)
        self._update_window(ll_score)

        return ll_score < ll_thresh
