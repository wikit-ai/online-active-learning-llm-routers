from typing import Any

import numpy as np

from active_learning.sparsity.mlp.active_learner import SparsityActiveLearner
from sampling_strategies.sparsity_learning.base_inferred_sparse_performance import (
    BaseInferredSparsePerformanceStrategy,
)


class InferredSparsePerformanceStrategy(BaseInferredSparsePerformanceStrategy):
    """Infers KL divergence from precomputed embeddings via a small MLP."""

    def __init__(
        self,
        budget: int,
        model_settings: dict[str, Any] | None = None,
        training_settings: dict[str, Any] | None = None,
        cold_start_n: int = 50,
        percentile: int = 75,
        seed: int | None = None,
    ) -> None:
        super().__init__(
            cold_start_n=cold_start_n, percentile=percentile, budget=budget, seed=seed
        )
        self.learner = None
        model_settings = model_settings or {}
        training_settings = training_settings or {}

        self._hidden_dim = model_settings.get("hidden_dim", 16)

        self._learning_rate = training_settings.get("learning_rate", 1e-3)
        self._weight_decay = training_settings.get("weight_decay", 1e-3)
        self._loader_batch_size = training_settings.get("loader_batch_size", 16)
        self._n_epochs = training_settings.get("n_epochs", 50)
        self._trigger_every = training_settings.get("trigger_every", 5)

    def _extract_input(self, item: dict[str, str | float | int]) -> np.ndarray:
        return np.array(item["embeddings"])

    def _get_learner(self, inp: np.ndarray | None = None) -> SparsityActiveLearner:
        if self.learner is None:
            assert inp is not None
            self.learner = SparsityActiveLearner(
                embedding_dim=inp.shape[-1],
                hidden_dim=self._hidden_dim,
                learning_rate=self._learning_rate,
                weight_decay=self._weight_decay,
                loader_batch_size=self._loader_batch_size,
                n_epochs=self._n_epochs,
                trigger_every=self._trigger_every,
                cold_start_n=self.cold_start_n,
                seed=self.seed,
            )
        return self.learner

    def get_params(self) -> dict[str, str | float]:
        return {"cold_start_n": self.cold_start_n}

    def get_name(self) -> str:
        return "inferred_sparse_nn"
