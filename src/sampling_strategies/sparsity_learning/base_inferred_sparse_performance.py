from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import polars as pl

from sampling_strategies.base_strategy import BaseSelectionStrategy
from sampling_strategies.oracle_sparse_performance import _kl_from_uniform  # type: ignore

from logging_config import logger


class BaseInferredSparsePerformanceStrategy(BaseSelectionStrategy, ABC):
    """Template for strategies that infer KL divergence from an item and update
    a learner incrementally whenever a sample is selected.

    """

    def __init__(
        self, cold_start_n: int, percentile: int, budget: int, seed: int | None = None
    ) -> None:
        self.cold_start_n = cold_start_n
        self.percentile = percentile
        self.budget = budget
        self.seed = seed

        self.learner: Any = None
        self._kl_history: list[float] = []
        self._val_loss: float = np.inf
        self._cold_start_buffer: list[tuple[Any, float]] = []

    @abstractmethod
    def _extract_input(self, item: dict[str, str | float | int]) -> Any:
        """Return the learner input extracted from *item* (e.g. embedding or text)."""

    @abstractmethod
    def _get_learner(self, inp: Any = None) -> Any:
        """Return (or lazily construct) the active learner."""

    def _predict_kl(self, inp: Any) -> float:
        learner = self._get_learner(inp)
        return learner.predict_kl(inp)

    def _guardrail_degraded_performance(self, inp: Any, true_kl: float) -> bool | None:
        """Force selection when val_loss drifts above 0.20 after cold start.

        Logs every time it fires so degradation episodes are visible.  Returns
        ``True`` when active, ``None`` to let normal selection proceed.
        """
        if self._val_loss < 0.20:
            return None

        logger.info(
            f"[Guardrail] val_loss={self._val_loss:.4f} >= 0.20 — "
            f"forcing selection and retraining (n_selected={len(self._kl_history)})"
        )
        learner = self._get_learner(inp)
        self._kl_history.append(true_kl)
        val_loss = learner.update(inp, true_kl)
        if val_loss is not None:
            self._val_loss = val_loss
        return True

    def _cold_start(self, inp: Any, true_kl: float) -> bool | None:
        """Buffer items during cold start and select all unconditionally.

        Calls ``learner.fit`` once the buffer reaches ``cold_start_n`` items and
        records the resulting validation loss.  Returns ``True`` while active,
        ``None`` once the cold-start phase is complete.
        """
        if len(self._kl_history) >= self.cold_start_n:
            return None

        self._cold_start_buffer.append((inp, true_kl))
        self._kl_history.append(true_kl)
        if len(self._kl_history) == self.cold_start_n:
            learner = self._get_learner(inp)
            inputs = [x for x, _ in self._cold_start_buffer]
            kls = [k for _, k in self._cold_start_buffer]
            self._val_loss = learner.fit(inputs, kls)
            self._cold_start_buffer = []
        return True

    def should_select(
        self,
        item: dict[str, str | float | int],
        current_train_ds: pl.DataFrame | None,
    ) -> bool:
        inp = self._extract_input(item)
        true_kl = _kl_from_uniform(np.array(item["models_performance"]))

        if len(self._kl_history) >= self.budget:
            return False

        if (result := self._cold_start(inp, true_kl)) is not None:
            return result

        if (result := self._guardrail_degraded_performance(inp, true_kl)) is not None:
            return result

        predicted_kl = self._predict_kl(inp)
        threshold = float(np.percentile(self._kl_history, self.percentile))

        if predicted_kl > threshold:
            self._kl_history.append(true_kl)
            learner = self._get_learner(inp)
            val_loss = learner.update(inp, true_kl)
            if val_loss is not None:
                self._val_loss = val_loss
            return True

        return False
