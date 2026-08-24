import numpy as np
import polars as pl

from sampling_strategies.base_strategy import BaseSelectionStrategy


def _kl_from_uniform(perf: np.ndarray) -> float:
    """KL divergence from the given distribution to a uniform distribution.

    KL(P || Uniform) = log(n) - H(P), where H(P) is Shannon entropy.
    The performance vector is normalized to a valid probability distribution
    before computation; zeros are replaced with a small epsilon to avoid
    log(0).
    """
    eps = 1e-10
    p = np.array(perf, dtype=float)
    p = np.clip(p, 0, None)
    total = p.sum()
    if total <= 0:
        return 0.0
    p = p / total
    p = np.where(p == 0, eps, p)
    n = len(p)
    entropy = -np.sum(p * np.log(p))
    return float(np.log(n) - entropy)


class OracleSparsePerformanceStrategy(BaseSelectionStrategy):
    """Oracle strategy that selects items whose model-performance distribution
    diverges more from uniform than the 75th-quantile KL of all selected items.
    """

    def __init__(
        self,
        cold_start_n: int = 50,
        kl_quantile: float = 0.75,
    ):
        """Args:
        cold_start_n: Number of items to select unconditionally at the start.
        kl_quantile: Quantile of KL history used as selection threshold after cold start.
        """
        self.cold_start_n = cold_start_n
        self.kl_quantile = kl_quantile
        self._kl_history: list[float] = []

    def get_params(self) -> dict[str, str | float]:
        return {
            "cold_start_n": self.cold_start_n,
            "kl_quantile": self.kl_quantile,
        }

    def _register(self, candidate_kl: float) -> None:
        """Record a selected item's KL divergence into the history."""
        self._kl_history.append(candidate_kl)

    def should_select(
        self,
        item: dict[str, str | float | int],
        current_train_ds: pl.DataFrame | None,
    ) -> bool:
        """Select the item if its KL divergence exceeds the 75th percentile of history.

        During cold start, all items are selected unconditionally. After cold start,
        an item is selected if its performance distribution diverges from uniform more
        than 75% of previously selected items.

        Args:
            item: Candidate sample; must contain a "models_performance" key with a
                numeric array of per-model performance values.
            current_train_ds: The current training dataset, or None if empty.

        Returns:
            True if the item should be added to the training set, False otherwise.
        """
        candidate_kl = _kl_from_uniform(np.array(item["models_performance"]))

        if current_train_ds is None or len(current_train_ds) < self.cold_start_n:
            self._register(candidate_kl)
            return True

        kl_threshold = (
            float(np.quantile(self._kl_history, self.kl_quantile))
            if self._kl_history
            else 0.0
        )

        if candidate_kl > kl_threshold:
            self._register(candidate_kl)
            return True
        return False

    def get_name(self) -> str:
        return "oracle_kl_above_q75"
