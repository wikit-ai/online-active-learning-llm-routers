import polars as pl
from .base_strategy import BaseSelectionStrategy


class PassiveStrategy(BaseSelectionStrategy):
    """Selects all items until the training dataset reaches n samples."""

    def __init__(self, n_max: int):
        self.n_max = n_max

    def get_params(self) -> dict[str, str | float]:
        return {"n": self.n_max}

    def should_select(
        self,
        item: dict[str, str | float],
        current_train_ds: pl.DataFrame | None,
    ) -> bool:
        """Select the item if the training dataset has fewer than n samples.

        Args:
            item: The candidate sample to evaluate.
            current_train_ds: The current training dataset, or None if empty.

        Returns:
            True until n samples have been collected, False afterwards.
        """
        if current_train_ds is None or current_train_ds.is_empty():
            return True
        return len(current_train_ds) < self.n_max

    def get_name(self) -> str:
        return f"passive_stop_at_{self.n_max}"
