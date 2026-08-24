import random
import polars as pl
from .base_strategy import BaseSelectionStrategy


class RandomStrategy(BaseSelectionStrategy):
    """Selects items randomly with a fixed probability — a stochastic baseline strategy."""

    def __init__(self, prob_true: float):
        """Args:
            prob_true: Probability of selecting any given item (between 0 and 1).
        """
        self.prob_true = prob_true

    def get_params(self) -> dict[str, str | float]:
        return {}

    def should_select(
        self,
        item: dict[str, str | float],
        current_train_ds: pl.DataFrame,
    ) -> bool:
        """Randomly selects the item based on prob_true.

        Args:
            item: The candidate sample to evaluate.
            current_train_ds: The current training dataset.

        Returns:
            True with probability prob_true, False otherwise.
        """
        selection = random.choices(
            [True, False], weights=[self.prob_true, 1 - self.prob_true]
        )[0]
        return selection

    def get_name(self) -> str:
        """Return the strategy name."""
        return "random"
