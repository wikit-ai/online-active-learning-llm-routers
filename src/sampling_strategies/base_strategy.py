from abc import ABC, abstractmethod
import polars as pl


class BaseSelectionStrategy(ABC):
    """Base class for sample selection strategies in streaming learning."""

    @abstractmethod
    def should_select(
        self,
        item: dict[str, str | float],
        current_train_ds: pl.DataFrame,
    ) -> bool:
        """Determine if a sample should be selected for training.

        Args:
            item: The candidate sample to evaluate
            current_train_ds: The current training dataset
            budget: Maximum number of samples that may be selected in total.
                Defaults to 0 (no budget limit) for strategies that ignore it.

        Returns:
            True if the sample should be selected, False otherwise
        """
        pass
