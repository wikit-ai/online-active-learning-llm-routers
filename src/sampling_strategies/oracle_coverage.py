from collections import Counter
import polars as pl
from sampling_strategies.base_strategy import BaseSelectionStrategy


class OracleCoverageStrategy(BaseSelectionStrategy):
    """Selects items to match the corpus dataset distribution up to a fixed budget."""

    def __init__(self, counter: Counter[str], n_corpus: int = 100):
        """Args:
            counter: Frequency count of each dataset in the full corpus.
            n_corpus: Total number of samples to collect across all datasets.
        """
        self.dict_datasets_seen: dict[str, int] = {}

        self.prop_datasets = self.get_proportion_per_dataset(
            counter=counter, n_corpus=n_corpus
        )

    def get_params(self) -> dict[str, str | float]:
        return {}

    def get_proportion_per_dataset(
        self, counter: Counter[str], n_corpus: int
    ) -> dict[str, int]:
        """Compute per-dataset sample quotas proportional to corpus frequencies.

        Args:
            counter: Frequency count of each dataset in the full corpus.
            n_corpus: Total budget to distribute across datasets.

        Returns:
            Mapping from dataset name to its sample quota (at least 1 per dataset).
        """
        total = sum(counter.values())
        return {
            str(name): max(1, int((count / total) * n_corpus))
            for name, count in counter.items()
        }

    def should_select(
        self,
        item: dict[str, str | float | int],
        current_train_ds: pl.DataFrame,
    ) -> bool:
        """Select the item if its dataset quota has not yet been filled.

        Args:
            item: Candidate sample; must contain a "dataset" key.
            current_train_ds: The current training dataset.

        Returns:
            True if the dataset's quota allows another sample, False otherwise.
        """
        ds: str = str(item["dataset"])
        if ds not in self.dict_datasets_seen:
            self.dict_datasets_seen[ds] = 0

        if self.dict_datasets_seen[ds] <= self.prop_datasets[ds]:
            self.dict_datasets_seen[ds] += 1
            return True
        else:
            return False

    def get_name(self) -> str:
        """Return the strategy name."""
        return "oracle_ds"
