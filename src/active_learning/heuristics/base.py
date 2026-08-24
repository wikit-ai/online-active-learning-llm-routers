from abc import ABC, abstractmethod

import numpy.typing as npt
import polars as pl


class BaseHeuristic(ABC):
    """Abstract base class for active learning heuristics.

    Provides shared EMA tracking and sliding window utilities used by
    uncertainty and diversity heuristics.
    """

    def __init__(self, window_size: int):
        self.window_size = window_size
        self._window: list[float] = []

    def _update_window(self, value: float) -> None:
        """Append value to the sliding window, dropping oldest if over capacity."""
        self._window.append(value)
        if len(self._window) > self.window_size:
            self._window = self._window[-self.window_size :]

    @abstractmethod
    def get_score(self, *args, **kwargs) -> float: ...

    @abstractmethod
    def get_params(self) -> dict: ...

    @abstractmethod
    def __call__(
        self,
        embedding_query: npt.ArrayLike,
        training_dataset: pl.DataFrame | None,
    ) -> bool: ...
