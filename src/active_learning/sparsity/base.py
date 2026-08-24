from abc import ABC, abstractmethod
from typing import Generic, TypeVar

T = TypeVar("T")


class SparsityLearnerBase(ABC, Generic[T]):
    """Shared scaffolding for online KL-divergence learners.

    Generic over the input type T:
    - T = np.ndarray for embedding-based learners (MLP)

    The learner is updated only when a sample has been selected — i.e. when the
    true KL is available.  For non-selected candidates the learner runs in
    inference-only mode.
    """

    def __init__(self, cold_start_n: int = 20) -> None:
        self.cold_start_n = cold_start_n

    @abstractmethod
    def _update_model(self, input: T, kl: float, /) -> float | None:
        """Feed one (input, kl) observation to the underlying model."""
        ...

    @abstractmethod
    def _predict(self, input: T, /) -> float:
        """Return the raw predicted KL divergence for a single input."""
        ...

    @property
    @abstractmethod
    def has_been_trained(self) -> bool:
        """True once the model has completed at least one training run."""
        ...

    def update(self, input: T, kl: float, /) -> float | None:
        """Register a new (input, true_kl) pair and update the model."""
        return self._update_model(input, float(kl))

    def predict_kl(self, input: T, /) -> float:
        """Return predicted KL divergence, clipped to [0, ∞)."""
        return max(0.0, self._predict(input))
