import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

_KLItem = tuple[torch.Tensor, torch.Tensor]


class KLDataset(Dataset[_KLItem]):
    """Accumulates (embedding, kl) pairs observed during active learning.

    Samples are added incrementally via add(). The dataset can be wrapped in a
    DataLoader at any point; the returned loader reflects all samples added so far.
    """

    def __init__(self) -> None:
        self._embeddings: list[np.ndarray] = []
        self._kls: list[float] = []

    def add(self, embedding: np.ndarray, kl: float) -> None:
        """Append a new (embedding, kl) pair to the dataset.

        Args:
            embedding: Feature vector for the sample.
            kl: KL divergence value associated with the sample.
        """
        self._embeddings.append(embedding)
        self._kls.append(float(kl))

    def __len__(self) -> int:
        """Return the number of samples accumulated so far."""
        return len(self._kls)

    def __getitem__(self, idx: int) -> _KLItem:
        """Return the (embedding, kl) pair at position *idx* as float32 tensors."""
        x = torch.tensor(self._embeddings[idx], dtype=torch.float32)
        y = torch.tensor(self._kls[idx], dtype=torch.float32)
        return x, y

    @property
    def kls(self) -> list[float]:
        """All KL divergence values accumulated so far, in insertion order."""
        return self._kls

    def build_loader(
        self, batch_size: int = 16, shuffle: bool = True
    ) -> DataLoader[_KLItem]:
        """Return a DataLoader over all accumulated samples."""
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle)
