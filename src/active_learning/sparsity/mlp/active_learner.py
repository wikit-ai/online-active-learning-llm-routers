import copy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from active_learning.sparsity.base import SparsityLearnerBase
from active_learning.sparsity.mlp.data_loader import KLDataset, _KLItem  # type: ignore
from active_learning.sparsity.mlp.model import RegressionHead


class SparsityActiveLearner(SparsityLearnerBase[np.ndarray]):
    """Predicts KL divergence from embedding via an online neural network.

    Each time the dataset grows by trigger_every samples, the model is retrained
    from scratch for n_epochs full passes over all accumulated data using a
    DataLoader built from KLDataset.
    """

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int = 16,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-3,
        loader_batch_size: int = 16,
        n_epochs: int = 20,
        trigger_every: int = 20,
        cold_start_n: int = 100,
        val_fraction: float = 0.15,
        patience: int = 10,
        seed: int | None = None,
    ) -> None:
        """Initialise the learner and its underlying MLP.

        Args:
            embedding_dim: Dimensionality of the input embedding vectors.
            hidden_dim: Width of the hidden layer in the regression head.
            learning_rate: AdamW learning rate.
            weight_decay: AdamW L2 penalty.
            loader_batch_size: Mini-batch size used during training.
            n_epochs: Maximum number of full passes over the training data
                per retraining round.
            trigger_every: Number of new (embedding, kl) observations required
                to trigger a retraining round.
            cold_start_n: Minimum dataset size before inference is allowed;
                inherited from SparsityLearnerBase.
            val_fraction: Fraction of the *old* data (i.e. not the newest
                trigger batch) held out for validation and early stopping.
            patience: Early-stopping patience in epochs.
        """
        super().__init__(cold_start_n=cold_start_n)
        self.seed = seed
        if seed is not None:
            torch.manual_seed(seed)  # type: ignore
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)
        self.n_epochs = n_epochs
        self.trigger_every = trigger_every
        self.loader_batch_size = loader_batch_size
        self.val_fraction = val_fraction
        self.patience = patience

        self.dataset = KLDataset()
        self.model = RegressionHead(input_dim=embedding_dim, hidden_dim=hidden_dim)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        self.loss_fn = nn.MSELoss()
        self._n_updates: int = 0

    @property
    def has_been_trained(self) -> bool:
        """True once at least one retraining round has completed."""
        return self._n_updates > 0

    def _split_dataset(
        self, n_new: int
    ) -> tuple[Subset[_KLItem], Subset[_KLItem] | None]:
        """Carve train/val splits, keeping the newest n_new points in train."""
        n_total = len(self.dataset)
        n_old = n_total - n_new
        new_indices = list(range(n_old, n_total))

        n_val = max(1, int(n_old * self.val_fraction)) if n_old >= 10 else 0
        if n_val == 0:
            return Subset(self.dataset, list(range(n_total))), None

        perm = np.random.default_rng(self.seed).permutation(n_old).tolist()
        val_indices = perm[:n_val]
        train_indices = perm[n_val:] + new_indices
        return Subset(self.dataset, train_indices), Subset(self.dataset, val_indices)

    def _compute_mse(self, split: Subset[_KLItem]) -> float:
        """Return MSE of the current model on a dataset split."""
        loader = DataLoader(split, batch_size=len(split), shuffle=False)
        self.model.eval()
        with torch.no_grad():
            embeddings, kls = next(iter(loader))
            return self.loss_fn(self.model(embeddings).squeeze(-1), kls).item()

    def _run_epoch(self, train_loader: DataLoader[_KLItem]) -> None:
        """Run one gradient-descent pass over the training data."""
        self.model.train()
        for embeddings, kls in train_loader:
            self.optimizer.zero_grad()
            preds = self.model(embeddings).squeeze(-1)
            self.loss_fn(preds, kls).backward()
            self.optimizer.step()  # type: ignore[misc]

    def _log_training_result(
        self,
        epoch: int,
        train_mse: float,
        best_val_mse: float,
        new_batch_mse: float | None,
    ) -> None:
        """Print a one-line training summary."""
        val_str = (
            f"val MSE: {best_val_mse:.6f}"
            if best_val_mse < float("inf")
            else "no val set yet"
        )
        new_batch_str = (
            f" | new-batch MSE (before train): {new_batch_mse:.6f}"
            if new_batch_mse is not None
            else ""
        )
        print(
            f"[SparsityActiveLearner] train #{self._n_updates + 1} "
            f"(epoch {epoch + 1}/{self.n_epochs}) — "
            f"train MSE: {train_mse:.6f} | {val_str}{new_batch_str}"
        )

    def _train(self, new_batch_mse: float | None = None, n_new: int = 0) -> float:
        """Retrain the model on all accumulated data with early stopping.

        A held-out validation set is taken from the *old* data (all entries
        before the current trigger batch), so fresh observations are never
        wasted on validation.  The best checkpoint is restored when early
        stopping fires.

        Args:
            new_batch_mse: Pre-training MSE on the newest trigger batch,
                logged for monitoring.  None before the first training round.
            n_new: Number of newest entries belonging to the current trigger
                batch; always placed in the training split.

        Returns:
            Best validation MSE achieved (or inf when there is no val set).
        """
        train_split, val_split = self._split_dataset(n_new)
        generator = (
            torch.Generator().manual_seed(self.seed) if self.seed is not None else None
        )
        train_loader = DataLoader(
            train_split,
            batch_size=self.loader_batch_size,
            shuffle=True,
            generator=generator,
        )

        best_val_mse = float("inf")
        epochs_without_improvement = 0
        best_weights = copy.deepcopy(self.model.state_dict())
        epoch = 0

        for epoch in range(self.n_epochs):
            self._run_epoch(train_loader)

            if val_split is not None:
                val_mse = self._compute_mse(val_split)
                if val_mse < best_val_mse:
                    best_val_mse = val_mse
                    epochs_without_improvement = 0
                    best_weights = copy.deepcopy(self.model.state_dict())
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= self.patience:
                        self.model.load_state_dict(best_weights)
                        print(
                            f"[SparsityActiveLearner] early stop at epoch {epoch + 1} "
                            f"(no val improvement for {self.patience} epochs, "
                            f"best val MSE: {best_val_mse:.6f})"
                        )
                        break

        train_mse = self._compute_mse(train_split)
        self._log_training_result(epoch, train_mse, best_val_mse, new_batch_mse)
        self._n_updates += 1
        return best_val_mse

    def _eval_new_batch(self) -> float:
        """Compute MSE on the newest trigger_every points without updating weights.

        Because these points were added since the last retraining round, the
        model has never seen them: this gives the only truly held-out signal
        of generalization quality.  Called by _update_model just before
        retraining so the score can be logged alongside the new training run.

        Returns:
            MSE of the current model on the newest trigger batch.
        """
        n = self.trigger_every
        indices = list(range(len(self.dataset) - n, len(self.dataset)))
        subset = Subset(self.dataset, indices)
        loader = DataLoader(subset, batch_size=n, shuffle=False)
        self.model.eval()
        with torch.no_grad():
            embeddings, kls = next(iter(loader))
            mse = self.loss_fn(self.model(embeddings).squeeze(-1), kls).item()
        return mse

    def _update_model(self, embedding: np.ndarray, kl: float) -> float | None:
        """Append one observation and retrain when the trigger threshold is reached.

        The dataset grows by one entry on every call.  Retraining fires only
        when the dataset size is an exact multiple of trigger_every.

        Returns:
            Best validation MSE from the retraining run, or None if no
            retraining was triggered this call.
        """
        self.dataset.add(embedding, kl)
        if len(self.dataset) % self.trigger_every == 0:
            new_batch_mse = self._eval_new_batch() if self.has_been_trained else None
            val_loss = self._train(
                new_batch_mse=new_batch_mse, n_new=self.trigger_every
            )
            return val_loss

    def fit(self, embeddings: list[np.ndarray], kls: list[float]) -> float:
        """Bulk-load (embedding, kl) pairs and run a single training pass.

        Bypasses the trigger_every mechanism — useful for an initial warm-start
        before switching to online updates via update().

        Returns:
            Best validation MSE from the training run.
        """
        for embedding, kl in zip(embeddings, kls):
            self.dataset.add(np.asarray(embedding, dtype=float), float(kl))
        val_loss = self._train()
        return val_loss

    def _predict(self, embedding: np.ndarray) -> float:
        """Return the raw MLP prediction for a single embedding vector."""
        x = torch.tensor(embedding, dtype=torch.float32)
        self.model.eval()
        with torch.no_grad():
            return float(self.model(x))
