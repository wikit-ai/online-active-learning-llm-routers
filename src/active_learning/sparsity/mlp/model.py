import torch
import torch.nn as nn


class RegressionHead(nn.Module):
    """Small MLP mapping an embedding to a scalar KL divergence estimate."""

    def __init__(self, input_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()  # type: ignore
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)  # type: ignore
