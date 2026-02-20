from __future__ import annotations

import torch
import torch.nn as nn


class LowRankMetric(nn.Module):
    """Low-rank PSD metric: g = A^T A + lambda * I."""

    def __init__(self, dim: int, rank: int = 32, init_scale: float = 0.02, init_lambda: float = 0.1):
        super().__init__()
        self.dim = dim
        self.rank = min(rank, dim)
        self.A = nn.Parameter(torch.randn(self.rank, dim) * init_scale)
        self.log_lambda = nn.Parameter(torch.log(torch.tensor(init_lambda)))

    def forward(self) -> torch.Tensor:
        g = self.A.transpose(0, 1) @ self.A
        lam = torch.exp(self.log_lambda)
        eye = torch.eye(self.dim, device=g.device, dtype=g.dtype)
        return g + lam * eye


def curvature_regularizer(metric: LowRankMetric, weight: float = 1e-3) -> torch.Tensor:
    """Simple metric regularizer to avoid uncontrolled metric growth."""
    g = metric()
    eye = torch.eye(g.size(0), device=g.device, dtype=g.dtype)
    return weight * torch.norm(g - eye, p="fro")
