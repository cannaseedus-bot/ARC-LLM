from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .core.manifold import LowRankMetric, curvature_regularizer


@dataclass
class MuARCConfig:
    vocab_size: int = 32000
    dim: int = 128
    depth: int = 2
    hidden_dim: int = 256
    rank: int = 32


class GeodesicAttention(nn.Module):
    """Attention based on learned metric-induced geodesic distances."""

    def __init__(self, dim: int, rank: int = 32):
        super().__init__()
        self.metric = LowRankMetric(dim=dim, rank=rank)

    @staticmethod
    def geodesic_distance_sq(x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        x_i = x.unsqueeze(2)
        x_j = x.unsqueeze(1)
        diff = x_i - x_j
        g_diff = torch.einsum("btjd,dk->btjk", diff, g)
        return torch.einsum("btjd,btjd->btj", g_diff, diff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g = self.metric()
        dist_sq = self.geodesic_distance_sq(x, g)
        attn = F.softmax(-dist_sq, dim=-1)
        return torch.matmul(attn, x)


class MuArcLayer(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, rank: int = 32):
        super().__init__()
        self.attn = GeodesicAttention(dim=dim, rank=rank)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class MuARCLLM(nn.Module):
    """µARC-LLM v0.1: geodesic-attention micro-transformer."""

    def __init__(self, config: MuARCConfig):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList(
            [MuArcLayer(config.dim, config.hidden_dim, rank=config.rank) for _ in range(config.depth)]
        )
        self.norm = nn.LayerNorm(config.dim)
        self.head = nn.Linear(config.dim, config.vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        return self.head(x)

    def curvature_loss(self, weight: float = 1e-3) -> torch.Tensor:
        losses = [curvature_regularizer(layer.attn.metric, weight=weight) for layer in self.layers]
        return torch.stack(losses).sum() if losses else torch.tensor(0.0, device=self.head.weight.device)


def train_step(
    model: MuARCLLM,
    optimizer: torch.optim.Optimizer,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    curvature_weight: float = 1e-3,
) -> float:
    optimizer.zero_grad()
    logits = model(input_ids)
    lm_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
    loss = lm_loss + model.curvature_loss(weight=curvature_weight)
    loss.backward()
    optimizer.step()
    return float(loss.item())
