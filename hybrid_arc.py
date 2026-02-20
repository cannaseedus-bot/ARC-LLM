"""Hybrid ARC: product-manifold transformer block.

Manifold: R^{d_e} x B_c^{d_h}
- Euclidean subspace for local composition
- Hyperbolic Poincare-ball subspace for hierarchy
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-5


class PoincareBall(nn.Module):
    def __init__(self, c: float = 1.0):
        super().__init__()
        if c <= 0:
            raise ValueError("Curvature parameter c must be positive")
        self.register_buffer("c", torch.tensor(float(c)))

    def _c_like(self, ref: torch.Tensor) -> torch.Tensor:
        return self.c.to(device=ref.device, dtype=ref.dtype)

    def _sqrt_c(self, ref: torch.Tensor) -> torch.Tensor:
        return self._c_like(ref).sqrt()

    def project(self, x: torch.Tensor, margin: float = 1e-3) -> torch.Tensor:
        sqrt_c = self._sqrt_c(x)
        max_norm = (1.0 - margin) / sqrt_c
        norm = torch.norm(x, dim=-1, keepdim=True).clamp_min(EPS)
        scale = torch.clamp(max_norm / norm, max=1.0)
        return x * scale

    def mobius_add(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        c = self._c_like(x)
        x2 = (x * x).sum(dim=-1, keepdim=True)
        y2 = (y * y).sum(dim=-1, keepdim=True)
        xy = (x * y).sum(dim=-1, keepdim=True)
        num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
        den = 1 + 2 * c * xy + (c * c) * x2 * y2
        out = num / den.clamp_min(EPS)
        return self.project(out)

    def exp0(self, v: torch.Tensor) -> torch.Tensor:
        sqrt_c = self._sqrt_c(v)
        norm = torch.norm(v, dim=-1, keepdim=True).clamp_min(EPS)
        out = torch.tanh(sqrt_c * norm) * v / (sqrt_c * norm)
        return self.project(out)

    def log0(self, x: torch.Tensor) -> torch.Tensor:
        x = self.project(x)
        sqrt_c = self._sqrt_c(x)
        norm = torch.norm(x, dim=-1, keepdim=True).clamp_min(EPS)
        scaled = (sqrt_c * norm).clamp_max(1 - 1e-5)
        return torch.atanh(scaled) * x / (sqrt_c * norm)

    def dist(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        sqrt_c = self._sqrt_c(x)
        diff = self.mobius_add(-x, y)
        norm = torch.norm(diff, dim=-1).clamp_max((1 - 1e-5) / sqrt_c)
        return (2.0 / sqrt_c) * torch.atanh((sqrt_c * norm).clamp_max(1 - 1e-5))


@dataclass
class HybridARCConfig:
    vocab_size: int = 32000
    dim_e: int = 128
    dim_h: int = 128
    depth: int = 4
    num_heads: int = 8
    c: float = 1.0
    mlp_ratio: int = 4

    @property
    def total_dim(self) -> int:
        return self.dim_e + self.dim_h


class HybridAttention(nn.Module):
    def __init__(self, dim_e: int, dim_h: int, num_heads: int = 8, c: float = 1.0):
        super().__init__()
        self.dim_e = dim_e
        self.dim_h = dim_h
        self.total_dim = dim_e + dim_h
        if self.total_dim % num_heads != 0:
            raise ValueError("(dim_e + dim_h) must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = self.total_dim // num_heads
        self.ball = PoincareBall(c)

        self.q_proj = nn.Linear(self.total_dim, self.total_dim)
        self.k_proj = nn.Linear(self.total_dim, self.total_dim)
        self.v_proj = nn.Linear(self.total_dim, self.total_dim)
        self.out_proj = nn.Linear(self.total_dim, self.total_dim)

        # alpha >= 0 via softplus
        self.alpha_unconstrained = nn.Parameter(torch.tensor(0.0))

    def _split(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return x[..., : self.dim_e], x[..., self.dim_e :]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq, _ = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        qe, qh = self._split(q)
        ke, kh = self._split(k)

        # Euclidean term
        qe_exp = qe.unsqueeze(2)  # [B,T,1,De]
        ke_exp = ke.unsqueeze(1)  # [B,1,T,De]
        d_e = ((qe_exp - ke_exp) ** 2).sum(dim=-1)

        # Hyperbolic term
        qh_h = self.ball.exp0(qh)
        kh_h = self.ball.exp0(kh)
        qh_exp = qh_h.unsqueeze(2).expand(-1, -1, seq, -1)
        kh_exp = kh_h.unsqueeze(1).expand(-1, seq, -1, -1)
        d_h = self.ball.dist(qh_exp, kh_exp)

        alpha = F.softplus(self.alpha_unconstrained)
        dist = d_e + alpha * (d_h**2)
        logits = (-dist / math.sqrt(max(self.total_dim, 1))).clamp(min=-50.0, max=0.0)
        attn = F.softmax(logits, dim=-1)
        out = torch.matmul(attn, v)
        return self.out_proj(out)


class HybridARCLayer(nn.Module):
    def __init__(self, dim_e: int, dim_h: int, num_heads: int = 8, mlp_ratio: int = 4, c: float = 1.0):
        super().__init__()
        self.dim_e = dim_e
        self.dim_h = dim_h
        self.total_dim = dim_e + dim_h
        self.ball = PoincareBall(c)

        self.attn = HybridAttention(dim_e, dim_h, num_heads=num_heads, c=c)
        self.norm1 = nn.LayerNorm(self.total_dim)
        self.norm2 = nn.LayerNorm(self.total_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.total_dim, self.total_dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(self.total_dim * mlp_ratio, self.total_dim),
        )

    def _split(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return x[..., : self.dim_e], x[..., self.dim_e :]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = self.attn(self.norm1(x))
        xe, xh = self._split(x)
        de, dh = self._split(delta)

        xe = xe + de
        xh = self.ball.mobius_add(xh, self.ball.exp0(dh))
        x = torch.cat([xe, xh], dim=-1)

        xe, xh = self._split(x)
        xh_tan = self.ball.log0(xh)
        tan = torch.cat([xe, xh_tan], dim=-1)
        delta2 = self.mlp(self.norm2(tan))
        de2, dh2 = self._split(delta2)

        xe = xe + de2
        xh = self.ball.mobius_add(xh, self.ball.exp0(dh2))
        return torch.cat([xe, xh], dim=-1)


class HybridARC(nn.Module):
    def __init__(self, config: HybridARCConfig):
        super().__init__()
        self.config = config
        self.ball = PoincareBall(config.c)
        self.embed = nn.Embedding(config.vocab_size, config.total_dim)
        self.layers = nn.ModuleList(
            [
                HybridARCLayer(
                    config.dim_e,
                    config.dim_h,
                    num_heads=config.num_heads,
                    mlp_ratio=config.mlp_ratio,
                    c=config.c,
                )
                for _ in range(config.depth)
            ]
        )
        self.norm = nn.LayerNorm(config.total_dim)
        self.head = nn.Linear(config.total_dim, config.vocab_size)

    def _split(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return x[..., : self.config.dim_e], x[..., self.config.dim_e :]

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids)
        xe, xh = self._split(x)
        xh = self.ball.exp0(xh)
        x = torch.cat([xe, xh], dim=-1)

        for layer in self.layers:
            x = layer(x)

        xe, xh = self._split(x)
        xh = self.ball.log0(xh)
        x = torch.cat([xe, xh], dim=-1)
        return self.head(self.norm(x))
