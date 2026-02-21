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

    def _c_like(self, ref: torch.Tensor, c: torch.Tensor | None = None) -> torch.Tensor:
        if c is None:
            return self.c.to(device=ref.device, dtype=ref.dtype)
        return c.to(device=ref.device, dtype=ref.dtype)

    def _sqrt_c(self, ref: torch.Tensor, c: torch.Tensor | None = None) -> torch.Tensor:
        return self._c_like(ref, c).sqrt()

    def project(self, x: torch.Tensor, c: torch.Tensor | None = None, margin: float = 1e-3) -> torch.Tensor:
        sqrt_c = self._sqrt_c(x, c)
        max_norm = (1.0 - margin) / sqrt_c
        norm = torch.norm(x, dim=-1, keepdim=True).clamp_min(EPS)
        scale = torch.clamp(max_norm / norm, max=1.0)
        return x * scale

    def mobius_add(self, x: torch.Tensor, y: torch.Tensor, c: torch.Tensor | None = None) -> torch.Tensor:
        c_val = self._c_like(x, c)
        x2 = (x * x).sum(dim=-1, keepdim=True)
        y2 = (y * y).sum(dim=-1, keepdim=True)
        xy = (x * y).sum(dim=-1, keepdim=True)
        num = (1 + 2 * c_val * xy + c_val * y2) * x + (1 - c_val * x2) * y
        den = 1 + 2 * c_val * xy + (c_val * c_val) * x2 * y2
        out = num / den.clamp_min(EPS)
        return self.project(out, c=c_val)

    def exp0(self, v: torch.Tensor, c: torch.Tensor | None = None) -> torch.Tensor:
        sqrt_c = self._sqrt_c(v, c)
        norm = torch.norm(v, dim=-1, keepdim=True).clamp_min(EPS)
        out = torch.tanh(sqrt_c * norm) * v / (sqrt_c * norm)
        return self.project(out, c=c)

    def log0(self, x: torch.Tensor, c: torch.Tensor | None = None) -> torch.Tensor:
        x = self.project(x, c=c)
        sqrt_c = self._sqrt_c(x, c)
        norm = torch.norm(x, dim=-1, keepdim=True).clamp_min(EPS)
        scaled = (sqrt_c * norm).clamp_max(1 - 1e-5)
        return torch.atanh(scaled) * x / (sqrt_c * norm)

    def dist(self, x: torch.Tensor, y: torch.Tensor, c: torch.Tensor | None = None) -> torch.Tensor:
        sqrt_c = self._sqrt_c(x, c)
        diff = self.mobius_add(-x, y, c=c)
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
        if dim_e % num_heads != 0 or dim_h % num_heads != 0:
            raise ValueError("dim_e and dim_h must each be divisible by num_heads for per-head geometry")
        self.num_heads = num_heads
        self.head_dim = self.total_dim // num_heads
        self.head_dim_e = dim_e // num_heads
        self.head_dim_h = dim_h // num_heads
        self.ball = PoincareBall(c)

        self.q_proj = nn.Linear(self.total_dim, self.total_dim)
        self.k_proj = nn.Linear(self.total_dim, self.total_dim)
        self.v_proj = nn.Linear(self.total_dim, self.total_dim)
        self.out_proj = nn.Linear(self.total_dim, self.total_dim)

        # per-head hybrid weighting alpha_h >= 0
        self.alpha_unconstrained = nn.Parameter(torch.zeros(num_heads))
        # per-head curvature: c_h = softplus(global + head) + eps
        self.global_curvature_unconstrained = nn.Parameter(torch.tensor(c))
        self.curvature_unconstrained = nn.Parameter(torch.zeros(num_heads))

    def curvature_values(self) -> torch.Tensor:
        return F.softplus(self.global_curvature_unconstrained + self.curvature_unconstrained) + 1e-4

    def alpha_values(self) -> torch.Tensor:
        return F.softplus(self.alpha_unconstrained)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq, _ = x.shape
        q = self.q_proj(x).view(bsz, seq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq, self.num_heads, self.head_dim).transpose(1, 2)

        curvatures = self.curvature_values()
        alphas = self.alpha_values()

        outputs = []
        for h in range(self.num_heads):
            qh = q[:, h]  # [B,T,Hd]
            kh = k[:, h]
            vh = v[:, h]

            qe = qh[..., : self.head_dim_e]
            qhyp = qh[..., self.head_dim_e :]
            ke = kh[..., : self.head_dim_e]
            khyp = kh[..., self.head_dim_e :]

            qe_exp = qe.unsqueeze(2)
            ke_exp = ke.unsqueeze(1)
            d_e = ((qe_exp - ke_exp) ** 2).sum(dim=-1)

            c_h = curvatures[h]
            qhyp = self.ball.exp0(qhyp, c=c_h)
            khyp = self.ball.exp0(khyp, c=c_h)

            qhyp_exp = qhyp.unsqueeze(2).expand(-1, -1, seq, -1)
            khyp_exp = khyp.unsqueeze(1).expand(-1, seq, -1, -1)
            d_h = self.ball.dist(qhyp_exp, khyp_exp, c=c_h)

            dist = d_e + alphas[h] * (d_h**2)
            logits = (-dist / math.sqrt(max(self.head_dim, 1))).clamp(min=-50.0, max=0.0)
            attn = F.softmax(logits, dim=-1)
            outputs.append(torch.matmul(attn, vh))

        out = torch.cat(outputs, dim=-1)
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

    def head_curvatures(self) -> torch.Tensor:
        return self.layers[0].attn.curvature_values()

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
