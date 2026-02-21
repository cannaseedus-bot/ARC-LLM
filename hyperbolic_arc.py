"""Hyperbolic ARC prototype using the Poincare ball model.

This module provides a research-focused constant-negative-curvature ARC variant.
"""

from __future__ import annotations

import math

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

    def _sqrt_c(self, ref: torch.Tensor) -> torch.Tensor:
        return self.c.to(device=ref.device, dtype=ref.dtype).sqrt()

    def project(self, x: torch.Tensor) -> torch.Tensor:
        sqrt_c = self._sqrt_c(x)
        max_norm = (1.0 - 1e-3) / sqrt_c
        norm = torch.norm(x, dim=-1, keepdim=True).clamp_min(EPS)
        scale = torch.clamp(max_norm / norm, max=1.0)
        return x * scale

    def mobius_add(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        c = self.c.to(device=x.device, dtype=x.dtype)
        x2 = (x * x).sum(dim=-1, keepdim=True)
        y2 = (y * y).sum(dim=-1, keepdim=True)
        xy = (x * y).sum(dim=-1, keepdim=True)
        num = (1 + 2 * c * xy + c * y2) * x + (1 - c * x2) * y
        den = 1 + 2 * c * xy + (c**2) * x2 * y2
        out = num / den.clamp_min(EPS)
        return self.project(out)

    def exp_map_zero(self, v: torch.Tensor) -> torch.Tensor:
        sqrt_c = self._sqrt_c(v)
        v_norm = torch.norm(v, dim=-1, keepdim=True).clamp_min(EPS)
        factor = torch.tanh(sqrt_c * v_norm) / (sqrt_c * v_norm)
        return self.project(factor * v)

    def log_map_zero(self, x: torch.Tensor) -> torch.Tensor:
        x = self.project(x)
        sqrt_c = self._sqrt_c(x)
        x_norm = torch.norm(x, dim=-1, keepdim=True).clamp_min(EPS)
        scaled = (sqrt_c * x_norm).clamp_max(1 - 1e-5)
        return torch.atanh(scaled) * x / (sqrt_c * x_norm)

    def distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        sqrt_c = self._sqrt_c(x)
        diff = self.mobius_add(-x, y)
        norm = torch.norm(diff, dim=-1).clamp_max((1 - 1e-5) / sqrt_c)
        return (2.0 / sqrt_c) * torch.atanh((sqrt_c * norm).clamp_max(1 - 1e-5))


class HyperbolicAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, c: float = 1.0):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError("dim must be divisible by num_heads")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.ball = PoincareBall(c)

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq, _ = x.shape
        q = self.q_proj(x).view(bsz, seq, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(bsz, seq, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(bsz, seq, self.num_heads, self.head_dim).transpose(1, 2)

        outs = []
        for h in range(self.num_heads):
            qh = self.ball.exp_map_zero(q[:, h])
            kh = self.ball.exp_map_zero(k[:, h])

            q_exp = qh.unsqueeze(2).expand(-1, -1, seq, -1)
            k_exp = kh.unsqueeze(1).expand(-1, seq, -1, -1)
            dist = self.ball.distance(q_exp, k_exp)
            logits = (-dist / math.sqrt(self.head_dim)).clamp(min=-50.0, max=0.0)
            attn = F.softmax(logits, dim=-1)
            outs.append(torch.matmul(attn, v[:, h]))

        out = torch.cat(outs, dim=-1)
        return self.out_proj(out)


class HyperbolicARCLayer(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, mlp_ratio: int = 4, c: float = 1.0):
        super().__init__()
        self.ball = PoincareBall(c)
        self.attn = HyperbolicAttention(dim, num_heads=num_heads, c=c)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(dim * mlp_ratio, dim),
        )

    def forward(self, x_h: torch.Tensor) -> torch.Tensor:
        x_t = self.ball.log_map_zero(x_h)
        delta_t = self.attn(self.norm1(x_t))
        delta_h = self.ball.exp_map_zero(delta_t)
        x_h = self.ball.mobius_add(x_h, delta_h)

        x_t = self.ball.log_map_zero(x_h)
        mlp_t = self.mlp(self.norm2(x_t))
        mlp_h = self.ball.exp_map_zero(mlp_t)
        x_h = self.ball.mobius_add(x_h, mlp_h)
        return x_h


class HyperbolicARC(nn.Module):
    def __init__(self, vocab_size: int, dim: int = 256, depth: int = 4, num_heads: int = 8, c: float = 1.0):
        super().__init__()
        self.ball = PoincareBall(c)
        self.embed = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([HyperbolicARCLayer(dim, num_heads=num_heads, c=c) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids)
        x_h = self.ball.exp_map_zero(x)
        for layer in self.layers:
            x_h = layer(x_h)
        x_t = self.ball.log_map_zero(x_h)
        return self.head(self.norm(x_t))
