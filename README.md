# ARC-LLM

miniARC-LLM v0.1 is a compact, runnable curvature-aware language model prototype.

## Design goals

- mathematically coherent
- trainable on a laptop
- runnable in pure PyTorch
- curvature-aware without infeasible `O(d^3)` tensors

## Architecture (miniARC-LLM v0.1)

- **Low-rank metric**: `g = A^T A + λI`
- **Geodesic attention**: `softmax(-d_g(x_i, x_j)^2)` where `d_g` is metric distance
- **Residual + tangent MLP**: transformer-like block with geometric similarity kernel
- **Curvature regularization**: `||g - I||_F` penalty to prevent metric explosion

## Minimal usage

```python
import torch
from arc_llm import MiniARCLLM, MiniARCConfig, train_step

cfg = MiniARCConfig(vocab_size=1000, dim=64, depth=2, hidden_dim=128, rank=16)
model = MiniARCLLM(cfg)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

input_ids = torch.randint(0, 1000, (2, 12))
targets = torch.randint(0, 1000, (2, 12))

loss = train_step(model, optimizer, input_ids, targets, curvature_weight=1e-3)
print(loss)
```

## Notes

This implementation intentionally avoids explicit Christoffel symbols, explicit geodesic arc enumeration, and full Ricci tensor computation. It keeps the core geometric idea while remaining practical.
