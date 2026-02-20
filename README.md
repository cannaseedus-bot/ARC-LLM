# ARC-LLM

This repository now separates the µARC implementation into its own folder:

- `mu_arc_llm/` → canonical **µARC-LLM v0.1** implementation
- `arc_llm/` → compatibility wrappers that re-export the µARC model API

## µARC-LLM (separate folder)

mini, runnable curvature-aware language model prototype with:

- low-rank metric `g = A^T A + λI`
- geodesic attention `softmax(-d_g(x_i, x_j)^2)`
- residual + tangent MLP blocks
- optional curvature regularization

## Minimal usage (new canonical path)

```python
import torch
from mu_arc_llm import MuARCLLM, MuARCConfig, train_step

cfg = MuARCConfig(vocab_size=1000, dim=64, depth=2, hidden_dim=128, rank=16)
model = MuARCLLM(cfg)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

input_ids = torch.randint(0, 1000, (2, 12))
targets = torch.randint(0, 1000, (2, 12))

loss = train_step(model, optimizer, input_ids, targets, curvature_weight=1e-3)
print(loss)
```

## Backward compatibility

Legacy imports from `arc_llm` still work via aliases/re-exports.
