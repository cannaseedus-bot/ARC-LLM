# ARC-LLM

This repository contains four tracks:

- `mu_arc_llm/`: canonical µARC-LLM micro-transformer package.
- `arc_llm/`: compatibility wrappers for previous import paths.
- `arc_llm_unified.py`: unified metric-geometric ARC prototype.
- `hyperbolic_arc.py` + `hybrid_arc.py`: constant-curvature and product-manifold geometric variants.

## Hybrid ARC (Euclidean × Hyperbolic)

`hybrid_arc.py` implements a product manifold:

- `R^{d_e}` Euclidean subspace
- `B_c^{d_h}` hyperbolic Poincaré-ball subspace

with hybrid distance attention:

- Euclidean distance term: `||q_e - k_e||^2`
- Hyperbolic distance term: `alpha * d_hyp(q_h, k_h)^2`
- `alpha` is learnable and constrained non-negative via `softplus`

Residual updates are split by geometry:

- Euclidean: additive residual
- Hyperbolic: Möbius-add residual using exp/log maps

## Hyperbolic ARC

`hyperbolic_arc.py` provides a pure constant-negative-curvature variant:

- Möbius addition
- exp/log maps at origin
- hyperbolic distance attention

## Unified metric ARC

`arc_llm_unified.py` provides:

- tensor sharding with explicit header/index finalization
- per-head low-rank Riemannian metrics (`g = A^T A + λI`)
- optional position-dependent metric modulation (correct broadcasting)
- geodesic attention with stability scaling/clamping
- tangent-projected residual transport
- training helper with gradient-norm logging

## Quick examples

```python
import torch
from arc_llm_unified import ARCLLMUnified, train_step_with_metrics
from hyperbolic_arc import HyperbolicARC
from hybrid_arc import HybridARC, HybridARCConfig

model = ARCLLMUnified(vocab_size=1000, dim=64, depth=2, num_heads=4, rank=8)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
metrics = train_step_with_metrics(
    model,
    optimizer,
    torch.randint(0, 1000, (2, 12)),
    torch.randint(0, 1000, (2, 12)),
)

h_model = HyperbolicARC(vocab_size=1000, dim=64, depth=2, num_heads=4, c=1.0)
h_logits = h_model(torch.randint(0, 1000, (2, 12)))

hy_model = HybridARC(HybridARCConfig(vocab_size=1000, dim_e=32, dim_h=32, depth=2, num_heads=4, c=1.0))
hy_logits = hy_model(torch.randint(0, 1000, (2, 12)))
```

## Backward compatibility

Legacy imports from `arc_llm` continue to work via alias/re-export to `mu_arc_llm`.
