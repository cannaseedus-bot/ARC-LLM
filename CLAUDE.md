# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ARC-LLM is a research project implementing geometric deep learning models for Abstract Reasoning Corpus (ARC) tasks. The codebase contains four distinct but related implementations, each exploring different manifold geometries for transformer-based reasoning:

- **µARC-LLM** (`mu_arc_llm/`): Canonical micro-transformer with low-rank Riemannian metric-based geodesic attention
- **ARC-LLM Unified** (`arc_llm_unified.py`): Research prototype with tensor sharding, per-head metrics, exporters for geometry inspection
- **Hyperbolic ARC** (`hyperbolic_arc.py`): Constant-negative-curvature variant using Poincaré ball geometry
- **Hybrid ARC** (`hybrid_arc.py`): Product manifold combining Euclidean (R^{d_e}) and hyperbolic (B_c^{d_h}) subspaces
- **Legacy compatibility** (`arc_llm/`): Wrapper maintaining backward compatibility with previous import paths

## Architecture & Key Concepts

### Geodesic Attention (Core Innovation)

All models replace standard dot-product attention with metric-induced geodesic distance attention:

```
distance²(x_i, x_j) = (x_i - x_j)ᵀ g (x_i - x_j)
```

where `g` is a learned positive-definite metric. This allows the model to learn problem-specific geometry for comparing tokens.

**Implementation details**:
- µARC-LLM uses a fixed metric `g = AᵀA + λI` (low-rank via outer product)
- Unified ARC supports position-dependent metrics: `g(pos)` varies by sequence position
- Hyperbolic and Hybrid variants embed this in non-Euclidean spaces

### Tensor Sharding (Unified ARC)

Large models can be decomposed into shards with binary format:

```
[256-byte header][tensor payloads...][index_len (4 bytes)][JSON index]
```

- Header contains: version, shard_id, num_tensors, index_offset
- Enables saving/loading models that don't fit in memory
- `ShardedModel` manager provides LRU caching for tensor access

### Hyperbolic Geometry (Hyperbolic & Hybrid ARC)

Poincaré ball representation for hierarchical modeling:

- **exp₀**: Maps tangent vectors at origin to ball surface
- **mobius_add**: Gyrovector space addition (non-Euclidean)
- **distance**: Hyperbolic metric on ball

Hybrid ARC combines this with Euclidean residuals using exp/log maps for smooth gradient flow.

## Development Commands

### Installation

Dependencies are minimal but essential. Install via:

```bash
pip install torch pytest
```

For development with strict type checking (optional):
```bash
pip install mypy
```

### Running Tests

```bash
# All tests
python -m pytest tests/

# Single test file
python -m pytest tests/test_smoke.py

# Single test function
python -m pytest tests/test_smoke.py::test_mu_arc_forward_shape

# Verbose output
python -m pytest tests/ -v

# Show print statements
python -m pytest tests/ -s

# Run with coverage (requires pytest-cov)
python -m pytest tests/ --cov=. --cov-report=html
```

### Quick Verification

```bash
# Run a minimal forward pass
python -c "
import torch
from mu_arc_llm import MuARCConfig, MuARCLLM
cfg = MuARCConfig(vocab_size=128, dim=32, depth=2)
model = MuARCLLM(cfg)
logits = model(torch.randint(0, 128, (2, 8)))
print(f'Output shape: {logits.shape}')
"
```

## Import Patterns & Module Structure

### Canonical Imports

```python
# µARC-LLM (canonical)
from mu_arc_llm import MuARCConfig, MuARCLLM, train_step

# Unified metric-geometric variant
from arc_llm_unified import ARCLLMUnified, train_step_with_metrics, RiemannianMetric

# Hyperbolic variant
from hyperbolic_arc import HyperbolicARC, PoincareBall

# Hybrid variant
from hybrid_arc import HybridARC, HybridARCConfig

# Legacy compatibility (backward compatible)
from arc_llm import ARCLLM, ARCConfig
```

### Module Layout

```
mu_arc_llm/
  ├── __init__.py           # Exports MuARCLLM, MuARCConfig, train_step
  ├── model.py              # GeodesicAttention, MuArcLayer, MuARCLLM class
  ├── core/
  │   └── manifold.py       # LowRankMetric, curvature_regularizer

arc_llm/
  ├── __init__.py           # Legacy compatibility aliases
  ├── model.py              # ARCLLM, ARCConfig (re-exports)
  ├── core/
  │   └── manifold.py       # Copies of manifold utilities

arc_llm_unified.py         # All-in-one: ShardedModel, TensorShard, exporters
hyperbolic_arc.py          # HyperbolicARC, PoincareBall
hybrid_arc.py              # HybridARC, HybridARCConfig, Poincare ball
```

## Key Implementation Details

### GeodesicAttention

Located: `mu_arc_llm/model.py:21-40`

Replaces standard softmax attention with learned metric-induced distances. The metric is position-invariant in µARC-LLM but position-dependent in unified ARC (gated by learned scalar per position).

### LowRankMetric & RiemannianMetric

Located: `mu_arc_llm/core/manifold.py` (and mirrored in `arc_llm/core/manifold.py`)

- **LowRankMetric**: `g = AᵀA + λI` where A is `(dim, rank)` matrix. Ensures positive-definiteness while reducing parameters.
- **RiemannianMetric** (unified): Extends with position-dependent gates (optional).

### Training Loop

All variants follow similar pattern:

```python
logits = model(input_ids)
lm_loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))
curvature_loss = model.curvature_loss()  # if applicable
total_loss = lm_loss + weight * curvature_loss
loss.backward()
optimizer.step()
```

`curvature_loss` penalizes degenerate metrics (near-singular A) to maintain well-conditioned geometry.

## Testing Strategy

Test suite in `tests/test_smoke.py` covers:

1. **Forward pass shapes** (all variants): Verify output dimensions match input
2. **Training steps**: Ensure backward pass runs without NaNs
3. **Geometric operations**: Poincaré ball exp/log/mobius operations
4. **Tensor sharding**: Write/read roundtrips with headers/indexes
5. **Exporters**: SVG/XML generation for geometry visualization
6. **Curvatures**: Per-head positive curvatures in hybrid ARC

Pattern: Import from canonical module, construct minimal config, run operation, check invariants (shape, finiteness, positivity where required).

## Important Notes

### Dependencies

- **PyTorch**: Core dependency, no version constraint in code (tested with recent stable)
- **No external geometry libraries**: All manifold operations implemented from first principles
- **Optional**: pytest for testing, mypy for type checking

### Backward Compatibility

The `arc_llm` package is a **compatibility wrapper** that re-exports from `mu_arc_llm`. Legacy code using `from arc_llm import ARCLLM` continues to work without changes.

### Performance Considerations

- Geodesic attention is O(seq_len²) like standard attention (no approximations yet)
- Low-rank metric parameterization uses `(dim × rank)` parameters per attention head, with default rank = dim/4
- Sharding is beneficial only for very large models (>1GB); overhead not worth it for research prototypes
- Hyperbolic operations (exp/log/mobius) have small computational cost but ensure numerical stability is critical

### Curvature Regularization

All models support per-layer curvature loss to prevent metric collapse:

```python
loss = lm_loss + α * model.curvature_loss(weight=β)
```

Typical values: α=1, β=1e-3. This penalizes when low-rank factor A becomes near-singular.
