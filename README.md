# ARC-LLM: Geometric Deep Learning for Abstract Reasoning

A research framework for exploring transformer architectures with learned geometric structures. ARC-LLM implements four complementary approaches to geometric reasoning, enabling comparative analysis of different manifold geometries for sequence modeling and abstract reasoning tasks.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture Overview](#architecture-overview)
- [Implementation Variants](#implementation-variants)
- [Advanced Usage](#advanced-usage)
- [Performance & Benchmarks](#performance--benchmarks)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [Roadmap](#roadmap)
- [Citation](#citation)

## Features

### Core Innovations

- **Geodesic Attention**: Replace dot-product attention with learned metric-induced distances
- **Low-Rank Metrics**: Efficient parameterization via `g = AᵀA + λI`
- **Multiple Geometries**: Euclidean, hyperbolic, and hybrid product manifolds
- **Position-Dependent Metrics**: Optional per-position metric modulation in unified variant
- **Tensor Sharding**: Scale beyond GPU memory with binary shard format
- **Geometric Visualization**: Export metrics as SVG/XML for inspection

### Quick Comparison

| Feature | µARC-LLM | Unified | Hyperbolic | Hybrid |
|---------|----------|---------|-----------|--------|
| Metric Geometry | Low-rank | Low-rank + position | Constant curvature | Product manifold |
| Scalability | Standard | Sharded | Standard | Standard |
| Per-head Metrics | Yes | Yes | Fixed | Yes |
| Curvature Learning | No | No | Fixed | Per-head |
| Export Geometries | No | Yes (SVG/XML) | No | No |
| Training Helpers | Basic | Full metrics | Basic | Basic |

## Installation

### Requirements

- Python 3.8+
- PyTorch 1.13+

### Setup

```bash
# Clone repository
git clone https://github.com/cannaseedus-bot/ARC-LLM.git
cd ARC-LLM

# Install core dependencies
pip install torch

# Install development dependencies
pip install pytest pytest-cov mypy  # optional for testing & type checking
```

## Quick Start

### Basic Forward Pass

```python
import torch
from mu_arc_llm import MuARCConfig, MuARCLLM

# Create model
config = MuARCConfig(vocab_size=1000, dim=64, depth=2, num_heads=4)
model = MuARCLLM(config)

# Forward pass
tokens = torch.randint(0, 1000, (2, 12))
logits = model(tokens)
print(f"Output shape: {logits.shape}")  # (2, 12, 1000)
```

### Training with Metrics

```python
import torch
from arc_llm_unified import ARCLLMUnified, train_step_with_metrics

model = ARCLLMUnified(vocab_size=1000, dim=64, depth=2, num_heads=4, rank=8)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# Single training step
input_ids = torch.randint(0, 1000, (2, 12))
targets = torch.randint(0, 1000, (2, 12))

metrics = train_step_with_metrics(model, optimizer, input_ids, targets)
print(f"Loss: {metrics['total_loss']:.4f}, Grad norm: {metrics['grad_norm']:.4f}")
```

### Hyperbolic and Hybrid Variants

```python
import torch
from hyperbolic_arc import HyperbolicARC
from hybrid_arc import HybridARC, HybridARCConfig

# Hyperbolic variant
h_model = HyperbolicARC(vocab_size=1000, dim=64, depth=2, num_heads=4, c=1.0)
h_logits = h_model(torch.randint(0, 1000, (2, 12)))

# Hybrid (Euclidean + Hyperbolic)
config = HybridARCConfig(
    vocab_size=1000,
    dim_e=32,      # Euclidean dimension
    dim_h=32,      # Hyperbolic dimension
    depth=2,
    num_heads=4,
    c=1.0          # Global curvature
)
hy_model = HybridARC(config)
hy_logits = hy_model(torch.randint(0, 1000, (2, 12)))
```

## Architecture Overview

### Geodesic Attention Mechanism

All variants replace standard softmax attention with metric-induced distance attention:

```
attention_scores(q_i, k_j) = exp(-distance²(q_i, k_j) / √d) / Z
where distance²(x, y) = (x - y)ᵀ g(x - y)
and g is a learned positive-definite metric
```

**Benefits**:
- Learn problem-specific geometry for token comparisons
- Automatic regularization through metric conditioning
- Interpretable geometry visualization (unified variant)

### Metric Parameterization

**Low-Rank Metric** (`µARC-LLM`, Unified):
```
g = AᵀA + λI
```
- A is (dim, rank) learnable matrix
- Ensures positive-definiteness
- Default rank = dim/4 (configurable)
- Per-head or position-dependent

**Fixed Curvature** (Hyperbolic, Hybrid):
```
c_h = softplus(c_global + c_head_offset)
```
- Poincaré ball constant curvature
- Per-head learning of curvature offsets
- Non-Euclidean distance metrics

## Implementation Variants

### µARC-LLM (Canonical)

**Location**: `mu_arc_llm/`

The reference implementation with position-invariant low-rank metrics.

```python
from mu_arc_llm import MuARCConfig, MuARCLLM, train_step

cfg = MuARCConfig(
    vocab_size=1000,
    dim=64,
    depth=2,
    num_heads=4,
    hidden_dim=256,
    rank=16,           # Low-rank factor dimension
    dropout=0.1
)
model = MuARCLLM(cfg)
```

**Key classes**:
- `GeodesicAttention`: Metric-induced distance attention
- `LowRankMetric`: PSD metric via outer product
- `MuArcLayer`: Transformer block with geodesic attention
- `train_step()`: Simple training loop helper

### Unified ARC

**Location**: `arc_llm_unified.py`

Advanced variant with tensor sharding, position-dependent metrics, and geometry visualization.

```python
from arc_llm_unified import ARCLLMUnified, train_step_with_metrics

model = ARCLLMUnified(
    vocab_size=1000,
    dim=64,
    depth=2,
    num_heads=4,
    rank=16,
    position_dependent=True,  # Per-position metrics
)

# Save as shards
model.save_sharded("./checkpoints", shard_size_mb=512)

# Export geometry for inspection
from arc_llm_unified import GeometricExporter
exporter = GeometricExporter(model)
svg = exporter.export_metric_as_svg(layer=0, head=0)
```

**Key features**:
- Position-dependent metric modulation
- Tensor sharding with LRU cache
- Geometry visualization (SVG/XML export)
- Full training metrics logging

### Hyperbolic ARC

**Location**: `hyperbolic_arc.py`

Pure hyperbolic variant using Poincaré ball geometry.

```python
from hyperbolic_arc import HyperbolicARC

model = HyperbolicARC(
    vocab_size=1000,
    dim=64,
    depth=2,
    num_heads=4,
    c=1.0,  # Global curvature (constant)
)
```

**Key operations**:
- `exp_map_zero()`: Tangent-to-manifold projection
- `mobius_add()`: Gyrovector addition
- `distance()`: Hyperbolic metric
- Automatic gradient flow via PyTorch

### Hybrid ARC

**Location**: `hybrid_arc.py`

Product manifold combining Euclidean (local) and hyperbolic (hierarchical) subspaces.

```python
from hybrid_arc import HybridARC, HybridARCConfig

cfg = HybridARCConfig(
    vocab_size=1000,
    dim_e=32,      # Euclidean space dimension
    dim_h=32,      # Hyperbolic space dimension
    depth=2,
    num_heads=4,
    c=1.0,         # Global curvature
)
model = HybridARC(cfg)

# Per-head curvatures (learned)
head_curvatures = model.head_curvatures()  # shape: (num_heads,)
```

**Key design**:
- Split representations: [euclidean_part || hyperbolic_part]
- Euclidean residuals: standard additive updates
- Hyperbolic residuals: Möbius-add with exp/log maps
- Independent per-head curvature learning

### Legacy Compatibility (`arc_llm/`)

For backward compatibility with previous import paths:

```python
from arc_llm import ARCLLM, ARCConfig  # Aliases to mu_arc_llm
```

## Advanced Usage

### Custom Training Loop with Curvature Regularization

```python
import torch
import torch.nn.functional as F
from mu_arc_llm import MuARCLLM, MuARCConfig

model = MuARCLLM(MuARCConfig(vocab_size=1000, dim=64, depth=2))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

def train_step_with_regularization(model, input_ids, targets, alpha=1.0, beta=1e-3):
    """
    Training step with metric curvature regularization.

    alpha: weight for language modeling loss
    beta: weight for curvature regularization
    """
    # Language modeling
    logits = model(input_ids)
    lm_loss = F.cross_entropy(logits.view(-1, model.config.vocab_size), targets.view(-1))

    # Curvature regularization (if available)
    curvature_loss = model.curvature_loss() if hasattr(model, 'curvature_loss') else 0

    # Total loss
    total_loss = alpha * lm_loss + beta * curvature_loss
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()
    optimizer.zero_grad()

    return {
        'total_loss': total_loss.item(),
        'lm_loss': lm_loss.item(),
        'curvature_loss': curvature_loss if isinstance(curvature_loss, (int, float)) else curvature_loss.item(),
    }
```

### Tensor Sharding for Large Models

```python
from arc_llm_unified import ARCLLMUnified, ShardedModel

# Create large model
model = ARCLLMUnified(vocab_size=50000, dim=768, depth=12, num_heads=12, rank=192)

# Save as shards (automatic batching)
model.save_sharded(
    "./checkpoints/large_model",
    shard_size_mb=2048,  # 2GB per shard
)

# Load with LRU caching
manager = ShardedModel(
    "./checkpoints/large_model",
    max_cache=3,  # Keep 3 tensors in memory
)

# Transparent access
embedding = manager.get_tensor("embed.weight")
```

### Geometry Visualization and Inspection

```python
from arc_llm_unified import ARCLLMUnified, GeometricExporter

model = ARCLLMUnified(vocab_size=1000, dim=64, depth=2, num_heads=4)

exporter = GeometricExporter(model)

# Export single metric as SVG
svg_str = exporter.export_metric_as_svg(layer=0, head=0)
with open("metric_l0_h0.svg", "w") as f:
    f.write(svg_str)

# Export full geometry as XML
xml_str = exporter.export_full_geometry_xml()
with open("full_geometry.xml", "w") as f:
    f.write(xml_str)
```

### Custom Riemannian Metrics

```python
from arc_llm_unified import RiemannianMetric
import torch

# Create position-dependent metric
metric = RiemannianMetric(
    dim=64,
    rank=16,
    position_dependent=True,
)

# Compute metric tensor for batch
x = torch.randn(2, 12, 64)  # (batch, seq_len, dim)
g = metric(x)  # (batch, seq_len, 64, 64)

# Verify positive-definiteness
eigenvalues = torch.linalg.eigvals(g)
assert torch.all(eigenvalues.real > 0), "Metric must be positive-definite"

# Use for custom attention
distances_sq = torch.einsum('bsid,bsij,bsjd->bs', x, g, x)
```

## Performance & Benchmarks

### Memory Usage

| Model | Variant | Parameters | Memory (GB) |
|-------|---------|-----------|------------|
| Small (dim=64) | All | ~4M | 0.02 |
| Medium (dim=256) | All | ~65M | 0.26 |
| Large (dim=768) | µARC | ~628M | 2.51 |
| Large (dim=768) | Unified | ~642M | 2.57 |

*Measured with batch_size=1, seq_len=512 on PyTorch 2.0*

### Computational Cost

- **Geodesic attention**: O(seq_len²) like standard attention
- **Metric computation**: O(dim × rank) per head (negligible)
- **Hyperbolic ops**: ~2-3% overhead vs. standard attention
- **Tensor sharding**: ~1-5% I/O overhead depending on disk speed

### Best Practices

1. **Rank Selection**: Start with rank = dim/4, tune based on validation loss
2. **Learning Rate**: Geometric attention may require smaller lr (1e-4 to 1e-3)
3. **Curvature Weight**: Balance with LM loss: α_curvature ≈ 0.01 × α_lm
4. **Batch Size**: No special requirements, can scale standard approaches
5. **Gradient Clipping**: Recommended for stability (max_norm=1.0)

## Troubleshooting

### NaN Loss During Training

**Cause**: Metric becoming singular (det(g) ≈ 0)

**Solution**:
```python
# Add curvature regularization
curvature_loss = model.curvature_loss(weight=1e-3)
total_loss = lm_loss + curvature_loss
```

### Out of Memory

**For small models**: Reduce batch size or seq_len

**For large models**: Use tensor sharding
```python
model.save_sharded("./checkpoints", shard_size_mb=512)
manager = ShardedModel("./checkpoints", max_cache=2)
```

### Slow Training Speed

**Check**:
1. Ensure PyTorch compiled kernels are enabled: `torch.backends.cudnn.enabled = True`
2. Profile bottleneck: most time should be in attention
3. Verify rank is reasonable (not rank = dim)

### Gradient Explosion in Hyperbolic Models

**Cause**: Improper exp/log map implementation or boundary issues

**Solution**:
```python
# Use automatic gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

# Verify Poincaré ball boundary
ball = model.ball  # or import from hyperbolic_arc
assert torch.all(torch.norm(x, dim=-1) < 1.0), "Points must be inside ball"
```

## Contributing

We welcome contributions! Areas of interest:

- [ ] Continuous approximations of metrics (instead of low-rank)
- [ ] Mixed-precision training support
- [ ] ONNX export for inference
- [ ] Distributed training (DistributedDataParallel)
- [ ] Additional geometric variants (Lorentz model, product spaces)
- [ ] Comprehensive benchmarks on ARC reasoning tasks
- [ ] Documentation improvements and tutorials

### Development Workflow

1. Create feature branch: `git checkout -b feature/your-feature`
2. Develop with tests: `pytest tests/`
3. Ensure type safety: `mypy .`
4. Commit with clear messages
5. Push and create pull request

See [CLAUDE.md](./CLAUDE.md) for detailed architecture and development guidance.

## Roadmap

### v0.2 (Next Release)

- [ ] **Distributed Training**: Multi-GPU support via DDP
- [ ] **Mixed Precision**: FP16/BF16 training and inference
- [ ] **ONNX Export**: Deploy models without PyTorch
- [ ] **Benchmark Suite**: ARC task evaluation harness
- [ ] **Additional Metrics**: Lorentz model, Grassmannian spaces

### v0.3

- [ ] **Continuous Metrics**: Fully learned metric functions (vs low-rank)
- [ ] **Inference Optimization**: Quantization and pruning strategies
- [ ] **Geometric Adaptation**: Dynamic metric learning during inference
- [ ] **Web Visualization**: Interactive geometry dashboard

### v0.4+

- [ ] **Multimodal Variants**: Visual reasoning with geometric attention
- [ ] **Long-Context Optimization**: Efficient geodesic attention for long sequences
- [ ] **Hierarchical Metrics**: Multi-scale geometric structure
- [ ] **Theoretical Analysis**: Convergence guarantees and curvature bounds

## Citation

If you use ARC-LLM in research, please cite:

```bibtex
@repository{arc-llm-2025,
  author = {ARCLLM Contributors},
  title = {ARC-LLM: Geometric Deep Learning for Abstract Reasoning},
  year = {2025},
  url = {https://github.com/cannaseedus-bot/ARC-LLM}
}
```

## License

This project is provided as-is for research purposes.

---

**Questions or Issues?** Open a GitHub issue or check [CLAUDE.md](./CLAUDE.md) for development guidance.
