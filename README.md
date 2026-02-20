# ARC-LLM

This repository contains two tracks:

- `mu_arc_llm/`: canonical µARC-LLM micro-transformer package.
- `arc_llm/`: compatibility wrappers for previous import paths.

## Unified research-core prototype

`arc_llm_unified.py` provides an executable unified stack centered on real geometry:

- tensor sharding with explicit header/index finalization
- per-head low-rank Riemannian metrics (`g = A^T A + λI`)
- optional position-dependent metric modulation (with corrected broadcasting)
- geodesic attention (`softmax(-d_g^2)`) with stability scaling/clamping
- tangent-projected residual transport
- curvature regularization on global base metrics
- training helper with gradient-norm logging
- SVG/XML metric export

`arc_runner.ps1` is intentionally scoped to orchestration/inspection only (manifest + shard info). Inference stays in Python so model math is not duplicated in shell scripts.

## Quick example

```python
import torch
from arc_llm_unified import ARCLLMUnified, GeometricExporter, train_step_with_metrics

model = ARCLLMUnified(vocab_size=1000, dim=64, depth=2, num_heads=4, rank=8)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

input_ids = torch.randint(0, 1000, (2, 12))
targets = torch.randint(0, 1000, (2, 12))
metrics = train_step_with_metrics(model, optimizer, input_ids, targets)

model.save_sharded("arc_model_shards", shard_size_mb=1)
exporter = GeometricExporter(model)
svg = exporter.export_metric_as_svg(0, 0)
xml = exporter.export_full_geometry_xml()
```

## Backward compatibility

Legacy imports from `arc_llm` continue to work via alias/re-export to `mu_arc_llm`.
