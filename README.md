# ARC-LLM

This repository contains two tracks:

- `mu_arc_llm/`: canonical µARC-LLM micro-transformer package.
- `arc_llm/`: compatibility wrappers for previous import paths.

## Unified prototype

A single-file unified architecture is also provided in `arc_llm_unified.py` with:

- tensor sharding (`TensorShard`, `ShardedModel`)
- geometric attention (`RiemannianMetric`, `GeodesicAttention`)
- complete model (`ARCLLMUnified`)
- geometry exporters (`GeometricExporter`)
- PowerShell runner generation (`generate_powershell_runner`)

A basic PowerShell entrypoint is included at `arc_runner.ps1`.

## Quick example

```python
import torch
from arc_llm_unified import ARCLLMUnified, GeometricExporter

model = ARCLLMUnified(vocab_size=1000, dim=64, depth=2, num_heads=4, rank=8)
input_ids = torch.randint(0, 1000, (2, 12))
logits = model(input_ids)

model.save_sharded("arc_model_shards", shard_size_mb=1)
exporter = GeometricExporter(model)
svg = exporter.export_metric_as_svg(0, 0)
xml = exporter.export_full_geometry_xml()
```

## Backward compatibility

Legacy imports from `arc_llm` continue to work via alias/re-export to `mu_arc_llm`.
