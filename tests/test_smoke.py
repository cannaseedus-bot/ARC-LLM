from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from arc_llm import ARCLLM, ARCConfig
from arc_llm_unified import (
    ARCLLMUnified,
    GeometricExporter,
    RiemannianMetric,
    ShardedModel,
    TensorShard,
    train_step_with_metrics,
)
from hybrid_arc import HybridARC, HybridARCConfig, PoincareBall as HybridBall
from hyperbolic_arc import HyperbolicARC, PoincareBall
from mu_arc_llm import MuARCConfig, MuARCLLM, train_step


def test_mu_arc_forward_shape():
    cfg = MuARCConfig(vocab_size=128, dim=32, depth=2, hidden_dim=64, rank=8)
    model = MuARCLLM(cfg)
    tokens = torch.randint(0, cfg.vocab_size, (3, 7))
    logits = model(tokens)
    assert logits.shape == (3, 7, cfg.vocab_size)


def test_mu_arc_train_step_runs():
    cfg = MuARCConfig(vocab_size=128, dim=32, depth=1, hidden_dim=64, rank=8)
    model = MuARCLLM(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    input_ids = torch.randint(0, cfg.vocab_size, (2, 5))
    targets = torch.randint(0, cfg.vocab_size, (2, 5))

    loss = train_step(model, optimizer, input_ids, targets)
    assert isinstance(loss, float)
    assert loss == loss


def test_arc_compatibility_aliases():
    cfg = ARCConfig(vocab_size=64, dim=16, depth=1, hidden_dim=32, rank=4)
    model = ARCLLM(cfg)
    tokens = torch.randint(0, cfg.vocab_size, (1, 4))
    logits = model(tokens)
    assert logits.shape == (1, 4, cfg.vocab_size)


def test_position_dependent_metric_shape_and_psd():
    metric = RiemannianMetric(dim=8, rank=4, position_dependent=True)
    x = torch.randn(2, 3, 8)
    g = metric(x)
    assert g.shape == (2, 3, 8, 8)
    diag = torch.diagonal(g, dim1=-2, dim2=-1)
    assert torch.all(diag > 0)


def test_unified_shard_roundtrip(tmp_path: Path):
    model = ARCLLMUnified(vocab_size=64, dim=16, depth=1, num_heads=4, rank=4)
    out_dir = tmp_path / "shards"
    model.save_sharded(str(out_dir), shard_size_mb=1)

    manager = ShardedModel(str(out_dir), max_cache=2)
    loaded = manager.get_tensor("embed.weight")
    assert loaded.shape == model.embed.weight.shape


def test_tensor_shard_header_and_index_roundtrip(tmp_path: Path):
    shard_path = tmp_path / "single.tsr"
    writer = TensorShard(str(shard_path), mode="w", shard_id=3)
    tensor = torch.randn(4, 5)
    writer.write_tensor("w", tensor)
    writer.close()

    reader = TensorShard(str(shard_path), mode="r")
    assert reader.header.shard_id == 3
    assert reader.header.num_tensors == 1
    reloaded = reader.get_tensor("w")
    assert reloaded.shape == tensor.shape
    reader.close()


def test_unified_train_step_metrics_keys():
    model = ARCLLMUnified(vocab_size=64, dim=16, depth=1, num_heads=4, rank=4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    input_ids = torch.randint(0, 64, (2, 6))
    targets = torch.randint(0, 64, (2, 6))

    m = train_step_with_metrics(model, optimizer, input_ids, targets)
    assert {"total_loss", "lm_loss", "curvature_loss", "grad_norm"}.issubset(m.keys())


def test_unified_exporters_generate_strings():
    model = ARCLLMUnified(vocab_size=64, dim=16, depth=1, num_heads=4, rank=4)
    exporter = GeometricExporter(model)
    svg = exporter.export_metric_as_svg(0, 0)
    xml = exporter.export_full_geometry_xml()
    assert "<svg" in svg
    assert "<arc-llm" in xml


def test_poincare_ball_ops_shapes():
    ball = PoincareBall(c=1.0)
    x = torch.randn(2, 5, 8) * 0.1
    y = torch.randn(2, 5, 8) * 0.1
    xh = ball.exp_map_zero(x)
    yh = ball.exp_map_zero(y)
    z = ball.mobius_add(xh, yh)
    d = ball.distance(xh, yh)

    assert z.shape == x.shape
    assert d.shape == (2, 5)
    assert torch.all(torch.isfinite(d))


def test_hyperbolic_arc_forward_shape():
    model = HyperbolicARC(vocab_size=256, dim=32, depth=2, num_heads=4, c=1.0)
    tokens = torch.randint(0, 256, (2, 6))
    logits = model(tokens)
    assert logits.shape == (2, 6, 256)


def test_hybrid_ball_shapes():
    ball = HybridBall(c=1.0)
    x = torch.randn(2, 4, 6) * 0.1
    y = torch.randn(2, 4, 6) * 0.1
    xh = ball.exp0(x)
    yh = ball.exp0(y)
    z = ball.mobius_add(xh, yh)
    d = ball.dist(xh, yh)
    assert z.shape == x.shape
    assert d.shape == (2, 4)


def test_hybrid_arc_forward_shape():
    cfg = HybridARCConfig(vocab_size=256, dim_e=16, dim_h=16, depth=2, num_heads=4, c=1.0)
    model = HybridARC(cfg)
    tokens = torch.randint(0, cfg.vocab_size, (2, 6))
    logits = model(tokens)
    assert logits.shape == (2, 6, cfg.vocab_size)
