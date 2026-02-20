import pytest

torch = pytest.importorskip("torch")

from mu_arc_llm import MuARCConfig, MuARCLLM, train_step
from arc_llm import ARCLLM, ARCConfig


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
