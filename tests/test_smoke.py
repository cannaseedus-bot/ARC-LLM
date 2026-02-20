import pytest

torch = pytest.importorskip("torch")

from arc_llm import MiniARCConfig, MiniARCLLM, train_step


def test_mini_arc_forward_shape():
    cfg = MiniARCConfig(vocab_size=128, dim=32, depth=2, hidden_dim=64, rank=8)
    model = MiniARCLLM(cfg)
    tokens = torch.randint(0, cfg.vocab_size, (3, 7))
    logits = model(tokens)
    assert logits.shape == (3, 7, cfg.vocab_size)


def test_mini_arc_train_step_runs():
    cfg = MiniARCConfig(vocab_size=128, dim=32, depth=1, hidden_dim=64, rank=8)
    model = MiniARCLLM(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    input_ids = torch.randint(0, cfg.vocab_size, (2, 5))
    targets = torch.randint(0, cfg.vocab_size, (2, 5))

    loss = train_step(model, optimizer, input_ids, targets)
    assert isinstance(loss, float)
    assert loss == loss
