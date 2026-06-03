import torch

from localagent.model import LocalAgentLM, ModelConfig


def _tiny():
    # Shrunken config to keep the CPU test fast while exercising GQA/RoPE/SwiGLU.
    return ModelConfig(vocab_size=256, d_model=64, n_layers=2, n_heads=4, n_kv_heads=2,
                       ffn_hidden=128, max_seq_len=64)


def test_forward_shapes_and_loss():
    cfg = _tiny()
    model = LocalAgentLM(cfg)
    idx = torch.randint(0, cfg.vocab_size, (2, 16))
    logits, loss = model(idx, targets=idx)
    assert logits.shape == (2, 16, cfg.vocab_size)
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_tied_embeddings_param_count():
    cfg = _tiny()
    model = LocalAgentLM(cfg)
    # tied weights counted once; close to the closed-form estimate
    assert model.num_params() == sum(
        p.numel() for p in {id(p): p for p in model.parameters()}.values()
    )


def test_budget_enforced_at_construction():
    import pytest

    big = ModelConfig(vocab_size=32000, d_model=2048, n_layers=24, n_heads=16, n_kv_heads=4,
                      ffn_hidden=8192)
    with pytest.raises(ValueError):
        LocalAgentLM(big)
