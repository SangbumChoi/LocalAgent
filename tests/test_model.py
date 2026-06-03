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


def test_factorized_embeddings_and_recurrence_forward():
    # Mirrors the ultra-tiny structure: byte vocab + factorized embed + shared-weight loops.
    cfg = ModelConfig(vocab_size=256, d_model=96, embed_dim=32, n_layers=2, n_loops=4,
                      n_heads=6, n_kv_heads=2, ffn_hidden=128, max_seq_len=64)
    assert cfg.factorized and cfg.effective_depth == 8
    model = LocalAgentLM(cfg)
    assert model.in_proj is not None and model.out_proj is not None and model.loop_embed is not None
    idx = torch.randint(0, cfg.vocab_size, (2, 16))
    logits, loss = model(idx, targets=idx)
    assert logits.shape == (2, 16, cfg.vocab_size)
    assert torch.isfinite(loss)


def test_real_ultra_tiny_config_constructs_under_budget():
    cfg = ModelConfig.from_yaml("configs/model/ultra-tiny-1m.yaml")
    model = LocalAgentLM(cfg)
    assert model.num_params() < 1.3e6
    logits, _ = model(torch.randint(0, cfg.vocab_size, (1, 8)))
    assert logits.shape[-1] == cfg.vocab_size


def test_budget_enforced_at_construction():
    import pytest

    big = ModelConfig(vocab_size=32000, d_model=2048, n_layers=24, n_heads=16, n_kv_heads=4,
                      ffn_hidden=8192)
    with pytest.raises(ValueError):
        LocalAgentLM(big)
