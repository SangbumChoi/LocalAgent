import json

import torch

from localagent.inference.export.to_hf import export_hf
from localagent.model import LocalAgentLM, ModelConfig


def test_hf_bundle_builds_and_roundtrips(tmp_path):
    cfg = ModelConfig(vocab_size=256, d_model=64, embed_dim=32, n_layers=2, n_loops=2,
                      n_heads=4, n_kv_heads=2, ffn_hidden=128, max_seq_len=64, name="test-tiny")
    m = LocalAgentLM(cfg)
    ckpt = tmp_path / "m.pt"
    torch.save({"cfg": cfg.__dict__, "state_dict": m.state_dict(),
                "tool_head": None, "ptr_head": None}, ckpt)

    out = export_hf(str(ckpt), str(tmp_path / "hf"), push=False)
    cfg_d = json.load(open(f"{out}/config.json"))
    assert cfg_d["name"] == "test-tiny" and cfg_d["vocab_size"] == 256
    assert (tmp_path / "hf" / "README.md").exists()

    # reload weights into a fresh model
    from safetensors.torch import load_file
    m2 = LocalAgentLM(cfg)
    m2.load_state_dict(load_file(f"{out}/model.safetensors"))
    x = torch.randint(0, 256, (1, 8))
    assert torch.allclose(m(x)[0], m2(x)[0], atol=1e-5)
