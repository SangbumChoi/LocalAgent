"""Parity for the exported dispatch heads (route_head + dense_selector) vs PyTorch.

Mirrors test_web_export.py: prove the device can reproduce route selection and dense-selector
top-1 from the exported JSON (route weights + query tower + precomputed tool matrix) applied to the
model's final hidden state, matching the PyTorch RouteHead / BoundSelector within tolerance.
"""

import json

import numpy as np
import torch

from localagent.agent.dense_selector import (
    BoundSelector,
    DenseToolSelector,
    tool_embeddings,
)
from localagent.agent.routes import ROUTES, RouteHead
from localagent.agent.tool_head import _feat
from localagent.agent.toolset import STANDARD_TOOLS
from localagent.inference.export.to_dispatch import (
    dispatch_heads_json,
    selector_tool_matrix,
)
from localagent.model import LocalAgentLM, ModelConfig
from localagent.model import tokenizer as tk

_PROMPTS = [
    "What is the color of a monkey?",
    "Read the file data/loader.py.",
    "What's the weather like in Oslo?",
    "Commit the staged changes with message fix bug.",
    "Email Dana the quarterly report.",
    "What is 18 * 24?",
]


def _make_ck(tmp_path):
    cfg = ModelConfig(vocab_size=256, d_model=64, embed_dim=64, n_layers=2, n_loops=1,
                      n_heads=4, n_kv_heads=2, ffn_hidden=128, max_seq_len=128, name="t")
    m = LocalAgentLM(cfg).eval()
    rh = RouteHead(cfg.d_model)
    emb_dim = tool_embeddings(STANDARD_TOOLS[:1], dim=1024).shape[1]
    sel = DenseToolSelector(cfg.d_model, emb_dim=emb_dim, proj=256)
    examples = {STANDARD_TOOLS[0].name: ["the weather in Paris"]}
    ck = {
        "cfg": cfg.__dict__,
        "state_dict": m.state_dict(),
        "route_head": rh.state_dict(),
        "dense_selector": sel.state_dict(),
        "selector_proj": 256,
        "examples": examples,
    }
    return ck, cfg, m, rh, sel, examples


def _feats(m):
    tok = tk.load_tokenizer("byte")
    with torch.no_grad():
        return torch.stack([_feat(m, tok, p, "cpu") for p in _PROMPTS])


def test_route_head_parity(tmp_path):
    ck, cfg, m, rh, sel, examples = _make_ck(tmp_path)
    heads = dispatch_heads_json(ck)
    feats = _feats(m)

    W = np.array(heads["route_head"]["weight"], dtype=np.float32)   # (5, d_model)
    b = np.array(heads["route_head"]["bias"], dtype=np.float32)     # (5,)
    exp = feats.numpy() @ W.T + b
    with torch.no_grad():
        ref = rh(feats).numpy()

    assert heads["route_head"]["routes"] == list(ROUTES)
    assert heads["route_head"]["routes"][heads["route_head"]["stop_index"]] == "text"
    assert exp.shape == ref.shape == (len(_PROMPTS), 5)
    assert np.abs(exp - ref).max() < 1e-3
    assert (exp.argmax(-1) == ref.argmax(-1)).all()


def test_dense_selector_top1_parity(tmp_path):
    ck, cfg, m, rh, sel, examples = _make_ck(tmp_path)
    heads = dispatch_heads_json(ck)
    feats = _feats(m)

    # PyTorch reference: BoundSelector top-1 over STANDARD_TOOLS (with the same examples).
    bound = BoundSelector(sel, STANDARD_TOOLS, examples=examples)
    ref_top1 = [bound.rank(f)[0] for f in feats]

    # device recipe: q = normalize(feat @ q_proj_W.T + q_proj_b); j = argmax_j q @ T[j].
    qW = np.array(heads["dense_selector"]["q_proj_weight"], dtype=np.float32)
    qb = np.array(heads["dense_selector"]["q_proj_bias"], dtype=np.float32)
    T = np.array(heads["dense_selector"]["tool_matrix"], dtype=np.float32)
    names = heads["dense_selector"]["tool_names"]

    assert T.shape == (len(STANDARD_TOOLS), heads["dense_selector"]["proj"])
    assert names == [t.name for t in STANDARD_TOOLS]
    # shipped rows are L2-normalized (the tool tower side of the score).
    assert np.allclose(np.linalg.norm(T, axis=-1), 1.0, atol=1e-4)

    q = feats.numpy() @ qW.T + qb
    q = q / (np.linalg.norm(q, axis=-1, keepdims=True) + 1e-12)
    scores = q @ T.T
    exp_top1 = [names[int(i)] for i in scores.argmax(-1)]
    assert exp_top1 == ref_top1

    # full-score numerical parity vs the PyTorch selector.
    with torch.no_grad():
        ref_scores = sel(feats, bound.embs).numpy()
    assert np.abs(ref_scores - scores).max() < 1e-3


def test_selector_tool_matrix_matches_bound_embs(tmp_path):
    """The precomputed matrix == normalize(t_proj(tool_embeddings)) the BoundSelector uses."""
    import torch.nn.functional as F
    ck, cfg, m, rh, sel, examples = _make_ck(tmp_path)
    T = selector_tool_matrix(ck, STANDARD_TOOLS, examples)
    bound = BoundSelector(sel, STANDARD_TOOLS, examples=examples)
    with torch.no_grad():
        ref = F.normalize(sel.t_proj(bound.embs), dim=-1)
    assert torch.allclose(T, ref, atol=1e-5)


def test_dispatch_json_roundtrips(tmp_path):
    ck, *_ = _make_ck(tmp_path)
    heads = dispatch_heads_json(ck)
    p = tmp_path / "dispatch_heads.json"
    p.write_text(json.dumps(heads))
    back = json.loads(p.read_text())
    assert back["route_head"]["routes"] == list(ROUTES)
    assert back["dense_selector"]["normalize_query"] is True
    assert len(back["dense_selector"]["tool_names"]) == len(STANDARD_TOOLS)
