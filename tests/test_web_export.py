import json

import numpy as np
import pytest
import torch

from localagent.model import LocalAgentLM, ModelConfig

pytest.importorskip("onnx")
ort = pytest.importorskip("onnxruntime")
pytest.importorskip("onnxscript")


def _make_bundle(tmp_path):
    """Export a small full bundle (model.onnx + heads.json + meta.json) and return paths + model."""
    from localagent.agent.pointer_head import PointerHead
    from localagent.agent.tool_head import ToolHead
    from localagent.inference.export.to_onnx import export_web

    cfg = ModelConfig(vocab_size=256, d_model=64, embed_dim=64, n_layers=2, n_loops=1,
                      n_heads=4, n_kv_heads=2, ffn_hidden=128, max_seq_len=64, name="t")
    m = LocalAgentLM(cfg).eval()
    th = ToolHead(cfg.d_model)
    ph = PointerHead(cfg.d_model)
    ckpt = tmp_path / "m.pt"
    torch.save({
        "cfg": cfg.__dict__,
        "state_dict": m.state_dict(),
        "tool_head": th.state_dict(),
        "ptr_head": ph.state_dict(),
    }, ckpt)
    out_dir = tmp_path / "web"
    stats = export_web(str(ckpt), str(out_dir), fp16=True, check=False)
    return stats, m, th, ph, cfg


def test_web_export_hidden_logits_parity(tmp_path):
    stats, m, th, ph, cfg = _make_bundle(tmp_path)
    sess = ort.InferenceSession(stats["model.onnx"], providers=["CPUExecutionProvider"])

    x = torch.randint(0, 256, (1, 13))
    with torch.no_grad():
        ref_logits, ref_hidden = m(x, return_hidden=True)
    ref_logits, ref_hidden = ref_logits.numpy(), ref_hidden.numpy()
    got_logits, got_hidden = sess.run(["logits", "hidden"], {"input_ids": x.numpy()})

    assert got_logits.shape == ref_logits.shape == (1, 13, 256)
    assert got_hidden.shape == ref_hidden.shape == (1, 13, cfg.d_model)
    assert np.abs(ref_logits - got_logits).max() < 1e-3
    assert np.abs(ref_hidden - got_hidden).max() < 1e-3


def test_web_export_tool_head_from_json(tmp_path):
    """Prove JS can reproduce tool selection from the onnx `hidden` output + heads.json."""
    stats, m, th, ph, cfg = _make_bundle(tmp_path)
    heads = json.load(open(stats["heads.json"]))
    meta = json.load(open(stats["meta.json"]))
    sess = ort.InferenceSession(stats["model.onnx"], providers=["CPUExecutionProvider"])

    x = torch.randint(0, 256, (1, 11))
    got_logits, got_hidden = sess.run(["logits", "hidden"], {"input_ids": x.numpy()})

    # JS-side recipe: last position, matmul with tool_head weights.
    W = np.array(heads["tool_head"]["weight"], dtype=np.float32)   # (n_classes, d_model)
    b = np.array(heads["tool_head"]["bias"], dtype=np.float32)     # (n_classes,)
    last = got_hidden[:, -1]                                       # (1, d_model)
    js_logits = last @ W.T + b                                     # (1, n_classes)

    with torch.no_grad():
        ref_tool = th(torch.tensor(got_hidden[:, -1]))            # (1, 22)
    ref_tool = ref_tool.numpy()

    assert js_logits.shape == ref_tool.shape
    assert np.abs(js_logits - ref_tool).max() < 1e-3
    assert int(js_logits.argmax(-1)[0]) == int(ref_tool.argmax(-1)[0])

    # contract sanity: stop_index points at "text"; classes match meta.
    assert heads["tool_head"]["classes"][heads["tool_head"]["stop_index"]] == "text"
    assert meta["tool_classes"] == heads["tool_head"]["classes"]
    assert meta["pad_id"] == 0


def test_web_export_pointer_head_from_json(tmp_path):
    """Reproduce pointer-head span logits from heads.json (numpy) vs PyTorch."""
    stats, m, th, ph, cfg = _make_bundle(tmp_path)
    heads = json.load(open(stats["heads.json"]))
    sess = ort.InferenceSession(stats["model.onnx"], providers=["CPUExecutionProvider"])

    x = torch.randint(0, 256, (1, 9))
    _, hidden = sess.run(["logits", "hidden"], {"input_ids": x.numpy()})
    h = hidden[0]                                                  # (T, d_model)

    arg_emb = np.array(heads["pointer_head"]["arg_emb"], dtype=np.float32)
    start_W = np.array(heads["pointer_head"]["start_W"], dtype=np.float32)
    end_W = np.array(heads["pointer_head"]["end_W"], dtype=np.float32)
    arg = "query"
    i = heads["pointer_head"]["arg_idx"][arg]
    q = arg_emb[i]                                                 # (d_model,)
    qs = start_W @ q                                               # (d_model,)
    qe = end_W @ q
    s_js = h @ qs                                                  # (T,)
    e_js = h @ qe

    with torch.no_grad():
        s_ref, e_ref = ph.logits(torch.tensor(hidden),
                                 torch.tensor([i], dtype=torch.long))
    s_ref, e_ref = s_ref.numpy()[0], e_ref.numpy()[0]

    assert np.abs(s_js - s_ref).max() < 1e-3
    assert np.abs(e_js - e_ref).max() < 1e-3
