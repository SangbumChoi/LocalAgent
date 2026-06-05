"""Prompt-grounded constrained decoding: candidate proposal must ground slots in the prompt."""

from localagent.agent.constrained import candidates
from localagent.agent.toolset import STANDARD_TOOLS as TOOLS


def _bodies(prompt):
    return [b for b, _, _ in candidates(prompt, TOOLS)]


def test_weather_city_is_grounded_in_prompt():
    bodies = _bodies("What's the weather in Boston?")
    assert any('"city":"Boston"' in b and "get_weather" in b for b in bodies)


def test_calculator_expression_extracted():
    bodies = _bodies("What is 7 * 8?")
    assert any('"expression":"7*8"' in b for b in bodies)


def test_hello_is_text_only_not_planner():
    # "hello to X" must NOT fire the planner's " to " trigger.
    cands = candidates("Say hello to Zara.", TOOLS)
    assert all(not is_tool for _, is_tool, _ in cands)
    assert ("Hello, Zara!", False, "text") in cands


def test_thanks_abstains_with_text():
    cands = candidates("Thanks for your help!", TOOLS)
    assert all(not is_tool for _, is_tool, _ in cands)


def test_best_clamps_overlong_context():
    # A long multi-turn history + a candidate body can exceed max_seq_len; _best must trim the
    # oldest prompt tokens (left) instead of overflowing RoPE. Regression for the flywheel
    # multi-turn-eval crash (constrained.py:_best).
    from localagent.agent.constrained import _best
    from localagent.model import LocalAgentLM, ModelConfig
    from localagent.model.tokenizer import load_tokenizer

    cfg = ModelConfig(vocab_size=256, d_model=64, embed_dim=64, n_layers=2, n_loops=1,
                      n_heads=4, n_kv_heads=2, ffn_hidden=128, max_seq_len=128,
                      rope_theta=10000.0, norm_eps=1e-5, tie_embeddings=True, dropout=0.0)
    m = LocalAgentLM(cfg).eval()
    tok = load_tokenizer("byte")
    long_prompt = "step. " * 60                       # ~360 bytes >> 128-token window
    bodies = ['<tool_call>{"arguments":{"path":"a.py"},"name":"read_file"}</tool_call>',
              '<tool_call>{"arguments":{},"name":"run_tests"}</tool_call>']
    out = _best(m, tok, long_prompt, bodies, device="cpu")   # must not raise
    assert out in bodies
