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


def test_boolean_and_message_slots_are_typed_and_grounded():
    from localagent.data.schema import ToolSpec
    from localagent.agent.constrained import _tool_bodies

    toggle = ToolSpec(
        name="set_wifi_status",
        description="Enable or disable Wi-Fi.",
        parameters={
            "type": "object",
            "properties": {"on": {"type": "boolean"}},
            "required": ["on"],
        },
    )
    assert any('"on":false' in body for body in _tool_bodies("Turn off wifi", toggle))

    message = ToolSpec(
        name="send_message",
        description="Send a message.",
        parameters={
            "type": "object",
            "properties": {
                "phone_number": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["phone_number", "content"],
        },
    )
    bodies = _tool_bodies(
        "Send a message to +12453344098 saying: How's the new album coming along", message
    )
    assert any(
        '"phone_number":"+12453344098"' in body
        and '"content":"' in body
        and "How's the new album coming along" in body
        for body in bodies
    )


def test_identifier_slots_do_not_copy_the_entire_instruction():
    from localagent.agent.constrained import _tool_bodies
    from localagent.data.schema import ToolSpec

    update = ToolSpec(
        name="update_task_status",
        description="Update a task.",
        parameters={
            "type": "object",
            "properties": {
                "task_id": {"type": "string"},
                "status": {"type": "string", "enum": ["pending", "completed"]},
            },
            "required": ["task_id", "status"],
        },
    )
    bodies = _tool_bodies(
        "Mark the existing task task_1 as completed.",
        update,
    )
    assert any('"task_id":"task_1"' in body for body in bodies)
    assert all("Mark the existing task" not in body for body in bodies)


def test_phone_grounding_prefers_explicit_number_over_uuid_digits():
    from localagent.agent.constrained import _phone

    prompt = "TOOL_RESULT: person_id=9e137f06-916a-5310-8174-cf0b7e9f7054 phone=+12453344098"
    assert _phone(prompt) == ["+12453344098"]


def test_stateful_app_url_and_semantic_targets_prefer_typed_spans():
    from localagent.agent.constrained import _tool_bodies
    from localagent.agent.mobile_toolset import mobile_tools

    app = next(tool for tool in mobile_tools() if tool.name == "mobile_open_app")
    click = next(tool for tool in TOOLS if tool.name == "click")
    app_bodies = _tool_bodies(
        "Goal: Create a Notion page. Next required action: Start the Notion application on the handset.",
        app,
    )
    click_bodies = _tool_bodies(
        "Goal: Search. Next required action: Select the mail search field.", click
    )
    assert any('"app_name":"Notion"' in body for body in app_bodies)
    assert any('"target":"the mail search field"' in body for body in click_bodies)


def test_hello_is_text_only_not_planner():
    # "hello to X" must NOT fire the planner's " to " trigger.
    cands = candidates("Say hello to Zara.", TOOLS)
    assert all(not is_tool for _, is_tool, _ in cands)
    assert ("Hello, Zara!", False, "text") in cands


def test_thanks_abstains_with_text():
    cands = candidates("Thanks for your help!", TOOLS)
    assert all(not is_tool for _, is_tool, _ in cands)


def test_completion_language_abstains_from_tool_calls():
    cands = candidates("The work is already complete; no action is needed.", TOOLS)
    assert cands == [("I won't invoke a tool.", False, "text")]


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


def test_context_features_clamp_overlong_multi_turn_history():
    from localagent.agent.constrained import _ctx_feats
    from localagent.model import LocalAgentLM, ModelConfig
    from localagent.model.tokenizer import load_tokenizer

    cfg = ModelConfig(vocab_size=256, d_model=64, embed_dim=64, n_layers=2, n_loops=1,
                      n_heads=4, n_kv_heads=2, ffn_hidden=128, max_seq_len=128,
                      rope_theta=10000.0, norm_eps=1e-5, tie_embeddings=True, dropout=0.0)
    m = LocalAgentLM(cfg).eval()
    tok = load_tokenizer("byte")
    feats, ids = _ctx_feats(m, tok, "tool result. " * 200, device="cpu")
    assert len(ids) == cfg.max_seq_len
    assert feats.shape[0] == cfg.max_seq_len


def test_playwright_abi_forces_navigation_then_snapshot_without_inventing_refs():
    from localagent.agent.constrained import (
        _arg_options,
        _playwright_lexical_tool,
        _tool_bodies,
    )
    from localagent.data.schema import ToolSpec

    tools = [
        ToolSpec(
            name="browser_navigate",
            description="Navigate to a URL",
            parameters={
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
            },
        ),
        ToolSpec(
            name="browser_snapshot",
            description="Take an accessibility snapshot",
            parameters={"type": "object", "properties": {}, "required": []},
        ),
        ToolSpec(
            name="browser_click",
            description="Click an exact snapshot reference",
            parameters={
                "type": "object",
                "properties": {"ref": {"type": "string"}},
                "required": ["ref"],
            },
        ),
    ]
    prompt = "Navigate to https://example.test/inbox and then inspect the page."
    assert _playwright_lexical_tool(prompt, tools) == "browser_navigate"
    nav = _tool_bodies(prompt, tools[0])
    assert any('"url":"https://example.test/inbox"' in body for body in nav)
    assert _playwright_lexical_tool(
        prompt + '\nASSISTANT: <tool_call>{"name":"browser_navigate"}</tool_call>\n'
        "TOOL_RESULT: page loaded",
        tools,
    ) == "browser_snapshot"
    click = tools[2]
    assert _tool_bodies(prompt, click) == []
    grounded = prompt + "\nASSISTANT: browser_snapshot\nTOOL_RESULT: - button Save [ref=e12]"
    assert _arg_options(grounded, "ref", {"type": "string"}, True) == ["e12"]


def test_best_abstains_when_a_grounded_candidate_exceeds_context_window():
    from localagent.agent.constrained import _best
    from localagent.model import LocalAgentLM, ModelConfig
    from localagent.model.tokenizer import load_tokenizer

    cfg = ModelConfig(vocab_size=256, d_model=64, embed_dim=64, n_layers=2, n_loops=1,
                      n_heads=4, n_kv_heads=2, ffn_hidden=128, max_seq_len=128,
                      rope_theta=10000.0, norm_eps=1e-5, tie_embeddings=True, dropout=0.0)
    m = LocalAgentLM(cfg).eval()
    tok = load_tokenizer("byte")
    body = "<tool_call>" + "x" * 500 + "</tool_call>"
    assert _best(m, tok, "Find the message.", [body], device="cpu") == (
        "I cannot complete this request."
    )


def test_hybrid_grounding_prompt_does_not_copy_serialized_catalog_text():
    from localagent.agent.constrained import hybrid_decode
    from localagent.data.schema import ToolSpec

    send = ToolSpec(
        name="email_send",
        description="Send an email.",
        parameters={
            "type": "object",
            "properties": {"to": {"type": "string"}},
            "required": ["to"],
        },
    )
    catalog = '<|tool_catalog|>{"name":"catalog_only_noise"}</|tool_catalog|>'
    grounding = "<|user|>Send an email to alice@example.com<|assistant|>"
    output = hybrid_decode(
        None,
        None,
        catalog + grounding,
        [send],
        framed=True,
        selector_first=True,
        grounding_prompt=grounding,
    )
    assert "alice@example.com" in output
    assert "catalog_only_noise" not in output


def test_pointer_span_bounds_exclude_serialized_catalog_prefix():
    import torch

    from localagent.agent.constrained import _grounding_span
    from localagent.agent.pointer_head import PointerHead
    from localagent.model.tokenizer import ASSISTANT, load_tokenizer

    tok = load_tokenizer("byte")
    catalog = "<|tool_catalog|>schema prose alice@example.com</|tool_catalog|>"
    grounding = "<|user|>Send mail to bob@example.com"
    ids = tok.encode(catalog + grounding + ASSISTANT)
    bounds = _grounding_span(ids, tok, grounding)
    assert bounds is not None
    lo, hi = bounds
    assert tok.decode(ids[lo : hi + 1]) == grounding

    pointer = PointerHead(8, args=["recipient"])
    start, end = pointer.predict_span(torch.randn(len(ids), 8), "recipient", span_bounds=bounds)
    assert lo <= start <= end <= hi
