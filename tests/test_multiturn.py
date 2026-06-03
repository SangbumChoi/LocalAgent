from localagent.data.agent_synth import Generator
from localagent.data.render import IGNORE, render_conversation
from localagent.data.schema import Role
from localagent.model.tokenizer import load_tokenizer
from localagent.agent.pointer_head import PointerHead, gold_span


def test_episode_is_multiturn_with_tool_response():
    conv = Generator(3, 0, "train").coding_episode()
    roles = [m.role for m in conv.messages]
    assert Role.user in roles and Role.tool in roles
    assert sum(m.role == Role.assistant and bool(m.tool_calls) for m in conv.messages) >= 1


def test_render_conversation_masks_nonassistant():
    tok = load_tokenizer("byte")
    conv = Generator(3, 1, "train").coding_episode()
    ids, labels = render_conversation(conv, tok)
    assert len(ids) == len(labels)
    assert any(l != IGNORE for l in labels)   # assistant turns are learned
    assert any(l == IGNORE for l in labels)   # user/tool turns are masked


def test_gold_span_locates_subsequence():
    framed = [1, 2, 3, 4, 5]
    assert gold_span(framed, [3, 4]) == (2, 3)
    assert gold_span(framed, [9]) is None


def test_pointer_head_predict_span_in_range():
    import torch
    ph = PointerHead(d_model=16)
    s, e = ph.predict_span(torch.randn(7, 16), "path")
    assert 0 <= s <= e < 7
