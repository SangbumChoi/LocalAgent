from localagent.agent.runtime import Agent
from localagent.agent.tools import ToolRegistry
from localagent.data.tool_catalog import build_catalog


def _agent(n=80):
    tools = build_catalog(n, seed=0)
    reg = ToolRegistry()
    for t in tools:
        reg.register(t, lambda **kw: {"ok": True, **kw})
    return Agent(reg, catalog=tools), tools  # retriever auto-built from catalog


def test_agent_retrieves_grounds_dispatches():
    agent, tools = _agent()
    t = tools[3]
    verb, noun = t.name.split("_", 1)
    out = agent.chat(f"Please {verb} the {noun} 'demo value'.")   # literal -> top-1 should hit
    assert out.startswith(f"[{t.name}(") and "demo value" in out


def test_agent_scales_selection_to_topk():
    agent, _ = _agent(80)
    # retriever restricts candidates to top-k, not the whole catalog
    specs = agent._select_specs("open the report 'x'")
    assert len(specs) == agent.retrieve_k
