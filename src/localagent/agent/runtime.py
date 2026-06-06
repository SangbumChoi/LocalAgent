"""The agent loop (Phase 7): select a tool -> ground its args -> dispatch.

Two selection modes:
  - small fixed toolset: the trained tool head + pointer head (grounded_decode).
  - large catalog (100s–1000s of tools): a ToolRetriever picks the top-k candidates, then we
    ground+rank only those — selection cost is O(top-k), not O(catalog).

Every finished turn can be handed to the conversation store, which feeds the data flywheel.
"""

from __future__ import annotations

from localagent.agent.memory import Memory
from localagent.agent.tools import ToolRegistry


class Agent:
    def __init__(self, tools: ToolRegistry, model=None, tokenizer=None, memory: Memory | None = None,
                 catalog=None, retriever=None, retrieve_k: int = 10, tool_head=None, ptr_head=None):
        """`tools`: registry for dispatch. For a large tool space pass `catalog` (list of ToolSpec)
        and optionally a `retriever` (built from the catalog if omitted)."""
        self.tools = tools
        self.model = model
        self.tokenizer = tokenizer
        self.memory = memory or Memory()
        self.catalog = {t.name: t for t in (catalog or tools.specs())}
        self.retrieve_k = retrieve_k
        self.tool_head, self.ptr_head = tool_head, ptr_head
        self.retriever = retriever
        if self.retriever is None and catalog is not None:
            from localagent.agent.retriever import ToolRetriever
            self.retriever = ToolRetriever(list(self.catalog.values()))

    def _select_specs(self, msg: str):
        """Candidate ToolSpecs for this turn: top-k retrieved, or the whole (small) toolset."""
        if self.retriever is not None:
            return [self.catalog[n] for n in self.retriever.retrieve(msg, self.retrieve_k)]
        return list(self.catalog.values())

    def chat(self, user_message: str, max_tool_hops: int = 6) -> str:
        from localagent.agent.constrained import _best, _tool_bodies, grounded_decode
        from localagent.agent.parser import extract_tool_calls

        specs = self._select_specs(user_message)
        if self.model is not None:
            # rank the (retrieved) candidates' grounded bodies with the model
            out = grounded_decode(self.model, self.tokenizer, user_message, specs,
                                  tool_head=self.tool_head, ptr_head=self.ptr_head)
        else:
            # no model: retriever order + schema-grounded args (works over 1000s of tools)
            bodies = [b for t in specs for b in _tool_bodies(user_message, t)[:1]]
            out = bodies[0] if bodies else ""

        calls = extract_tool_calls(out)
        if not calls:
            return out  # plain text / abstention
        c = calls[0]
        result = self.tools.dispatch(c.name, c.arguments)
        return f"[{c.name}({c.arguments}) -> {result}]"
