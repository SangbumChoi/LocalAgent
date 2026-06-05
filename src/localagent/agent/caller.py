"""ToolCaller — the public, developer-facing tool-calling API.

Give it any JSON-schema tools (multi-argument, real APIs); for a user turn it returns a
**schema-valid, grounded** `ToolCall` or `None` (abstention). Selection scales to thousands of
tools via retrieval (auto-enabled for large catalogs); arguments are filled by the schema-guided
constrained decoder (`schema_decode.py`), so the output is never malformed.

    from localagent import ToolCaller
    from localagent.data.schema import ToolSpec

    tools = [ToolSpec("move_file", "move or rename a file", {
        "type": "object",
        "properties": {"source": {"type": "string", "format": "path"},
                       "dest":   {"type": "string", "format": "path"}},
        "required": ["source", "dest"]})]

    caller = ToolCaller(tools)
    caller.call("Move src/app.py to backup/app.py.")   # ToolCall(move_file, {source:.., dest:..})
    caller.call("What's the weather?")                  # None  (abstains)

No model is required (selection = retrieval, arguments = grounding). Pass a model + heads to use
the trained tool/pointer heads instead of retrieval on a small fixed toolset.
"""

from __future__ import annotations

from localagent.agent.schema_decode import fill_tool
from localagent.data.schema import ToolCall, ToolSpec


class ToolCaller:
    def __init__(self, tools: list[ToolSpec], retrieve_k: int = 12, examples: dict | None = None,
                 min_score: float = 0.0, retriever=None):
        """`min_score`: abstain if the top retrieved tool's similarity is below this (0 = off)."""
        self.tools = {t.name: t for t in tools}
        self.specs = list(tools)
        self.k = retrieve_k
        self.min_score = min_score
        from localagent.agent.retriever import ToolRetriever
        # Always rank by relevance (even a small toolset) so the *relevant* tool is tried first,
        # not just the first one that happens to be fillable.
        self.retriever = retriever or ToolRetriever(tools, examples=examples)

    def candidates(self, query: str) -> list[tuple[ToolSpec, float]]:
        """Relevance-ranked candidate tools (top-k by retrieval similarity)."""
        return [(self.tools[n], s) for n, s in self.retriever.retrieve_scored(query, self.k)]

    def call(self, query: str) -> ToolCall | None:
        """Return a grounded, schema-valid ToolCall — or None (abstain) if nothing fits/grounds."""
        cands = self.candidates(query)
        if cands and cands[0][1] < self.min_score:
            return None                                   # nothing relevant enough -> abstain
        for tool, _ in cands:
            args = fill_tool(query, tool)
            if args is not None:
                return ToolCall(tool.name, args)
        return None

    def plan(self, query: str) -> list[ToolCall]:
        """Decompose a multi-step request into an ordered list of calls (planner/executor pattern,
        à la AutoGen/OctoTools/CodeAct). Splits on connectives ('then', 'and then', 'after that',
        'and', ';') and grounds each step; drops steps that don't yield a call."""
        import re
        parts = re.split(r"\s*(?:;|,?\s*then\s+|\s+after\s+that\s+|\s+and\s+then\s+|\s+and\s+)\s*",
                         query, flags=re.I)
        plan = []
        for p in parts:
            p = p.strip()
            if len(p) < 4:
                continue
            r = self.call(p if p.endswith((".", "?", "!")) else p + ".")
            if r is not None:
                plan.append(r)
        return plan

    def explain(self, query: str, top: int = 5) -> dict:
        """Debug view: ranked candidates, the chosen tool, and the grounded args."""
        cands = self.candidates(query)[:top]
        result = self.call(query)
        return {"query": query,
                "candidates": [(t.name, round(s, 3)) for t, s in cands],
                "call": None if result is None else {"name": result.name, "arguments": result.arguments}}
