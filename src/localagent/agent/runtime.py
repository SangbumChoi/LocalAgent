"""The agent loop (Phase 7): generate -> if tool_call, dispatch + feed response, else answer.

    build prompt(system + tools + history + memory.core)
      -> generate
        -> parser.extract_tool_calls()
             calls?  -> tools.dispatch() -> append <tool_response> -> loop
             else    -> return assistant text (incl. correct abstention)

Every finished turn is handed to the conversation store, which feeds the data flywheel.
"""

from __future__ import annotations

from localagent.agent.memory import Memory
from localagent.agent.tools import ToolRegistry


class Agent:
    def __init__(self, model, tokenizer, tools: ToolRegistry, memory: Memory | None = None):
        self.model = model
        self.tokenizer = tokenizer
        self.tools = tools
        self.memory = memory or Memory()

    def chat(self, user_message: str, max_tool_hops: int = 6) -> str:
        """Single-turn agent step: grounded-decode a call, dispatch it, return the result.

        Uses prompt-grounded constrained decoding (reliable for the tiny model). Multi-turn /
        memory threading is TODO(phase-7); this is the working minimal loop for the demo.
        """
        from localagent.agent.constrained import grounded_decode
        from localagent.agent.parser import extract_tool_calls

        out = grounded_decode(self.model, self.tokenizer, user_message, self.tools.specs())
        calls = extract_tool_calls(out)
        if calls:
            c = calls[0]
            result = self.tools.dispatch(c.name, c.arguments)
            return f"[{c.name}({c.arguments}) -> {result}]"
        return out  # plain text answer / abstention
