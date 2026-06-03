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
        raise NotImplementedError(
            "TODO(phase-7): prompt build + generate + tool-dispatch loop + memory + logging"
        )
