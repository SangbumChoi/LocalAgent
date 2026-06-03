"""Web demo (Phase 7): visualize the agent loop.

A small Gradio app that streams tokens, renders the tool_call/tool_response trace as it
happens, and shows the two-tier memory state — the "visualizing the demos" goal.
Run: python demos/web/app.py runs/sft/latest.pt
"""

from __future__ import annotations

import sys


def build_app(checkpoint: str):
    raise NotImplementedError(
        "TODO(phase-7): Gradio UI — chat + live tool trace panel + memory panel + token stream"
    )


if __name__ == "__main__":
    build_app(sys.argv[1] if len(sys.argv) > 1 else "runs/sft/latest.pt")
