"""End-to-end deployment smoke: load a dispatch checkpoint into the Agent and run real chats.
Validates the full deployed path: checkpoint -> route head -> dense selector -> pointer-copy ->
ToolRegistry.dispatch. Tools here are stubs that echo the call."""
import argparse

from localagent.agent.runtime import Agent
from localagent.agent.tools import ToolRegistry
from localagent.agent.toolset import STANDARD_TOOLS as TOOLS

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="runs/tiny-30m-scenarios-best.pt")
args = ap.parse_args()

reg = ToolRegistry()
for t in TOOLS:
    reg.register(t, (lambda _n: (lambda **kw: {"ok": _n, "args": kw}))(t.name))

agent = Agent.from_checkpoint(args.ckpt, reg)
print(f"loaded {args.ckpt}: selector={agent.selector is not None} "
      f"route_head={agent.route_head is not None} ptr={agent.ptr_head is not None}\n", flush=True)

QUERIES = [
    "What is the color of a monkey?",
    "Look up who invented the telephone.",
    "Open https://github.com/pytorch/pytorch in the browser",
    "List the files in the src directory",
    "Search the codebase for the function train_step",
    "Make a directory called build",
    "Run the test suite",
    "What is 18 * 24?",
    "Email Dana the quarterly report",
    "Download the dataset from https://example.com/data.zip",
]
for q in QUERIES:
    print(f">> {q}\n   {agent.chat(q)}", flush=True)
print("\nDEPLOY_DONE", flush=True)
