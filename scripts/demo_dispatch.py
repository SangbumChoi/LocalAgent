"""Demo: natural question -> auto tool dispatch (route head -> dense selector -> grounded call).
Shows the realistic-agent behaviour: the user just asks, the system picks the tool + args."""
import torch

from localagent.agent.constrained import hybrid_decode
from localagent.agent.dense_selector import BoundSelector, train_dense_selector
from localagent.agent.pointer_head import PointerHead
from localagent.agent.routes import ROUTES, RouteHead, route_of
from localagent.agent.tool_head import _feat
from localagent.agent.toolset import STANDARD_TOOLS as TOOLS
from localagent.data.agent_synth import Generator
from localagent.model import LocalAgentLM, ModelConfig
from localagent.model.tokenizer import load_tokenizer

torch.set_num_threads(4)
tok = load_tokenizer()
ck = torch.load("runs/tiny-30m-50tools.pt", map_location="cpu")
cfg = ModelConfig(**ck["cfg"])
m = LocalAgentLM(cfg); m.load_state_dict(ck["state_dict"]); m.eval()
ptr = PointerHead(cfg.d_model); ptr.load_state_dict(ck["ptr_head"])
rk = torch.load("runs/tiny-30m-routed.pt", map_location="cpu")
route_head = RouteHead(cfg.d_model); route_head.load_state_dict(rk["route_head"]); route_head.eval()

sel = train_dense_selector(m, Generator(level=3, seed=11).generate_balanced(3), tok, TOOLS,
                           steps=500, device="cpu")
selector = BoundSelector(sel, TOOLS, device="cpu")

QUERIES = [
    "What is the color of a monkey?",
    "Look up who invented the telephone.",
    "Open https://github.com/pytorch/pytorch and show me the readme",
    "Download the dataset from https://example.com/data.zip",
    "List the files in the src directory",
    "Search the codebase for the function train_step",
    "Make a directory called build",
    "Run the test suite",
    "What is 18 * 24?",
    "Email Dana the quarterly report",
]

print(f"\n{'query':52}{'route(head)':>14}{'selector top-3':>34}", flush=True)
print("-" * 100, flush=True)
for q in QUERIES:
    with torch.no_grad():
        feat = _feat(m, tok, q, "cpu")
        pred_route = ROUTES[int(route_head(feat).argmax(-1))]
        top3 = selector.rank(feat)[:3]
    call = hybrid_decode(m, tok, q, TOOLS, selector=selector, top_m=1, ptr_head=ptr)
    print(f"{q[:50]:52}{pred_route:>14}   {', '.join(top3)}", flush=True)
    print(f"{'  -> dispatched call:':52}{call}", flush=True)
print("\nDEMO_DONE", flush=True)
