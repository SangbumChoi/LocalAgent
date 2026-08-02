"""The agent loop (Phase 7): select a tool -> ground its args -> dispatch.

Two selection modes:
  - small fixed toolset: the trained tool head + pointer head (grounded_decode).
  - large catalog (100s–1000s of tools): a ToolRetriever picks the top-k candidates, then we
    ground+rank only those — selection cost is O(top-k), not O(catalog).

Every finished turn can be handed to the conversation store, which feeds the data flywheel.
"""

from __future__ import annotations

from pathlib import Path

from localagent.agent.memory import Memory
from localagent.agent.tools import ToolRegistry


def _checkpoint_tokenizer(
    checkpoint: dict,
    vocab_size: int,
    *,
    tokenizer_path: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
):
    """Load and validate the tokenizer recorded in a training checkpoint.

    Dispatch checkpoints made before tokenizer metadata was introduced are byte-level, so a
    missing ``tokenizer`` entry intentionally keeps that behavior. ``tokenizer_path`` overrides a
    recorded BPE path when artifacts have been moved. For relative recorded paths, also try a path
    next to the checkpoint after preserving the historical current-working-directory lookup.
    """
    from localagent.model.tokenizer import load_tokenizer

    metadata = checkpoint.get("tokenizer")
    if metadata is None:
        metadata = {"kind": "byte"}
    if not isinstance(metadata, dict):
        raise ValueError("checkpoint tokenizer metadata must be a mapping")

    kind = str(metadata.get("kind", "byte"))
    path = tokenizer_path if tokenizer_path is not None else metadata.get("path")
    if kind == "bpe" and path is not None and tokenizer_path is None and checkpoint_path is not None:
        recorded = Path(path).expanduser()
        if not recorded.is_absolute() and not recorded.exists():
            colocated = Path(checkpoint_path).resolve().parent / recorded
            if colocated.exists():
                path = colocated

    tokenizer = load_tokenizer(kind, path)
    if tokenizer.vocab_size != vocab_size:
        raise ValueError(
            "checkpoint tokenizer vocabulary "
            f"({tokenizer.vocab_size}) does not match model config ({vocab_size})"
        )
    return tokenizer


class Agent:
    def __init__(self, tools: ToolRegistry, model=None, tokenizer=None, memory: Memory | None = None,
                 catalog=None, retriever=None, retrieve_k: int = 10, tool_head=None, ptr_head=None,
                 route_head=None, selector=None):
        """`tools`: registry for dispatch. For a large tool space pass `catalog` (list of ToolSpec)
        and optionally a `retriever` (built from the catalog if omitted).

        Dispatch path (preferred): pass a trained `selector` (a `dense_selector.BoundSelector`) and
        optional `route_head` — the *generable* route->select->copy pipeline (`hybrid_decode`), which
        scales to any tool pool. Falls back to the legacy fixed-N `tool_head` path when no selector
        is given. `Agent.from_checkpoint(...)` wires all of this from a dispatch checkpoint."""
        self.tools = tools
        self.model = model
        self.tokenizer = tokenizer
        self.memory = memory or Memory()
        self.catalog = {t.name: t for t in (catalog or tools.specs())}
        self.retrieve_k = retrieve_k
        self.tool_head, self.ptr_head = tool_head, ptr_head
        self.route_head, self.selector = route_head, selector
        self.retriever = retriever
        if self.retriever is None and catalog is not None:
            from localagent.agent.retriever import ToolRetriever
            self.retriever = ToolRetriever(list(self.catalog.values()))

    @classmethod
    def from_checkpoint(
        cls,
        ckpt_path: str | Path,
        tools: ToolRegistry,
        *,
        tokenizer_path: str | Path | None = None,
        **kw,
    ):
        """Load model + pointer head + dense selector + route head from a dispatch checkpoint
        (saved by train_dispatch_long / train_scenarios) and wire the generable dispatch path.

        ``tokenizer_path`` overrides a BPE path stored in the checkpoint, which is useful when the
        checkpoint and tokenizer bundle have moved. Legacy checkpoints without tokenizer metadata
        remain byte-tokenized.
        """
        import torch

        from localagent.agent.dense_selector import (
            BoundSelector, DenseToolSelector, tool_embeddings,
        )
        from localagent.agent.pointer_head import PointerHead
        from localagent.agent.routes import RouteHead
        from localagent.model import LocalAgentLM, ModelConfig

        ck = torch.load(ckpt_path, map_location="cpu")
        cfg = ModelConfig(**ck["cfg"])
        model = LocalAgentLM(cfg)
        model.load_state_dict(ck["state_dict"])
        model.eval()
        tokenizer = _checkpoint_tokenizer(
            ck,
            cfg.vocab_size,
            tokenizer_path=tokenizer_path,
            checkpoint_path=ckpt_path,
        )
        specs = tools.specs()
        examples = ck.get("examples", {})
        ptr = None
        if ck.get("ptr_head"):
            ptr = (
                PointerHead(cfg.d_model, args=ck["ptr_args"])
                if ck.get("ptr_args")
                else PointerHead(cfg.d_model)
            )
            ptr.load_state_dict(ck["ptr_head"])
            ptr.eval()
        selector = route_head = None
        if ck.get("dense_selector"):
            emb_dim = tool_embeddings(specs[:1]).shape[1]
            sel = DenseToolSelector(cfg.d_model, emb_dim=emb_dim, proj=ck.get("selector_proj", 256))
            sel.load_state_dict(ck["dense_selector"])
            selector = BoundSelector(sel, specs, examples=examples)
        if ck.get("route_head"):
            route_head = RouteHead(cfg.d_model)
            route_head.load_state_dict(ck["route_head"])
            route_head.eval()
        return cls(tools, model=model, tokenizer=tokenizer, catalog=specs,
                   ptr_head=ptr, route_head=route_head, selector=selector, **kw)

    def _select_specs(self, msg: str):
        """Candidate ToolSpecs for this turn: top-k retrieved, or the whole (small) toolset."""
        if self.retriever is not None:
            return [self.catalog[n] for n in self.retriever.retrieve(msg, self.retrieve_k)]
        return list(self.catalog.values())

    def chat(self, user_message: str, max_tool_hops: int = 6) -> str:
        from localagent.agent.constrained import _tool_bodies, grounded_decode, hybrid_decode
        from localagent.agent.parser import extract_tool_calls

        specs = self._select_specs(user_message)
        if self.model is not None and self.selector is not None:
            # generable path: route gate -> dense selector -> pointer-copy args (scales to any pool)
            out = hybrid_decode(self.model, self.tokenizer, user_message, list(self.catalog.values()),
                                selector=self.selector, route_head=self.route_head,
                                ptr_head=self.ptr_head, top_m=1)
        elif self.model is not None:
            # legacy: rank the (retrieved) candidates' grounded bodies with the fixed-N tool head
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
