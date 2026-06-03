"""Pointer/copy argument head (ARCHITECTURE_IDEAS §2a, the arg side of the dual head).

Replaces the heuristic span extractors in agent/constrained.py with a *learned* copy mechanism:
conditioned on which argument it is filling, the head predicts a (start, end) span over the
prompt's byte positions, and the argument value is the copied span. Trained jointly with the
model (auxiliary loss in SFT) so it generalizes to free-form values (shell commands, file
contents) that no regex/preposition heuristic can extract reliably.

One pointer arg per sample (our tools have ≤1 free-text string arg; enum/number/arithmetic stay
schema-extracted). Byte-level features make the span a literal byte slice of the prompt.
"""

from __future__ import annotations

import torch
import torch.nn as nn

# string args grounded by the pointer head (enum/number/arithmetic use schema extractors instead)
PTR_ARGS = ["city", "query", "goal", "term", "song", "topic",
            "path", "pattern", "command", "message", "task", "duration"]
ARG_IDX = {a: i for i, a in enumerate(PTR_ARGS)}


class PointerHead(nn.Module):
    def __init__(self, d_model: int, args=PTR_ARGS):
        super().__init__()
        self.args = args
        self.arg_emb = nn.Embedding(len(args), d_model)
        self.start = nn.Linear(d_model, d_model, bias=False)
        self.end = nn.Linear(d_model, d_model, bias=False)

    def logits(self, feats: torch.Tensor, arg_idx: torch.Tensor):
        """feats (B,T,d), arg_idx (B,) -> start/end logits (B,T)."""
        q = self.arg_emb(arg_idx)                       # (B,d)
        s = torch.einsum("btd,bd->bt", feats, self.start(q))
        e = torch.einsum("btd,bd->bt", feats, self.end(q))
        return s, e

    @torch.no_grad()
    def predict_span(self, feats_row: torch.Tensor, arg: str) -> tuple[int, int]:
        """feats_row (T,d) for a single prompt -> (start, end) byte indices, end >= start."""
        i = torch.tensor(ARG_IDX[arg], device=feats_row.device)
        q = self.arg_emb(i)
        s = feats_row @ self.start(q)
        e = feats_row @ self.end(q)
        start = int(s.argmax())
        e = e.clone()
        e[:start] = -1e9                                # end must not precede start
        return start, int(e.argmax())


def gold_span(framed_ids: list[int], value_ids: list[int]) -> tuple[int, int] | None:
    """First occurrence of value_ids within framed_ids -> (start, end) inclusive, else None."""
    n, m = len(framed_ids), len(value_ids)
    for i in range(n - m + 1):
        if framed_ids[i:i + m] == value_ids:
            return i, i + m - 1
    return None
