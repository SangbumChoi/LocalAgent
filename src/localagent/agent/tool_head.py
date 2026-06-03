"""Tool-selection head (ARCHITECTURE_IDEAS §2a, dual-head — tool side).

Generic argument *grounding* works (agent/constrained.py), but a ~1M byte model's intrinsic tool
*selection* is weak — per-tool trigger phrases used to mask that. The fix is a tiny linear
classifier over the prompt's final hidden state that picks the tool (or "text"), trained with a
clean label signal. Cheap to train (a linear probe on frozen features) and accurate.

Selection (this head) + grounding (constrained.py) = a trigger-free, schema-driven decoder.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from localagent.model.tokenizer import ASSISTANT, USER

CLASSES = ["get_weather", "calculator", "web_search", "planner", "text"]


def label_of(sample) -> str:
    return sample.ref_name if sample.kind == "tool" else "text"


@torch.no_grad()
def _feat(model, tok, prompt: str, device) -> torch.Tensor:
    ids = torch.tensor([tok.encode(f"{USER}{prompt}{ASSISTANT}")], dtype=torch.long, device=device)
    _, feats = model(ids, return_hidden=True)
    return feats[0, -1]  # final prompt-token features (d_model)


class ToolHead(nn.Module):
    def __init__(self, d_model: int, classes: list[str] = CLASSES):
        super().__init__()
        self.classes = classes
        self.fc = nn.Linear(d_model, len(classes))

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.fc(feat)

    def predict(self, model, tok, prompt: str, device="cpu") -> str:
        feat = _feat(model, tok, prompt, device)
        return self.classes[int(self(feat).argmax(-1))]


def train_tool_head(model, samples, tok, *, steps=300, batch_size=64, lr=5e-3, device="cpu",
                    log=lambda *a: None) -> ToolHead:
    model.eval()
    head = ToolHead(model.cfg.d_model).to(device)
    # cache frozen features + labels once (linear probe)
    feats = torch.stack([_feat(model, tok, s.prompt, device) for s in samples])
    labels = torch.tensor([CLASSES.index(label_of(s)) for s in samples], device=device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr)
    import random
    rng = random.Random(0)
    n = len(samples)
    for step in range(steps):
        idx = torch.tensor([rng.randrange(n) for _ in range(batch_size)], device=device)
        loss = F.cross_entropy(head(feats[idx]), labels[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step % max(1, steps // 5) == 0 or step == steps - 1:
            acc = (head(feats).argmax(-1) == labels).float().mean().item()
            log(f"  [tool-head] step {step}/{steps} loss {loss.item():.3f} train-acc {acc:.3f}")
    return head
