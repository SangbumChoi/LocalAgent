"""The tiny decoder: pre-norm RMSNorm + GQA(RoPE) + SwiGLU, tied embeddings.

Implemented for real (Phase 1) so the from-scratch model is constructible and runs a forward
pass on CPU. KV-cached incremental decoding lives in inference/generate.py.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from localagent.model.config import ModelConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm.type_as(x) * self.weight


def build_rope_cache(seq_len: int, head_dim: int, theta: float, device, dtype):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)  # (seq_len, head_dim/2)
    return torch.cos(freqs).to(dtype), torch.sin(freqs).to(dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (B, n_heads, T, head_dim)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    cos = cos[None, None, : x.shape[2], :]
    sin = sin[None, None, : x.shape[2], :]
    rx1 = x1 * cos - x2 * sin
    rx2 = x1 * sin + x2 * cos
    return torch.stack((rx1, rx2), dim=-1).flatten(-2)


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads, self.n_kv_heads, self.hd = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
        self.rep = cfg.n_heads // cfg.n_kv_heads
        self.q = nn.Linear(cfg.d_model, cfg.n_heads * cfg.head_dim, bias=False)
        self.k = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.v = nn.Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.o = nn.Linear(cfg.n_heads * cfg.head_dim, cfg.d_model, bias=False)

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        q = self.q(x).view(B, T, self.n_heads, self.hd).transpose(1, 2)
        k = self.k(x).view(B, T, self.n_kv_heads, self.hd).transpose(1, 2)
        v = self.v(x).view(B, T, self.n_kv_heads, self.hd).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        k = k.repeat_interleave(self.rep, dim=1)  # GQA expand
        v = v.repeat_interleave(self.rep, dim=1)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).reshape(B, T, -1)
        return self.o(out)


class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.gate = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.up = nn.Linear(cfg.d_model, cfg.ffn_hidden, bias=False)
        self.down = nn.Linear(cfg.ffn_hidden, cfg.d_model, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.ffn_norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.ffn = SwiGLU(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.attn_norm(x), cos, sin)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class LocalAgentLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        cfg.assert_within_budget()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight
        self._rope = None
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def _rope_cache(self, device, dtype):
        if self._rope is None or self._rope[0].device != device:
            self._rope = build_rope_cache(
                self.cfg.max_seq_len, self.cfg.head_dim, self.cfg.rope_theta, device, dtype
            )
        return self._rope

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        x = self.embed(idx)
        cos, sin = self._rope_cache(x.device, x.dtype)
        for blk in self.blocks:
            x = blk(x, cos, sin)
        logits = self.lm_head(self.norm(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-100
            )
        return logits, loss

    def num_params(self) -> int:
        seen = set()
        total = 0
        for p in self.parameters():
            if id(p) not in seen:  # avoid double-counting tied weights
                seen.add(id(p))
                total += p.numel()
        return total
