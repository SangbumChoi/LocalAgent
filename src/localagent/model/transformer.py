"""The tiny decoder: pre-norm RMSNorm + GQA(RoPE) + SwiGLU, tied factorized embeddings,
optional depth-recurrence, and a real KV cache for prefill + incremental decode.

Training/prefill path: full sequence, causal SDPA, pos=0, cache=None  (numerically vanilla).
Decode path:           one token at a time, attends to the cached K/V (state-of-the-art-ish
                       prefill-then-decode). With depth-recurrence each (loop, layer) pass keeps
                       its own cache slot.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from localagent.model.config import ModelConfig

# A KV cache is a list (one slot per loop×layer pass) of (k, v) tensors, or None entries.
KVCache = list


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
    # x: (B, n_heads, T, head_dim); cos/sin: (T, head_dim/2) already sliced to abs positions
    x1, x2 = x[..., 0::2], x[..., 1::2]
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
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

    def forward(self, x, cos, sin, cache=None):
        B, T, _ = x.shape
        q = self.q(x).view(B, T, self.n_heads, self.hd).transpose(1, 2)
        k = self.k(x).view(B, T, self.n_kv_heads, self.hd).transpose(1, 2)
        v = self.v(x).view(B, T, self.n_kv_heads, self.hd).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if cache is not None:
            pk, pv = cache
            k = torch.cat([pk, k], dim=2)
            v = torch.cat([pv, v], dim=2)
        new_cache = (k, v)
        k = k.repeat_interleave(self.rep, dim=1)  # GQA expand
        v = v.repeat_interleave(self.rep, dim=1)
        # causal only matters when q and k have equal length (prefill/training); a single decode
        # query (T==1) legitimately attends to all cached keys, so no causal mask there.
        causal = q.shape[2] == k.shape[2] and q.shape[2] > 1
        out = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
        out = out.transpose(1, 2).reshape(B, T, -1)
        return self.o(out), new_cache


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

    def forward(self, x, cos, sin, cache=None):
        a, new_cache = self.attn(self.attn_norm(x), cos, sin, cache)
        x = x + a
        x = x + self.ffn(self.ffn_norm(x))
        return x, new_cache


class LocalAgentLM(nn.Module):
    """Tiny decoder with optional factorized embeddings + depth-recurrence + KV cache."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        cfg.assert_within_budget()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.embed_dim)
        self.in_proj = nn.Linear(cfg.embed_dim, cfg.d_model, bias=False) if cfg.factorized else None
        self.out_proj = nn.Linear(cfg.d_model, cfg.embed_dim, bias=False) if cfg.factorized else None
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.loop_embed = (
            nn.Parameter(torch.zeros(cfg.n_loops, cfg.d_model)) if cfg.n_loops > 1 else None
        )
        self.norm = RMSNorm(cfg.d_model, cfg.norm_eps)
        self.lm_head = (
            None if cfg.tie_embeddings else nn.Linear(cfg.embed_dim, cfg.vocab_size, bias=False)
        )
        self._rope = None
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def _rope_slice(self, pos: int, T: int, device, dtype):
        if self._rope is None or self._rope[0].device != device or self._rope[0].dtype != dtype:
            self._rope = build_rope_cache(
                self.cfg.max_seq_len, self.cfg.head_dim, self.cfg.rope_theta, device, dtype
            )
        cos, sin = self._rope
        return cos[pos:pos + T], sin[pos:pos + T]

    def n_cache_slots(self) -> int:
        return self.cfg.n_loops * self.cfg.n_layers

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None,
                pos: int = 0, caches=None):
        """If `caches` is provided (a list of n_cache_slots entries), runs the cached path and
        returns (logits, loss, new_caches). Otherwise returns (logits, loss)."""
        return_cache = caches is not None
        if caches is None:
            caches = [None] * self.n_cache_slots()
        x = self.embed(idx)
        if self.in_proj is not None:
            x = self.in_proj(x)
        cos, sin = self._rope_slice(pos, x.shape[1], x.device, x.dtype)
        new_caches = [None] * self.n_cache_slots()
        slot = 0
        for loop in range(self.cfg.n_loops):
            if self.loop_embed is not None:
                x = x + self.loop_embed[loop]
            for blk in self.blocks:
                x, nc = blk(x, cos, sin, caches[slot])
                new_caches[slot] = nc
                slot += 1
        h = self.norm(x)
        if self.out_proj is not None:
            h = self.out_proj(h)
        logits = F.linear(h, self.embed.weight) if self.lm_head is None else self.lm_head(h)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100
            )
        if return_cache:
            return logits, loss, new_caches
        return logits, loss

    def num_params(self) -> int:
        seen, total = set(), 0
        for p in self.parameters():
            if id(p) not in seen:  # avoid double-counting tied weights
                seen.add(id(p))
                total += p.numel()
        return total
