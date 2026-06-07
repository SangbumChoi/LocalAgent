"""Model configuration + the <100M-param budget guard.

Beyond a vanilla decoder, the config exposes two structural levers that make the
**ultra-tiny (~1M)** tier actually feasible (see docs/ARCHITECTURE_IDEAS.md):

  * ``embed_dim``  — factorized embeddings (ALBERT-style). At tiny scale the vocab×d_model
    table dominates params; we instead learn vocab×embed_dim and up-project embed_dim→d_model.
    Set ``embed_dim = d_model`` (or null in YAML) to disable factorization.
  * ``n_loops``    — depth-recurrence (Universal Transformer / ALBERT cross-layer sharing). The
    block stack is run ``n_loops`` times with shared weights, so *effective depth =
    n_layers × n_loops* at the param cost of ``n_layers`` blocks.

A byte-level vocab (``vocab_size: 256``) is the third lever — it removes the embedding tax
entirely, which is what lets a 1M model spend its budget on computation instead of the vocab.
"""

from __future__ import annotations

from dataclasses import dataclass

import yaml

PARAM_BUDGET = 100_000_000  # hard ceiling — configs must stay under this


@dataclass
class ModelConfig:
    name: str = "tiny-30m"
    vocab_size: int = 32000
    d_model: int = 384
    embed_dim: int | None = None  # None -> = d_model (no factorization)
    n_layers: int = 12            # number of blocks that hold parameters
    n_loops: int = 1              # depth-recurrence: run the stack this many times (shared weights)
    n_heads: int = 6
    n_kv_heads: int = 2
    ffn_hidden: int = 1024
    max_seq_len: int = 2048
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    tie_embeddings: bool = True
    dropout: float = 0.0
    # --- Hybrid backbone levers (LFM2/Nemotron-H + Qwen3 stability) ---
    # Defaults preserve the existing all-attention, QK-Norm-off model EXACTLY.
    qk_norm: bool = False                  # RMSNorm on Q/K per-head before RoPE (Qwen3/Kimi)
    conv_kernel: int = 3                   # depthwise causal short-conv kernel (LFM2 LIV)
    layer_types: list[str] | None = None   # per-layer block kind: "attn"|"conv"; None => all "attn"

    def __post_init__(self) -> None:
        if self.embed_dim is None:
            self.embed_dim = self.d_model
        assert self.d_model % self.n_heads == 0, "d_model must divide n_heads"
        assert self.n_heads % self.n_kv_heads == 0, "n_heads must be a multiple of n_kv_heads"
        assert self.n_loops >= 1, "n_loops must be >= 1"
        if self.layer_types is not None:
            assert len(self.layer_types) == self.n_layers, (
                f"layer_types has {len(self.layer_types)} entries, expected n_layers={self.n_layers}"
            )
            assert all(t in ("attn", "conv") for t in self.layer_types), (
                "layer_types entries must be 'attn' or 'conv'"
            )
            assert any(t == "attn" for t in self.layer_types), (
                "keep >=1 attention layer for verbatim argument-copying"
            )

    def block_types(self) -> list[str]:
        """Resolved per-layer block kinds; None => all attention (legacy behavior)."""
        return self.layer_types if self.layer_types is not None else ["attn"] * self.n_layers

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def factorized(self) -> bool:
        return self.embed_dim != self.d_model

    @property
    def effective_depth(self) -> int:
        return self.n_layers * self.n_loops

    @classmethod
    def from_yaml(cls, path: str) -> "ModelConfig":
        with open(path) as f:
            raw = yaml.safe_load(f)
        fields = {k: v for k, v in raw.items() if k in cls.__dataclass_fields__}
        return cls(**fields)

    def estimate_params(self) -> int:
        """Closed-form parameter estimate; used for the budget assertion and reporting."""
        d, h, kv, hd, k = self.d_model, self.n_heads, self.n_kv_heads, self.head_dim, self.embed_dim
        embed_table = self.vocab_size * k        # vocab × embed_dim (tied/shared)
        in_proj = k * d if self.factorized else 0
        out_proj = d * k if self.factorized else 0
        head = 0 if self.tie_embeddings else self.vocab_size * k
        loop_embed = self.n_loops * d if self.n_loops > 1 else 0
        ffn = 3 * d * self.ffn_hidden  # SwiGLU: gate, up, down (shared by both block kinds)

        def attn_mixer() -> int:
            qk = 2 * hd if self.qk_norm else 0  # RMSNorm(head_dim) gains on Q and K (shared/head)
            return (
                d * (h * hd)           # q proj
                + 2 * d * (kv * hd)    # k, v proj (GQA → smaller)
                + (h * hd) * d         # out proj
                + qk
            )

        def conv_mixer() -> int:
            # 3 in-projections (B, C, h̃) of width d, depthwise causal conv (k weights/channel),
            # and an out-projection. No bias.
            return 3 * d * d + d * self.conv_kernel + d * d

        total_blocks = 0
        for kind in self.block_types():
            mixer = conv_mixer() if kind == "conv" else attn_mixer()
            total_blocks += mixer + ffn + 2 * d  # + two RMSNorm gains (pre-mixer, pre-ffn)
        final = d  # final norm
        return embed_table + in_proj + out_proj + head + loop_embed + total_blocks + final

    def assert_within_budget(self) -> None:
        n = self.estimate_params()
        if n > PARAM_BUDGET:
            raise ValueError(
                f"Model '{self.name}' is ~{n/1e6:.1f}M params, over the {PARAM_BUDGET/1e6:.0f}M budget."
            )
