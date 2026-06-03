"""Model configuration + the <100M-param budget guard."""

from __future__ import annotations

from dataclasses import dataclass

import yaml

PARAM_BUDGET = 100_000_000  # hard ceiling — configs must stay under this


@dataclass
class ModelConfig:
    name: str = "tiny-30m"
    vocab_size: int = 32000
    d_model: int = 384
    n_layers: int = 12
    n_heads: int = 6
    n_kv_heads: int = 2
    ffn_hidden: int = 1024
    max_seq_len: int = 2048
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    tie_embeddings: bool = True
    dropout: float = 0.0

    def __post_init__(self) -> None:
        assert self.d_model % self.n_heads == 0, "d_model must divide n_heads"
        assert self.n_heads % self.n_kv_heads == 0, "n_heads must be a multiple of n_kv_heads"

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @classmethod
    def from_yaml(cls, path: str) -> "ModelConfig":
        with open(path) as f:
            raw = yaml.safe_load(f)
        fields = {k: v for k, v in raw.items() if k in cls.__dataclass_fields__}
        return cls(**fields)

    def estimate_params(self) -> int:
        """Closed-form parameter estimate; used for the budget assertion and reporting."""
        d, h, kv, hd = self.d_model, self.n_heads, self.n_kv_heads, self.head_dim
        embed = self.vocab_size * d  # tied, counted once
        per_layer = (
            d * (h * hd)              # q proj
            + 2 * d * (kv * hd)       # k, v proj (GQA → smaller)
            + (h * hd) * d            # out proj
            + 3 * d * self.ffn_hidden  # SwiGLU: gate, up, down
            + 2 * d                   # two RMSNorm gains
        )
        final = d  # final norm
        head = 0 if self.tie_embeddings else self.vocab_size * d
        return embed + self.n_layers * per_layer + final + head

    def assert_within_budget(self) -> None:
        n = self.estimate_params()
        if n > PARAM_BUDGET:
            raise ValueError(
                f"Model '{self.name}' is ~{n/1e6:.1f}M params, over the {PARAM_BUDGET/1e6:.0f}M budget."
            )
