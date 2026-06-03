"""Pretrain corpus: download -> quality filter -> tokenize -> pack into shards (Phase 2).

Data-centric (SmolLM2-style): prefer quality-filtered/structured text over raw web scrape.
Output is a directory of token shards (uint16 .npy) that the pretrain loop memory-maps.
"""

from __future__ import annotations


def download_sample(out_dir: str) -> None:
    """Fetch a small public text sample for toy/speedrun runs."""
    raise NotImplementedError("TODO(phase-2): download a small public corpus sample")


def quality_filter(docs):
    """Heuristic + classifier quality filtering of raw docs."""
    raise NotImplementedError("TODO(phase-2): length/lang/dedup/quality filters")


def pack_shards(docs, tokenizer, seq_len: int, shards_dir: str) -> None:
    """Tokenize, concatenate, and pack into fixed-length training windows."""
    raise NotImplementedError("TODO(phase-2): tokenize + pack into uint16 shards")
