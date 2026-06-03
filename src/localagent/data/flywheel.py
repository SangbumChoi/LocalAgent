"""Data flywheel (Phase 8) — learn from stored local conversations.

Loop (Airbnb Agent-in-the-Loop): ingest logged conversations + feedback -> mine good
trajectories -> dual-verify (reuse agent_synth.verify) -> append to the train pool ->
schedule a retrain/distill -> eval -> redeploy.
"""

from __future__ import annotations


def ingest(store_path: str):
    """Pull conversations + feedback signals from the conversation store."""
    raise NotImplementedError("TODO(phase-8): read SQLite store + AITL feedback")


def mine(conversations):
    """Select high-value trajectories (preference-positive, adopted, low-uncertainty)."""
    raise NotImplementedError("TODO(phase-8): mine candidates from logged usage")


def build_training_pool(out_jsonl: str) -> int:
    """Ingest -> mine -> verify -> append. Returns #samples added."""
    raise NotImplementedError("TODO(phase-8): close the loop into the SFT/distill pool")
