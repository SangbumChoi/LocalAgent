"""Real public datasets for the GPU pipeline (pretrain text + function-calling SFT).

Import-guarded: needs `datasets` (present in the HF Jobs image, not in the CPU dev box). Used by
`scripts/train_job.py` when run with `--real`. Falls back to the synthetic generators otherwise.

  pretrain : HuggingFaceFW/fineweb-edu      (raw English text  -> byte stream)
  sft      : Salesforce/xlam-function-calling-60k (query + available tools -> tool call)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field


def _require_datasets():
    try:
        from datasets import load_dataset
        return load_dataset
    except ImportError as e:  # pragma: no cover - only in the dev box
        raise RuntimeError("pip install datasets (in the HF Jobs image) to use real data") from e


# ---- pretrain: FineWeb-edu -> byte stream ---------------------------------------------------
def fineweb_byte_stream(tok, max_chars: int = 50_000_000, dataset: str = "HuggingFaceFW/fineweb-edu",
                        name: str = "sample-10BT", log=print) -> list[int]:
    """Stream FineWeb-edu and return a flat byte-id stream (doc-separated by EOS) for `pretrain`."""
    load_dataset = _require_datasets()
    ds = load_dataset(dataset, name=name, split="train", streaming=True)
    stream: list[int] = []
    chars = 0
    for i, row in enumerate(ds):
        text = row.get("text") or ""
        stream.extend(tok.encode(text))
        stream.append(tok.eos_id)
        chars += len(text)
        if i % 2000 == 0:
            log(f"  [fineweb] {i} docs, {chars/1e6:.1f}M chars")
        if chars >= max_chars:
            break
    log(f"  [fineweb] done: {len(stream)/1e6:.1f}M byte tokens from {chars/1e6:.1f}M chars")
    return stream


# ---- SFT: xLAM function-calling -> in-context tool-call samples -----------------------------
@dataclass
class Row:
    """Minimal SFT sample compatible with render.render_sft / assistant_body."""
    prompt: str
    target: str                       # canonical compact JSON of the FIRST call
    kind: str = "tool"
    calls: list | None = None         # [{"name","arguments"}, ...] for multi-call turns
    ref_name: str = ""
    ref_args: str = "{}"
    category: str = "xlam"
    group: str = "tool"


def _tool_catalog(tools: list[dict]) -> str:
    """Compact `name(arg1, arg2) - description` catalog from xLAM tool dicts (in-context schemas)."""
    lines = []
    for t in tools:
        params = (t.get("parameters") or {})
        # xLAM params can be {name:{type,description}} or JSON-schema {properties:{...}}
        names = list((params.get("properties") or params).keys())
        lines.append(f"{t.get('name','')}({', '.join(names)}) - {t.get('description','')}".strip())
    return "\n".join(lines)


def xlam_sft_samples(tok, n: int = 60000, dataset: str = "Salesforce/xlam-function-calling-60k",
                     log=print) -> list[Row]:
    """Render xLAM rows as `<|tools|>{catalog}\n<|user|>{query}` -> first tool call (canonical JSON).
    Teaches generative, in-context function calling over arbitrary tool sets."""
    from localagent.agent.incontext import TOOLS_MARKER

    load_dataset = _require_datasets()
    ds = load_dataset(dataset, split="train")
    rows: list[Row] = []
    for r in ds:
        try:
            tools = json.loads(r["tools"]) if isinstance(r["tools"], str) else r["tools"]
            answers = json.loads(r["answers"]) if isinstance(r["answers"], str) else r["answers"]
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
        if not answers:
            continue
        prompt = f"{TOOLS_MARKER}\n{_tool_catalog(tools)}\n{r['query']}"
        calls = [{"name": a["name"], "arguments": a.get("arguments", {})} for a in answers]
        target = json.dumps({"name": calls[0]["name"], "arguments": calls[0]["arguments"]},
                            separators=(",", ":"), sort_keys=True)
        rows.append(Row(prompt=prompt, target=target,
                        calls=calls if len(calls) > 1 else None, ref_name=calls[0]["name"]))
        if len(rows) >= n:
            break
    log(f"  [xlam] {len(rows)} SFT samples")
    return rows
