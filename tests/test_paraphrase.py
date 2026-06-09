"""Tests for the paraphrase-rich tool-calling data source.

Asserts the contracts the retrain code relies on: full tool coverage, literal-substring arg
grounding, and a TRUE train/eval split (disjoint in BOTH phrasings AND slot values).
"""

import json

from localagent.agent.toolset import STANDARD_TOOLS
from localagent.data.paraphrase import TOOL_EXAMPLES, paraphrase_samples

TOOL_NAMES = {t.name for t in STANDARD_TOOLS}


def test_covers_all_tools():
    samples = paraphrase_samples(n=3, seed=0, split="train")
    covered = {s.ref_name for s in samples}
    assert covered == TOOL_NAMES, f"missing: {TOOL_NAMES - covered}"


def test_balanced_across_tools():
    n = 4
    samples = paraphrase_samples(n=n, seed=1, split="train")
    assert len(samples) == n * len(TOOL_NAMES)
    counts: dict[str, int] = {}
    for s in samples:
        counts[s.ref_name] = counts.get(s.ref_name, 0) + 1
    assert all(c == n for c in counts.values())


def test_every_sample_is_a_tool_call():
    for split in ("train", "eval"):
        for s in paraphrase_samples(n=2, seed=2, split=split):
            assert s.kind == "tool"
            # target is canonical compact-JSON parseable back to name+arguments
            obj = json.loads(s.target)
            assert obj["name"] == s.ref_name
            assert json.loads(s.ref_args) == obj["arguments"]


def test_arg_values_are_literal_substrings_of_prompt():
    for split in ("train", "eval"):
        for s in paraphrase_samples(n=5, seed=3, split=split):
            args = json.loads(s.ref_args)
            for val in args.values():
                assert str(val) in s.prompt, (
                    f"{s.ref_name}: {val!r} not in prompt {s.prompt!r}")


def test_train_eval_phrasings_disjoint():
    # Use a large n so the template space is well-sampled, then compare the phrasing skeleton
    # (prompt with slot values masked out) per tool.
    def skeletons(split):
        sk: dict[str, set] = {}
        for s in paraphrase_samples(n=40, seed=7, split=split):
            args = json.loads(s.ref_args)
            masked = s.prompt
            for val in args.values():
                masked = masked.replace(str(val), "<V>")
            sk.setdefault(s.ref_name, set()).add(masked)
        return sk

    tr, ev = skeletons("train"), skeletons("eval")
    for name in TOOL_NAMES:
        overlap = tr[name] & ev[name]
        assert not overlap, f"{name} shares phrasings across split: {overlap}"


def test_train_eval_slot_values_disjoint():
    def values(split):
        v: dict[str, set] = {}
        for s in paraphrase_samples(n=60, seed=9, split=split):
            for val in json.loads(s.ref_args).values():
                v.setdefault(s.ref_name, set()).add(str(val))
        return v

    tr, ev = values("train"), values("eval")
    for name in TOOL_NAMES:
        # arg-less tools have no slot values; skip those (empty sets)
        if not tr.get(name) and not ev.get(name):
            continue
        assert not (tr[name] & ev[name]), f"{name} leaks slot values across split"


def test_tool_examples_covers_all_tools():
    assert set(TOOL_EXAMPLES.keys()) == TOOL_NAMES
    for name, examples in TOOL_EXAMPLES.items():
        assert examples, f"{name} has no example phrasings"
        assert all(isinstance(e, str) and e for e in examples)


def test_determinism():
    a = paraphrase_samples(n=3, seed=42, split="train")
    b = paraphrase_samples(n=3, seed=42, split="train")
    assert [s.prompt for s in a] == [s.prompt for s in b]
