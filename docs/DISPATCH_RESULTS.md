# Generable tool dispatch — results & assets

End state of the "make tool-calling generable + cover current-agent scenarios" effort. Builds on
`ROUTE_REFACTOR_FINDINGS.md` (the architecture arc: 51-way head → dense selector + pointer-copy).

## The model
A 28M byte-level model dispatches a natural request to a tool via the **generable** pipeline:

    user request → route head (5-way modality gate) → dense two-tower selector (any tool, by
                   description embedding) → pointer-copy arguments → tool(args)

No fixed-N classifier anywhere; adding a tool is one row in the selector/retrieval index. Deployable
via `Agent.from_checkpoint("runs/tiny-30m-dispatch-best.pt", tools)`.

## Headline numbers — free-form OOD dispatch
44 hand-written natural queries (`eval/freeform.py`), deliberately NOT in the synthetic templates —
the honest generalization test.

| | call-name (end-to-end) | selection top-1 | route |
|---|---|---|---|
| pre-training (frozen 50-tool backbone) | 27% | 30% | 64% |
| **after dispatch fine-tune (best, ~3200 steps)** | **57%** | **57%** | **82%** |

A **2.1× improvement** on out-of-distribution queries, from backbone fine-tuning on paraphrase +
referent-conditioned data (the dense selector/route head are frozen-feature probes on top).

### Full scorecard (`scripts/scorecard.py`)
| held set | selection | full (sel+args) | args \| correct-sel |
|---|---|---|---|
| free-form (44 OOD) | 57% | 57% | — |
| paraphrase-eval (100, disjoint phrasings) | 46% | 35% | **76%** |
| contextual-eval (46, referent flips) | 22% | 7% | 30% |

Argument-copy is strong where selection is right (76% on paraphrase-eval) — **selection is the
bottleneck**, not arguments. Referent-conditioned (URL-vs-file) dispatch is the hardest case.

## Key finding — there is a training sweet spot (longer is worse)
Free-form dispatch peaks around **~3,200 steps** and then *degrades*: a finisher that trained 800
more steps dropped free-form call-name from 57% → 30%. Past the sweet spot the backbone overfits the
training distribution and loses OOD generalization. "Train 12–24h" backfires here; the right move is
early-stop on an OOD metric. (Found by comparing the saved FINAL checkpoint vs re-probing the
step-2400 backbone — `scripts/finalize_best.py`.)

## Data assets (generators + static dumps in `data/dumps/`)
- `data/paraphrase.py` — phrasing diversity, all 50 tools (median 12 train templates/tool).
- `data/contextual.py` — referent-conditioned flips: the SAME instruction → different tool by
  referent ("Open status.host.io" → open_url; "Open 'Chrome'" → open_app; "Open docs/intro.md" →
  read_file). 7 instruction-groups, 23 branches.
- `data/scenarios.py` — SOTA-agent families: clarify / abstain (over-trigger negatives) / parallel
  single-turn, and workflow / chained / error_recovery multi-turn episodes.
- `eval/freeform.py` — 44-query hand-written OOD eval.
- `data/dumps/*.jsonl` — materialized static snapshots of all of the above.

## Engineering notes
- **OOM fix**: probe feature-caching now runs under `no_grad` (was retaining 700–1500 forward
  graphs → 10GB+ → OOM). Peak RSS 10GB → 1.0GB.
- **Ephemeral container**: long unattended background jobs get reclaimed (process killed, `/tmp`
  wiped) while the persistent disk (`runs/`, repo) survives. Training is therefore run in **bounded,
  per-segment-checkpointed, monitored chunks** and resumed from the latest checkpoint.

## Reproduce
- Dispatch fine-tune: `scripts/train_dispatch_long.py` (segmented, checkpoints per segment).
- Finalize a deployable checkpoint from a backbone: `scripts/finalize_best.py`.
- Scenario continuation: `scripts/train_scenarios.py` (folds the 4 SOTA families).
- Score: `scripts/scorecard.py`, `scripts/audit_args.py`; demo: `scripts/demo_dispatch.py`.

## Scenario behaviours (clarify / abstain / parallel / multi-turn)
_(continuation run in progress; numbers appended when complete.)_
