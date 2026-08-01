#!/usr/bin/env python
"""Train an additive standard+mobile dispatch selector on a verified mobile Conversation mix.

The LM backbone and legacy 50-tool head remain shape-compatible.  Only the route and dense
selector probes are continued, and the exported tool pool is explicitly recorded.  A deterministic
episode holdout reports selection and grounded-call metrics; it is not an Android emulator score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from localagent.agent.dense_selector import BoundSelector, DenseToolSelector, tool_embeddings
from localagent.agent.mobile_toolset import mobile_tools, realistic_productivity_tools
from localagent.agent.routes import ROUTE_INDEX, ROUTES, RouteHead, route_of
from localagent.agent.toolset import STANDARD_TOOLS
from localagent.data.agent_synth import Generator, Sample
from localagent.data.paraphrase import paraphrase_samples
from localagent.data.schema import Conversation, Role
from localagent.model import LocalAgentLM, ModelConfig
from localagent.model.tokenizer import load_tokenizer


def _sha256(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def _load_rows(paths: list[Path]) -> list[Conversation]:
    rows: list[Conversation] = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if line.strip():
                    try:
                        raw = json.loads(line)
                        conversation = Conversation.from_json(line)
                        metadata = raw.get("meta", {})
                        if isinstance(metadata, dict):
                            conversation.meta = dict(metadata)
                        conversation.meta.setdefault("record_id", raw.get("record_id", "unknown"))
                        rows.append(conversation)
                    except Exception as error:  # pragma: no cover - diagnostic context
                        raise ValueError(f"invalid Conversation at {path}:{line_number}") from error
    if not rows:
        raise ValueError("no conversations found")
    return rows


def _mobile_samples(rows: list[Conversation]) -> list[tuple[Sample, str]]:
    samples: list[tuple[Sample, str]] = []
    for row in rows:
        quality = row.meta.get("quality", {})
        episode_value = quality.get("source_episode_id") if isinstance(quality, dict) else None
        episode = str(episode_value if episode_value is not None else row.meta.get("parent_record_id", "unknown"))
        messages = row.messages
        for index, message in enumerate(messages):
            if message.role != Role.assistant or not message.tool_calls:
                continue
            call = message.tool_calls[0]
            if not call.name.startswith("mobile_"):
                continue
            prompt = ""
            for previous in reversed(messages[:index]):
                if previous.role == Role.user and previous.content:
                    prompt = previous.content
                    break
            if not prompt:
                continue
            sample = Sample(
                "mobile",
                "mobile",
                prompt,
                "tool",
                json.dumps(
                    {"arguments": call.arguments, "name": call.name},
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                call.name,
                json.dumps(call.arguments, sort_keys=True, separators=(",", ":")),
            )
            samples.append((sample, episode))
    if not samples:
        raise ValueError("no mobile tool turns found in the supplied Conversations")
    return samples


def _productivity_samples() -> list[Sample]:
    """Return deterministic full-field email/Notion prompts with a disjoint held-out tail."""

    rows = [
        ("Create a Notion page titled 'Sprint plan' with content 'Ship the browser pilot'.",
         "notion_create_page", {"title": "Sprint plan", "content": "Ship the browser pilot"}),
        ("Add a Notion page called 'Reading list' containing 'Review the agent papers'.",
         "notion_create_page", {"title": "Reading list", "content": "Review the agent papers"}),
        ("Save a Notion page titled 'Release notes' with content 'WebGPU fallback is documented'.",
         "notion_create_page", {"title": "Release notes", "content": "WebGPU fallback is documented"}),
        ("Create a Notion note named 'Meeting prep' containing 'Ask about emulator access'.",
         "notion_create_page", {"title": "Meeting prep", "content": "Ask about emulator access"}),
        ("Send an email to 'bob@example.com' with subject 'Build status' and body 'The tests pass'.",
         "email_send", {"to": "bob@example.com", "subject": "Build status", "body": "The tests pass"}),
        ("Email 'carol@example.com' about 'Dataset review' saying 'Please check the split'.",
         "email_send", {"to": "carol@example.com", "subject": "Dataset review", "body": "Please check the split"}),
        ("Compose a message to 'david@example.com' titled 'Receipt' with text 'The export is ready'.",
         "email_send", {"to": "david@example.com", "subject": "Receipt", "body": "The export is ready"}),
        ("Send 'eve@example.com' an email with subject 'WebGPU' and body 'The bundle is local'.",
         "email_send", {"to": "eve@example.com", "subject": "WebGPU", "body": "The bundle is local"}),
        ("Write a Notion page titled 'Training plan' with content 'Run the held-out evaluation'.",
         "notion_create_page", {"title": "Training plan", "content": "Run the held-out evaluation"}),
        ("Create a Notion page called 'Mobile notes' containing 'Keep the screen text compact'.",
         "notion_create_page", {"title": "Mobile notes", "content": "Keep the screen text compact"}),
        ("Email 'frank@example.com' with subject 'Pilot' and body 'The mobile loop passed'.",
         "email_send", {"to": "frank@example.com", "subject": "Pilot", "body": "The mobile loop passed"}),
        ("Send a message to 'grace@example.com' titled 'Next step' saying 'Run the real emulator'.",
         "email_send", {"to": "grace@example.com", "subject": "Next step", "body": "Run the real emulator"}),
        # Held-out rows: values and phrasings do not occur in the training prefix.
        ("Make a Notion page titled 'Workshop checklist' whose content is 'Verify every receipt'.",
         "notion_create_page", {"title": "Workshop checklist", "content": "Verify every receipt"}),
        ("Add a Notion page named 'Ablations' with the text 'Compare no-guard dispatch'.",
         "notion_create_page", {"title": "Ablations", "content": "Compare no-guard dispatch"}),
        ("Send an email to 'heidi@example.com' with subject 'Acceptance' and body 'Please review the gate'.",
         "email_send", {"to": "heidi@example.com", "subject": "Acceptance", "body": "Please review the gate"}),
        ("Write to 'ivan@example.com' with the subject 'Mobile result' and message 'Selector accuracy is pending'.",
         "email_send", {"to": "ivan@example.com", "subject": "Mobile result", "body": "Selector accuracy is pending"}),
    ]
    samples = []
    for prompt, name, arguments in rows:
        encoded_args = json.dumps(arguments, sort_keys=True, separators=(",", ":"))
        samples.append(
            Sample(
                "realistic_productivity",
                "app_action",
                prompt,
                "tool",
                json.dumps({"arguments": arguments, "name": name}, sort_keys=True, separators=(",", ":")),
                name,
                encoded_args,
            )
        )
    return samples


def _compact_mobile_samples(mobile: list[tuple[Sample, str]]) -> list[Sample]:
    """Add instruction-only views without changing the held-out raw screen observations."""

    compact: list[Sample] = []
    for sample, _episode in mobile:
        marker = " instruction:"
        prompt = sample.prompt.split(marker, 1)[-1].strip() if marker in sample.prompt else sample.prompt
        if prompt == sample.prompt:
            continue
        compact.append(
            Sample(
                sample.category,
                sample.group,
                prompt,
                sample.kind,
                sample.target,
                sample.ref_name,
                sample.ref_args,
                sample.calls,
            )
        )
    return compact


def _feature(model: LocalAgentLM, tok, prompt: str, device: str) -> torch.Tensor:
    from localagent.agent.tool_head import _feat

    return _feat(model, tok, prompt, device)


def _checkpoint_tokenizer(parent: dict):
    """Load the tokenizer recorded by the checkpoint, never silently falling back to bytes."""

    metadata = parent.get("tokenizer") or {"kind": "byte"}
    kind = str(metadata.get("kind", "byte")) if isinstance(metadata, dict) else "byte"
    path = metadata.get("path") if isinstance(metadata, dict) else None
    if kind == "bpe" and path is None:
        raise ValueError("BPE checkpoint is missing tokenizer.path")
    return load_tokenizer(kind, path)


def _train_probe(
    model: LocalAgentLM,
    tok,
    parent: dict,
    tools,
    mobile: list[tuple[Sample, str]],
    productivity: list[Sample],
    *,
    steps: int,
    device: str,
    seed: int,
) -> tuple[RouteHead, DenseToolSelector, dict[str, list[str]]]:
    random.seed(seed)
    torch.manual_seed(seed)
    standard = Generator(level=3, seed=seed, split="train").generate_balanced(3)
    standard += paraphrase_samples(2, seed=seed, split="train")
    mobile_samples = [sample for sample, _ in mobile]
    compact_mobile = _compact_mobile_samples(mobile)
    # Keep the broad standard catalog, but repeat the two deployment-critical productivity tools
    # enough that they are not drowned out by 50 legacy tools and 100+ mobile turns.
    productivity_augmented = productivity * 4
    pool = standard + mobile_samples + compact_mobile + productivity_augmented
    examples: dict[str, list[str]] = {
        name: list(values) for name, values in (parent.get("examples") or {}).items()
    }
    for sample in mobile_samples:
        examples.setdefault(sample.ref_name, []).append(sample.prompt)
    for sample in compact_mobile:
        examples.setdefault(sample.ref_name, []).append(sample.prompt)
    for sample in productivity_augmented:
        examples.setdefault(sample.ref_name, []).append(sample.prompt)
    route_rows = [(sample, route_of(sample.ref_name)) for sample in pool]
    with torch.no_grad():
        route_features = torch.stack([_feature(model, tok, sample.prompt, device) for sample, _ in route_rows])
    route_labels = torch.tensor([ROUTE_INDEX[label] for _, label in route_rows], device=device)
    route = RouteHead(model.cfg.d_model).to(device)
    if parent.get("route_head"):
        route.load_state_dict(parent["route_head"])
    route_opt = torch.optim.AdamW(route.parameters(), lr=5e-3)

    name_to_index = {tool.name: index for index, tool in enumerate(tools)}
    selector_rows = [(sample, name_to_index[sample.ref_name]) for sample in pool if sample.ref_name in name_to_index]
    with torch.no_grad():
        selector_features = torch.stack(
            [_feature(model, tok, sample.prompt, device) for sample, _ in selector_rows]
        )
    selector_labels = torch.tensor([label for _, label in selector_rows], device=device)
    embs = tool_embeddings(tools, device=device, examples=examples)
    selector = DenseToolSelector(
        model.cfg.d_model,
        emb_dim=embs.shape[1],
        proj=int(parent.get("selector_proj", 256)),
    ).to(device)
    if parent.get("dense_selector"):
        selector.load_state_dict(parent["dense_selector"])
    selector_opt = torch.optim.AdamW(selector.parameters(), lr=5e-3)
    rng = random.Random(seed)
    for step in range(max(1, steps)):
        route_idx = torch.tensor([rng.randrange(len(route_rows)) for _ in range(min(64, len(route_rows)))])
        route_loss = F.cross_entropy(route(route_features[route_idx]), route_labels[route_idx])
        route_opt.zero_grad(set_to_none=True)
        route_loss.backward()
        route_opt.step()

        selector_idx = torch.tensor(
            [rng.randrange(len(selector_rows)) for _ in range(min(64, len(selector_rows)))]
        )
        selector_loss = F.cross_entropy(
            selector(selector_features[selector_idx], embs), selector_labels[selector_idx]
        )
        selector_opt.zero_grad(set_to_none=True)
        selector_loss.backward()
        selector_opt.step()
    route.eval()
    selector.eval()
    return route, selector, examples


@torch.no_grad()
def _score(
    model: LocalAgentLM,
    tok,
    route: RouteHead,
    selector: BoundSelector,
    rows: list[tuple[Sample, str]],
    device: str,
) -> dict[str, float | int]:
    route_ok = selector_ok = 0
    for sample, _ in rows:
        feature = _feature(model, tok, sample.prompt, device)
        route_ok += ROUTES[int(route(feature).argmax(-1))] == route_of(sample.ref_name)
        selector_ok += selector.rank(feature)[0] == sample.ref_name
    total = len(rows)
    return {
        "rows": total,
        "route_accuracy": route_ok / max(1, total),
        "selector_top1": selector_ok / max(1, total),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, action="append", required=True)
    parser.add_argument("--init", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.steps < 1:
        raise ValueError("steps must be positive")

    parent = torch.load(args.init, map_location="cpu")
    cfg = ModelConfig(**parent["cfg"])
    model = LocalAgentLM(cfg).to(args.device).eval()
    model.load_state_dict(parent["state_dict"])
    tok = _checkpoint_tokenizer(parent)
    rows = _load_rows(args.data)
    mobile = _mobile_samples(rows)
    episode_ids = sorted({episode for _, episode in mobile})
    holdout_ids = set(episode_ids[-max(1, len(episode_ids) // 5) :])
    train_mobile = [item for item in mobile if item[1] not in holdout_ids]
    held_mobile = [item for item in mobile if item[1] in holdout_ids]
    tools = list(STANDARD_TOOLS) + mobile_tools() + realistic_productivity_tools()
    productivity = _productivity_samples()
    productivity_train = productivity[:-4]
    productivity_holdout = productivity[-4:]
    route, selector, examples = _train_probe(
        model,
        tok,
        parent,
        tools,
        train_mobile,
        productivity_train,
        steps=args.steps,
        device=args.device,
        seed=2027,
    )
    bound = BoundSelector(selector, tools, device=args.device, examples=examples)
    train_metrics = _score(
        model,
        tok,
        route,
        bound,
        train_mobile + [(sample, "productivity-train") for sample in productivity_train],
        args.device,
    )
    held_metrics = _score(model, tok, route, bound, held_mobile, args.device)
    held_productivity_metrics = _score(
        model,
        tok,
        route,
        bound,
        [(sample, "productivity-held") for sample in productivity_holdout],
        args.device,
    )

    child = dict(parent)
    child.update(
        {
            "stage": "sft_realistic_mobile_dispatch_pilot",
            "parent_checkpoint_sha256": _sha256(args.init)[1],
            "route_head": route.state_dict(),
            "dense_selector": selector.state_dict(),
            "selector_proj": int(parent.get("selector_proj", 256)),
            "examples": examples,
            "dispatch_tool_pool": [tool.name for tool in tools],
            "mobile_dispatch_training": {
                "rows": len(mobile),
                "episodes": episode_ids,
                "held_out_episodes": sorted(holdout_ids),
                "steps": args.steps,
                "train": train_metrics,
                "held_out": held_metrics,
                "productivity_train": _score(
                    model,
                    tok,
                    route,
                    bound,
                    [(sample, "productivity-train") for sample in productivity_train],
                    args.device,
                ),
                "productivity_held_out": held_productivity_metrics,
            },
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(child, args.output)
    report = {
        "kind": "localagent_realistic_mobile_dispatch_training_report",
        "parent": {"path": str(args.init), "sha256": _sha256(args.init)[1]},
        "child": {"path": str(args.output), "sha256": _sha256(args.output)[1]},
        "data": [{"path": str(path), "bytes": _sha256(path)[0], "sha256": _sha256(path)[1]} for path in args.data],
        "tool_pool": [tool.name for tool in tools],
        "mobile_dispatch_training": child["mobile_dispatch_training"],
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
