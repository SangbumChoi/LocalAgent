import hashlib
import json
from pathlib import Path

from localagent.data.stateful_productivity import (
    SUITE_ID,
    apply_action,
    build_tasks,
    canonical_json,
    conversation_for_task,
    initial_state,
    state_prompt,
    stateful_reward,
    stateful_reward_spec,
    suite_inventory,
    task_complete,
    task_prompts,
)
from localagent.data.schema import Role


def test_train_and_eval_tasks_are_slot_and_prompt_disjoint() -> None:
    train = build_tasks("train")
    evaluation = build_tasks("eval")
    train_text = "\n".join(task.goal + "\n" + "\n".join(task_prompts(task)) for task in train)
    eval_text = "\n".join(task.goal + "\n" + "\n".join(task_prompts(task)) for task in evaluation)
    assert train_text != eval_text
    assert "maya@example.com" in train_text
    assert "zoe@example.com" in eval_text
    assert "zoe@example.com" not in train_text
    assert "maya@example.com" not in eval_text


def test_email_notion_browser_and_recovery_transitions() -> None:
    for task in build_tasks("eval"):
        state = initial_state()
        for index, action in enumerate(task.actions):
            result = apply_action(task, index, state, action.tool, action.arguments)
            assert result.schema_valid
            assert result.exact_action
            assert result.state_transition, (task.task_id, index, result.error)
            state = result.state
        assert task_complete(task, state), task.task_id


def test_recovery_keeps_intermediate_error_and_abstention_is_noop() -> None:
    recovery = next(task for task in build_tasks("train") if task.family == "recovery")
    state = initial_state()
    first = apply_action(recovery, 0, state, "open_url", recovery.actions[0].arguments)
    assert first.state["browser"]["last_error"]
    assert first.state["browser"]["page"] is None
    abstain = next(task for task in build_tasks("train") if task.family == "abstention")
    result = apply_action(abstain, 0, initial_state(), None, {})
    assert result.closed_loop_success
    assert result.state == initial_state()
    assert stateful_reward(result, terminal=True) == 1.0
    assert sum(stateful_reward_spec().values()) == 1.0


def test_conversation_uses_canonical_schema_and_state_prompt() -> None:
    task = next(task for task in build_tasks("eval") if task.family == "notion")
    conversation = conversation_for_task(task)
    assert conversation.meta["suite"] == SUITE_ID
    assert conversation.messages[0].role == Role.user
    assert any(message.role == Role.tool for message in conversation.messages)
    prompt = state_prompt(task, 0, task.initial_state)
    assert canonical_json(task.initial_state) in prompt
    assert "notion_create_page" not in prompt
    assert len(conversation.tools) >= 60


def test_suite_inventory_is_balanced_and_prompt_free() -> None:
    inventory = suite_inventory("eval")
    assert inventory == {
        "suite": SUITE_ID,
        "split": "eval",
        "tasks": 5,
        "families": {"abstention": 1, "browser": 1, "email": 1, "notion": 1, "recovery": 1},
        "steps": 16,
        "recovery_tasks": 1,
        "abstention_tasks": 1,
        "conversation_rows": 5,
    }


def test_published_stateful_probe_receipt_is_self_hashed_and_negative() -> None:
    path = Path(__file__).parents[1] / "docs/paper/results/raw/m55-stateful-productivity-transfer-v1.json"
    receipt = json.loads(path.read_text(encoding="utf-8"))
    expected = receipt.pop("receipt_self_sha256")
    actual = hashlib.sha256(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert actual == expected
    assert receipt["source"]["public_benchmark_text_used"] is False
    assert receipt["arms"]["pretrained_frozen_backbone"]["closed_loop"]["task_complete_rate"] == 0.2
    assert receipt["arms"]["pretrained_frozen_backbone"]["closed_loop"]["recovery_task_complete_rate"] == 0.0


def test_published_lowrate_transfer_receipt_binds_all_three_arms() -> None:
    path = Path(__file__).parents[1] / "docs/paper/results/raw/m56-stateful-productivity-transfer-ablation-v1.json"
    receipt = json.loads(path.read_text(encoding="utf-8"))
    expected = receipt.pop("receipt_self_sha256")
    actual = hashlib.sha256(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert actual == expected
    assert set(receipt["arms"]) == {
        "pretrained_frozen_backbone",
        "pretrained_lowrate_unfrozen_backbone",
        "matched_random_backbone",
    }
    lowrate = receipt["arms"]["pretrained_lowrate_unfrozen_backbone"]
    assert lowrate["weight_movement"]["backbone"] > 0.0
    assert lowrate["closed_loop"]["task_complete_rate"] == 0.2
    assert receipt["source"]["native_runtime_executed"] is False
