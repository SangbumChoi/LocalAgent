import importlib.util
from pathlib import Path


_SCRIPT = Path(__file__).parents[1] / "scripts/native_browsergym_eval.py"
_SPEC = importlib.util.spec_from_file_location("native_browsergym_eval", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
_model_prompt = _MODULE._model_prompt
_browser_action = _MODULE._browser_action


def test_native_browsergym_prompt_matches_train_only_adapter_contract() -> None:
    prompt = _model_prompt({"goal": "Click the button", "axtree_object": {"nodes": []}})
    assert "Browser task: Click the button" in prompt
    assert "Live accessibility elements (quoted names are valid targets):" in prompt
    assert "Choose exactly one grounded computer action or abstain." in prompt


def test_realistic_browser_tools_map_grounded_ids_to_high_level_actions() -> None:
    observation = {
        "axtree_object": {
            "nodes": [
                {"browsergym_id": "b1", "role": {"value": "button"}, "name": {"value": "Send"}},
                {"browsergym_id": "t1", "role": {"value": "textbox"}, "name": {"value": "To"}},
                {"browsergym_id": "s1", "role": {"value": "combobox"}, "name": {"value": "Label"}},
            ]
        }
    }
    assert _browser_action("web_click", {"target_id": "b1"}, observation) == ("click('b1')", True)
    assert _browser_action("web_type", {"target_id": "t1", "text": "a@b.test"}, observation) == (
        "fill('t1', 'a@b.test')",
        True,
    )
    assert _browser_action("web_select", {"target_id": "s1", "value": "Work"}, observation) == (
        "select_option('s1', 'Work')",
        True,
    )
