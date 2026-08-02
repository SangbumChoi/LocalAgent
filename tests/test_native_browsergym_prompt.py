import importlib.util
from pathlib import Path


_SCRIPT = Path(__file__).parents[1] / "scripts/native_browsergym_eval.py"
_SPEC = importlib.util.spec_from_file_location("native_browsergym_eval", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
_model_prompt = _MODULE._model_prompt


def test_native_browsergym_prompt_matches_train_only_adapter_contract() -> None:
    prompt = _model_prompt({"goal": "Click the button", "axtree_object": {"nodes": []}})
    assert "Browser task: Click the button" in prompt
    assert "Live accessibility elements (quoted names are valid targets):" in prompt
    assert "Choose exactly one grounded computer action or abstain." in prompt
