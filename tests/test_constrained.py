"""Prompt-grounded constrained decoding: candidate proposal must ground slots in the prompt."""

from localagent.agent.constrained import candidates
from localagent.agent.toolset import STANDARD_TOOLS as TOOLS


def _bodies(prompt):
    return [b for b, _, _ in candidates(prompt, TOOLS)]


def test_weather_city_is_grounded_in_prompt():
    bodies = _bodies("What's the weather in Boston?")
    assert any('"city":"Boston"' in b and "get_weather" in b for b in bodies)


def test_calculator_expression_extracted():
    bodies = _bodies("What is 7 * 8?")
    assert any('"expression":"7*8"' in b for b in bodies)


def test_hello_is_text_only_not_planner():
    # "hello to X" must NOT fire the planner's " to " trigger.
    cands = candidates("Say hello to Zara.", TOOLS)
    assert all(not is_tool for _, is_tool, _ in cands)
    assert ("Hello, Zara!", False, "text") in cands


def test_thanks_abstains_with_text():
    cands = candidates("Thanks for your help!", TOOLS)
    assert all(not is_tool for _, is_tool, _ in cands)
