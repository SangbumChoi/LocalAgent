"""Synthetic agent-data generation (Phase 3, implemented — self-contained, no teacher API).

Produces deterministic, canonical tool-calling + text samples across categories:
  weather / calc  -> tool-calling group
  web_search      -> web search
  planner         -> planner
  text / no_tool  -> text generation (incl. correct abstention, Hammer-style)

Determinism matters: each input maps to ONE canonical target string (compact, sorted-key JSON
for tool calls), so a tiny byte-level model can actually reproduce it exactly — that's what makes
~100% reachable and exactly-scorable.

Generalization, not memorization: train and eval draw slot values (cities, names, numbers, …)
from **disjoint** pools, so eval accuracy reflects copying/structure learned, not lookup.

The flywheel enriches by raising ``level`` (1..5): more templates, bigger slot pools, and a bit
more structural complexity (weather units, multi-term arithmetic, harder phrasings).
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass

# ---- slot pools, split train/eval (disjoint) ------------------------------------------------
CITIES_TRAIN = ["Paris", "Tokyo", "Berlin", "Cairo", "Lima", "Oslo", "Delhi", "Madrid",
                "Seoul", "Rome", "Dublin", "Vienna", "Athens", "Bogota", "Hanoi", "Accra"]
CITIES_EVAL = ["Boston", "Quito", "Kyoto", "Bern", "Tunis", "Riga", "Perth", "Nairobi"]
NAMES_TRAIN = ["Alice", "Bob", "Carol", "David", "Eve", "Frank", "Grace", "Heidi",
               "Ivan", "Judy", "Mallory", "Niaj", "Olivia", "Peggy", "Sybil", "Trent"]
NAMES_EVAL = ["Walter", "Xena", "Yuki", "Zara", "Quinn", "Rosa", " Feliks".strip(), "Mira"]
QUERIES_TRAIN = ["best pizza in town", "history of jazz", "how tall is Everest",
                 "python list comprehension", "tides tomorrow", "nearest pharmacy",
                 "speed of light", "who wrote Hamlet"]
QUERIES_EVAL = ["origin of tea", "rules of chess", "how rainbows form", "fastest land animal"]
GOALS_TRAIN = ["plan a trip to the coast", "organize a birthday party",
               "learn to bake bread", "set up a home garden", "prepare for an exam",
               "build a bookshelf", "train for a 5k", "write a short story"]
GOALS_EVAL = ["plan a museum visit", "start a podcast", "fix a leaky faucet", "learn guitar"]


@dataclass
class Sample:
    category: str          # weather | calc | web_search | planner | text | no_tool
    group: str             # tool_call | web_search | planner | text
    prompt: str            # user text (without framing markers)
    kind: str              # "tool" or "text"
    target: str            # canonical assistant body (no markers/EOS)
    ref_name: str = ""     # tool name (for tool kind)
    ref_args: str = ""     # canonical sorted-key JSON args string (for tool kind)


def _tool_target(name: str, arguments: dict) -> str:
    return json.dumps({"name": name, "arguments": arguments},
                      separators=(",", ":"), sort_keys=True)


class Generator:
    def __init__(self, level: int = 1, seed: int = 0, split: str = "train"):
        self.level = level
        self.split = split
        self.rng = random.Random(seed)
        self.cities = CITIES_TRAIN if split == "train" else CITIES_EVAL
        self.names = NAMES_TRAIN if split == "train" else NAMES_EVAL
        self.queries = QUERIES_TRAIN if split == "train" else QUERIES_EVAL
        self.goals = GOALS_TRAIN if split == "train" else GOALS_EVAL

    # ---- per-category sample makers ----
    def weather(self) -> Sample:
        city = self.rng.choice(self.cities)
        phr = self.rng.choice([
            f"What's the weather in {city}?",
            f"Tell me the weather for {city}.",
            f"How is the weather in {city} right now?",
        ] + ([f"Weather in {city} please."] if self.level >= 4 else []))
        args = {"city": city}
        if self.level >= 2 and self.rng.random() < 0.5:
            unit = self.rng.choice(["c", "f"])
            args["unit"] = unit
            phr += f" In {'Celsius' if unit == 'c' else 'Fahrenheit'}."
        return Sample("weather", "tool_call", phr, "tool",
                      _tool_target("get_weather", args), "get_weather",
                      json.dumps(args, separators=(",", ":"), sort_keys=True))

    def calc(self) -> Sample:
        a, b = self.rng.randint(1, 20), self.rng.randint(1, 20)
        op = self.rng.choice(["+", "-", "*"])
        if self.level >= 3 and self.rng.random() < 0.5:
            c = self.rng.randint(1, 20)
            op2 = self.rng.choice(["+", "-", "*"])
            expr = f"{a}{op}{b}{op2}{c}"
            q = f"What is {a} {op} {b} {op2} {c}?"
        else:
            expr = f"{a}{op}{b}"
            q = f"What is {a} {op} {b}?"
        args = {"expression": expr}
        return Sample("calc", "tool_call", q, "tool",
                      _tool_target("calculator", args), "calculator",
                      json.dumps(args, separators=(",", ":"), sort_keys=True))

    def web_search(self) -> Sample:
        query = self.rng.choice(self.queries)
        phr = self.rng.choice([
            f"Search the web for {query}.",
            f"Look up {query} online.",
            f"Find information about {query}.",
        ])
        args = {"query": query}
        return Sample("web_search", "web_search", phr, "tool",
                      _tool_target("web_search", args), "web_search",
                      json.dumps(args, separators=(",", ":"), sort_keys=True))

    def planner(self) -> Sample:
        goal = self.rng.choice(self.goals)
        phr = self.rng.choice([
            f"Help me {goal}.",
            f"Make a plan to {goal}.",
            f"I want to {goal}.",
        ])
        args = {"goal": goal}
        return Sample("planner", "planner", phr, "tool",
                      _tool_target("planner", args), "planner",
                      json.dumps(args, separators=(",", ":"), sort_keys=True))

    def text(self) -> Sample:
        name = self.rng.choice(self.names)
        choice = self.rng.choice(["hello", "morning", "name"])
        if choice == "hello":
            return Sample("text", "text", f"Say hello to {name}.", "text", f"Hello, {name}!")
        if choice == "morning":
            return Sample("text", "text", f"Greet {name} in the morning.", "text",
                          f"Good morning, {name}!")
        return Sample("text", "text", "What is your name?", "text", "I am LocalAgent.")

    def no_tool(self) -> Sample:
        # Abstention: a query that should NOT trigger a tool (Hammer irrelevance).
        return Sample("no_tool", "text", "Thanks for your help!", "text", "You're welcome!")

    # ---- dataset assembly ----
    def makers(self):
        m = [self.weather, self.calc, self.web_search, self.planner, self.text]
        if self.level >= 2:
            m.append(self.no_tool)
        return m

    def generate(self, n: int) -> list[Sample]:
        makers = self.makers()
        out, seen = [], set()
        # round-robin categories for balance; dedupe identical prompts
        guard = 0
        while len(out) < n and guard < n * 50:
            guard += 1
            s = makers[len(out) % len(makers)]()
            if s.prompt in seen:
                continue
            seen.add(s.prompt)
            out.append(s)
        return out


def synthesize(config_path: str) -> None:  # CLI entry retained
    raise NotImplementedError("Use scripts/flywheel.py — Generator drives data generation in-process")
