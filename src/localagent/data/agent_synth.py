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

from localagent.data.schema import Conversation, Message, Role, ToolCall

# ---- slot pools, split train/eval (disjoint) ------------------------------------------------
CITIES_TRAIN = ["Paris", "Tokyo", "Berlin", "Cairo", "Lima", "Oslo", "Delhi", "Madrid",
                "Seoul", "Rome", "Dublin", "Vienna", "Athens", "Bogota", "Hanoi", "Accra",
                "Lisbon", "Prague", "Warsaw", "Helsinki", "Manila", "Jakarta", "Amman", "Doha",
                "Brussels", "Stockholm", "Budapest", "Belgrade", "Tbilisi", "Baku", "Tehran",
                "Karachi", "Lagos", "Nairobi", "Bangkok", "Taipei", "Osaka", "Munich", "Naples",
                "Porto", "Valencia", "Glasgow", "Leeds", "Ottawa", "Calgary", "Denver", "Austin",
                "Seattle", "Portland", "Phoenix", "Dallas", "Atlanta", "Miami", "Houston",
                "Toronto", "Montreal", "Santiago", "Brasilia", "Quebec", "Medellin", "Panama"]
CITIES_EVAL = ["Boston", "Quito", "Kyoto", "Bern", "Tunis", "Riga", "Perth", "Nairobi",
               "Caracas", "Bruges", "Cusco", "Almaty", "Reykjavik", "Tallinn", "Vilnius",
               "Sarajevo", "Tirana", "Maputo", "Kigali", "Dakar"]
NAMES_TRAIN = ["Alice", "Bob", "Carol", "David", "Eve", "Frank", "Grace", "Heidi",
               "Ivan", "Judy", "Mallory", "Niaj", "Olivia", "Peggy", "Sybil", "Trent",
               "Karl", "Nora", "Pablo", "Ruth", "Vince", "Wendy", "Hugo", "Iris",
               "Aaron", "Bianca", "Cedric", "Dora", "Elliot", "Fiona", "George", "Hannah",
               "Isaac", "Jasmine", "Kevin", "Laura", "Marcus", "Nadia", "Oscar", "Priya",
               "Quentin", "Rachel", "Samuel", "Tara", "Ulises", "Vera", "Wesley", "Yara"]
NAMES_EVAL = ["Walter", "Xena", "Yuki", "Zara", "Quinn", "Rosa", "Feliks", "Mira",
              "Otto", "Greta", "Diego", "Lena", "Soren", "Ingrid", "Mateo", "Noor"]
QUERIES_TRAIN = ["best pizza in town", "history of jazz", "how tall is Everest",
                 "python list comprehension", "tides tomorrow", "nearest pharmacy",
                 "speed of light", "who wrote Hamlet", "capital of Peru", "boiling point of water",
                 "longest river in Asia", "how to tie a tie", "weather on Mars",
                 "population of Canada", "how do magnets work", "best running shoes",
                 "cheapest flights to Rome", "symptoms of the flu", "how to make sourdough",
                 "current bitcoin price", "rules of cricket", "tallest building in the world",
                 "how planes fly", "history of the internet", "best sci-fi movies"]
QUERIES_EVAL = ["origin of tea", "rules of chess", "how rainbows form", "fastest land animal",
                "phases of the moon", "history of the violin", "how vaccines work",
                "deepest part of the ocean", "who invented radio", "best hiking trails"]
GOALS_TRAIN = ["plan a trip to the coast", "organize a birthday party", "learn to bake bread",
               "set up a home garden", "prepare for an exam", "build a bookshelf",
               "train for a 5k", "write a short story", "redecorate the living room",
               "save for a vacation", "declutter the garage", "start a vegetable patch",
               "learn to paint", "plan a wedding", "build a website", "run a marathon",
               "read more books", "cook healthier meals", "learn to swim", "start journaling"]
GOALS_EVAL = ["plan a museum visit", "start a podcast", "fix a leaky faucet", "learn guitar",
              "host a dinner party", "switch careers", "adopt a puppy", "renovate the kitchen"]
TERMS_TRAIN = ["photosynthesis", "inflation", "entropy", "recursion", "osmosis", "democracy",
               "gravity", "metaphor", "algorithm", "ecosystem", "capitalism", "evolution",
               "momentum", "encryption", "diffusion", "polymer", "tectonics", "antibody",
               "induction", "sarcasm", "monopoly", "velocity", "habitat", "irony"]
TERMS_EVAL = ["mitosis", "diplomacy", "friction", "syntax", "biome", "catalyst", "satire",
              "inertia", "symbiosis", "federalism"]
SONGS_TRAIN = ["Bohemian Rhapsody", "Hey Jude", "Yesterday", "Imagine", "Hallelujah",
               "Thunderstruck", "Clocks", "Africa", "Viva La Vida", "Wonderwall", "Stairway",
               "Billie Jean", "Rolling Stone", "Sweet Child", "Born To Run",
               "Dancing Queen", "Mr Brightside", "Take On Me", "Losing My Religion",
               "Black Dog", "Free Bird", "Brown Eyed Girl", "Tiny Dancer"]
SONGS_EVAL = ["Let It Be", "Smooth Criminal", "Sweet Caroline", "Purple Rain", "Hotel California",
              "Comfortably Numb", "Karma Police", "Paint It Black"]
TOPICS_TRAIN = ["the economy", "space exploration", "local elections", "climate policy",
                "technology", "the stock market", "public health", "renewable energy",
                "the labor market", "global trade", "education reform", "cybersecurity",
                "the music industry", "professional sports", "electric vehicles", "data privacy"]
TOPICS_EVAL = ["artificial intelligence", "the housing market", "ocean conservation", "world cup",
               "gene editing", "supply chains", "the gig economy", "quantum computing"]
# --- coding-agent surface (Claude Code / Codex-style tools) ---
PATHS_TRAIN = ["src/main.py", "utils/io.py", "app/server.js", "lib/parse.py", "tests/test_api.py",
               "README.md", "config/settings.yaml", "core/model.py", "src/train.py", "db/schema.sql",
               "src/utils/log.py", "api/handlers.go", "web/app.tsx", "scripts/deploy.sh",
               "models/encoder.py", "tests/test_db.py", "pkg/cache.rs", "cmd/root.go",
               "frontend/index.js", "data/clean.py", "services/auth.py", "internal/queue.go",
               "docs/setup.md", "build/make.sh", "ops/deploy.yaml", "src/cli.py", "lib/http_client.py"]
PATHS_EVAL = ["data/loader.py", "web/index.html", "bin/run.sh", "docs/guide.md", "api/routes.go",
              "src/router.ts", "tests/test_cli.py", "pkg/store.rs", "config/prod.yaml",
              "services/email.py"]
PATTERNS_TRAIN = ["TODO", "def main", "import os", "class Model", "async def", "API_KEY",
                  "NotImplementedError", "print(", "import torch", "FIXME", "return self",
                  "def __init__", "raise RuntimeError", "logging.info", "os.environ",
                  "if __name__", "await fetch", "useState", "SELECT *", "panic("]
PATTERNS_EVAL = ["def run", "return None", "raise ValueError", "self.cfg", "import json",
                 "class Config", "try:", "console.log"]
COMMANDS_TRAIN = ["ls -la", "npm install", "pip install torch", "docker build .", "make test",
                  "git status", "python -m pytest", "cargo run", "git diff", "npm run build",
                  "kubectl get pods", "terraform plan", "go test ./...", "ruff check src",
                  "docker compose up", "git log --oneline", "pip freeze", "yarn dev"]
COMMANDS_EVAL = ["git pull", "cargo build", "python app.py", "npm run dev", "make lint",
                 "docker ps", "go build ./...", "pytest -q"]
COMMITS_TRAIN = ["fix bug", "add tests", "update docs", "refactor parser", "bump version",
                 "improve logging", "fix typo", "add validation", "speed up query",
                 "drop unused deps", "handle timeout", "rename module", "add CI step"]
COMMITS_EVAL = ["handle edge case", "remove dead code", "tidy imports", "fix race condition",
                "add retry logic", "clarify error message"]
TASKS_TRAIN = ["call the dentist", "buy groceries", "submit the report", "back up the laptop",
               "email the team", "renew the subscription", "pay the rent", "schedule a checkup",
               "reply to Sam", "order more coffee", "review the PR", "update the resume",
               "cancel the trial", "charge the camera"]
TASKS_EVAL = ["water the plants", "renew the license", "book a flight", "return the package",
              "refill the prescription", "call the landlord"]
DURATIONS_TRAIN = ["10 minutes", "1 hour", "30 seconds", "20 minutes", "2 hours", "15 minutes",
                   "90 seconds", "4 hours", "25 minutes", "12 minutes", "6 hours", "40 seconds"]
DURATIONS_EVAL = ["5 minutes", "45 seconds", "3 hours", "8 minutes", "75 minutes"]
# --- computer-use / productivity surface (calendar, email, browser, Notion, Slack, Jira) ---
URLS_TRAIN = ["example.com", "github.com", "wikipedia.org", "stackoverflow.com", "nytimes.com",
              "python.org", "arxiv.org", "docs.google.com", "amazon.com", "youtube.com",
              "linkedin.com", "medium.com", "bbc.com", "cnn.com", "apple.com", "microsoft.com",
              "gitlab.com", "npmjs.com", "pypi.org", "wikipedia.de"]
URLS_EVAL = ["reddit.com", "figma.com", "openai.com", "huggingface.co", "spotify.com",
             "dropbox.com", "twitch.tv", "notion.so"]
EVENT_TITLES_TRAIN = ["Team sync", "Standup", "Design review", "Budget meeting", "Onboarding",
                      "Sprint review", "Project kickoff", "Retro", "All hands", "Lunch break",
                      "Code freeze", "Release call", "Customer demo", "Strategy session",
                      "Board meeting", "Coffee chat"]
EVENT_TITLES_EVAL = ["Demo day", "Planning", "Interview", "Sync up", "Town hall", "Roadmap review"]
NOTION_TRAIN = ["meeting notes", "weekly goals", "project ideas", "reading list", "action items",
                "release plan", "design spec", "team updates", "research summary", "launch checklist",
                "interview feedback", "quarterly review"]
NOTION_EVAL = ["bug triage", "roadmap draft", "retro notes", "onboarding guide", "budget plan"]
SLACK_TRAIN = ["deploy is done", "standup in 5", "PR is ready", "build passed", "reviewing now",
               "merging soon", "tests are green", "heading out", "back online", "looking into it",
               "good morning team", "rolling back"]
SLACK_EVAL = ["ship it", "need help", "on my way", "almost there", "taking a break"]
JIRA_TRAIN = ["login bug", "slow query", "add dark mode", "fix typo", "memory leak", "flaky test",
              "improve search", "add pagination", "fix crash", "update schema", "rate limiting",
              "export to csv", "mobile layout", "session timeout"]
JIRA_EVAL = ["broken link", "update deps", "cache miss", "form validation", "data loss", "404 page"]


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
        tr = split == "train"
        self.cities = CITIES_TRAIN if tr else CITIES_EVAL
        self.names = NAMES_TRAIN if tr else NAMES_EVAL
        self.queries = QUERIES_TRAIN if tr else QUERIES_EVAL
        self.goals = GOALS_TRAIN if tr else GOALS_EVAL
        self.terms = TERMS_TRAIN if tr else TERMS_EVAL
        self.songs = SONGS_TRAIN if tr else SONGS_EVAL
        self.topics = TOPICS_TRAIN if tr else TOPICS_EVAL
        self.paths = PATHS_TRAIN if tr else PATHS_EVAL
        self.patterns = PATTERNS_TRAIN if tr else PATTERNS_EVAL
        self.commands = COMMANDS_TRAIN if tr else COMMANDS_EVAL
        self.commits = COMMITS_TRAIN if tr else COMMITS_EVAL
        self.tasks = TASKS_TRAIN if tr else TASKS_EVAL
        self.durations = DURATIONS_TRAIN if tr else DURATIONS_EVAL
        self.urls = URLS_TRAIN if tr else URLS_EVAL
        self.events = EVENT_TITLES_TRAIN if tr else EVENT_TITLES_EVAL
        self.notion = NOTION_TRAIN if tr else NOTION_EVAL
        self.slack = SLACK_TRAIN if tr else SLACK_EVAL
        self.jira = JIRA_TRAIN if tr else JIRA_EVAL

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
            f"Can you look up {query}?",
            f"I'm searching for {query}.",
            f"Search for {query}.",
        ])
        args = {"query": query}
        return Sample("web_search", "web_search", phr, "tool",
                      _tool_target("web_search", args), "web_search",
                      json.dumps(args, separators=(",", ":"), sort_keys=True))

    def planner(self) -> Sample:
        goal = self.rng.choice(self.goals)
        phr = self.rng.choice([
            f"Make a plan to {goal}.",
            f"I want to {goal}.",
            f"I need to {goal}.",
            f"My goal is to {goal}.",
            f"Create a plan to {goal}.",
            f"Plan how to {goal}.",
        ])
        args = {"goal": goal}
        return Sample("planner", "planner", phr, "tool",
                      _tool_target("planner", args), "planner",
                      json.dumps(args, separators=(",", ":"), sort_keys=True))

    def _string_tool(self, category, group, name, arg, value, phrasings) -> Sample:
        args = {arg: value}
        return Sample(category, group, self.rng.choice(phrasings), "tool",
                      _tool_target(name, args), name,
                      json.dumps(args, separators=(",", ":"), sort_keys=True))

    def define(self) -> Sample:
        t = self.rng.choice(self.terms)
        return self._string_tool("define", "define", "define", "term", t,
                                 [f"Definition of {t}.", f"Define {t}.", f"Explain {t}.",
                                  f"Tell me about {t}.", f"Describe {t}.",
                                  f"Give me the definition of {t}."])

    def play_music(self) -> Sample:
        s = self.rng.choice(self.songs)
        return self._string_tool("play_music", "music", "play_music", "song", s,
                                 [f"Play {s}.", f"Put on {s}.", f"Start playing {s}.",
                                  f"Queue up {s}.", f"I want to hear {s}.", f"Play the song {s}."])

    def get_news(self) -> Sample:
        t = self.rng.choice(self.topics)
        return self._string_tool("get_news", "news", "get_news", "topic", t,
                                 [f"Show the news about {t}.", f"Latest news on {t}.",
                                  f"What's the news about {t}.", f"Any news about {t}?",
                                  f"Give me news on {t}.", f"Show news about {t}."])

    # --- coding-agent tools (Claude Code / Codex-style) ---
    def read_file(self) -> Sample:
        p = self.rng.choice(self.paths)
        return self._string_tool("read_file", "code", "read_file", "path", p,
                                 [f"Read the file {p}.", f"Open {p}.", f"Show the contents of {p}.",
                                  f"Display {p}.", f"Cat {p}.", f"Show me {p}."])

    def write_file(self) -> Sample:
        p = self.rng.choice(self.paths)
        return self._string_tool("write_file", "code", "write_file", "path", p,
                                 [f"Create the file {p}.", f"Write to {p}.",
                                  f"Save the file {p}.", f"Add a new file {p}."])

    def grep_search(self) -> Sample:
        pat = self.rng.choice(self.patterns)
        return self._string_tool("grep_search", "code", "grep_search", "pattern", pat,
                                 [f"Search the code for '{pat}'.", f"Grep for '{pat}'.",
                                  f"Find '{pat}' in the repo.", f"Where is '{pat}' used?",
                                  f"Locate '{pat}' in the code.", f"Search for '{pat}'."])

    def run_command(self) -> Sample:
        c = self.rng.choice(self.commands)
        return self._string_tool("run_command", "code", "run_command", "command", c,
                                 [f"Run the command '{c}'.", f"Execute '{c}' in the shell.",
                                  f"Run '{c}'.", f"Execute the command '{c}'."])

    def git_commit(self) -> Sample:
        msg = self.rng.choice(self.commits)
        return self._string_tool("git_commit", "code", "git_commit", "message", msg,
                                 [f"Commit with message '{msg}'.", f"Make a git commit saying '{msg}'.",
                                  f"Git commit with '{msg}'.", f"Commit the changes: '{msg}'."])

    def run_tests(self) -> Sample:
        phr = self.rng.choice(["Run the tests.", "Run the test suite.", "Execute all tests."])
        return Sample("run_tests", "code", phr, "tool", _tool_target("run_tests", {}),
                      "run_tests", "{}")

    # --- popular everyday tools ---
    def set_reminder(self) -> Sample:
        t = self.rng.choice(self.tasks)
        return self._string_tool("set_reminder", "tool_call", "set_reminder", "task", t,
                                 [f"Set a reminder to {t}.", f"Remind to {t}.",
                                  f"Remind me to {t}.", f"Add a reminder to {t}."])

    def set_timer(self) -> Sample:
        d = self.rng.choice(self.durations)
        return self._string_tool("set_timer", "tool_call", "set_timer", "duration", d,
                                 [f"Set a timer for {d}.", f"Start a timer for {d}.",
                                  f"Set a countdown for {d}.", f"Wake me in {d}."])

    # --- computer-use / productivity tools ---
    def calendar_event(self) -> Sample:
        t = self.rng.choice(self.events)
        return self._string_tool("calendar_event", "productivity", "calendar_event", "title", t,
                                 [f"Add a calendar event called '{t}'.",
                                  f"Schedule '{t}' on my calendar.",
                                  f"Create a calendar event '{t}'.", f"Put '{t}' on the calendar."])

    def send_email(self) -> Sample:
        nm = self.rng.choice(self.names)
        return self._string_tool("send_email", "productivity", "send_email", "recipient", nm,
                                 [f"Send an email to {nm}.", f"Email {nm}.",
                                  f"Write an email to {nm}.", f"Compose an email to {nm}."])

    def open_url(self) -> Sample:
        u = self.rng.choice(self.urls)
        return self._string_tool("open_url", "browser", "open_url", "url", u,
                                 [f"Open {u}.", f"Go to {u}.", f"Navigate to {u} in the browser.",
                                  f"Visit {u}.", f"Pull up {u}."])

    def notion_write(self) -> Sample:
        c = self.rng.choice(self.notion)
        return self._string_tool("notion_write", "productivity", "notion_write", "content", c,
                                 [f"Write '{c}' in Notion.", f"Add a Notion note saying '{c}'.",
                                  f"Note '{c}' in Notion.", f"Save '{c}' to Notion."])

    def slack_send(self) -> Sample:
        m = self.rng.choice(self.slack)
        return self._string_tool("slack_send", "productivity", "slack_send", "message", m,
                                 [f"Send a Slack message saying '{m}'.", f"Post '{m}' to Slack.",
                                  f"Slack the team '{m}'.", f"Send '{m}' on Slack."])

    def jira_issue(self) -> Sample:
        s = self.rng.choice(self.jira)
        return self._string_tool("jira_issue", "productivity", "jira_issue", "summary", s,
                                 [f"Create a Jira ticket titled '{s}'.",
                                  f"Open a Jira issue for '{s}'.",
                                  f"File a Jira bug for '{s}'.", f"Log a Jira issue: '{s}'."])

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
        m = [self.weather, self.calc, self.web_search, self.planner,
             self.define, self.play_music, self.get_news,
             self.read_file, self.write_file, self.grep_search, self.run_command,
             self.git_commit, self.run_tests, self.set_reminder, self.set_timer,
             self.calendar_event, self.send_email, self.open_url, self.notion_write,
             self.slack_send, self.jira_issue, self.text]
        if self.level >= 2:
            m.append(self.no_tool)
        return m

    # --- multi-turn coding episodes (Claude Code / Codex-style trajectories) ---
    def coding_episode(self) -> Conversation:
        """A short multi-turn tool-use trajectory: tool call -> tool response -> follow-up.
        Some follow-up args are grounded in the *tool response*, not the user turn (the case
        only a learned pointer head can handle)."""
        def A(name, args):
            return Message(role=Role.assistant, tool_calls=[ToolCall(name, args)])

        def T(resp):
            return Message(role=Role.tool, tool_response=resp)

        path = self.rng.choice(self.paths)
        which = self.rng.choice(["debug", "grep_read", "test_commit"])
        if which == "debug":
            msgs = [
                Message(role=Role.user, content=f"There's a bug in {path}, investigate it."),
                A("read_file", {"path": path}),
                T("def handler():\n    return None  # suspicious"),
                A("run_tests", {}),
                T("FAILED tests/test_handler.py::test_ok"),
                Message(role=Role.assistant,
                        content=f"The bug is in {path}: handler() returns None."),
            ]
        elif which == "grep_read":  # follow-up path comes from the tool response
            pat = self.rng.choice(self.patterns)
            msgs = [
                Message(role=Role.user, content=f"Where is '{pat}' used? Open that file."),
                A("grep_search", {"pattern": pat}),
                T(f"{path}:12:    {pat} appears here"),
                A("read_file", {"path": path}),
                T("<file contents>"),
                Message(role=Role.assistant, content=f"'{pat}' is used in {path}."),
            ]
        else:
            msg = self.rng.choice(self.commits)
            msgs = [
                Message(role=Role.user,
                        content=f"Run the tests and commit with message '{msg}'."),
                A("run_tests", {}),
                T("All tests passed."),
                A("git_commit", {"message": msg}),
                T("Committed abc123."),
                Message(role=Role.assistant, content=f"Done — committed '{msg}'."),
            ]
        return Conversation(messages=msgs, meta={"kind": "coding_episode", "type": which})

    def episodes(self, n: int) -> list[Conversation]:
        return [self.coding_episode() for _ in range(n)]

    def generate(self, n: int) -> list[Sample]:
        makers = self.makers()
        out, seen = [], set()
        # round-robin categories by attempt index (NOT len(out)) so a low-diversity category
        # hitting duplicates can't freeze the selector; dedupe identical prompts.
        attempts = 0
        while len(out) < n and attempts < n * 80:
            s = makers[attempts % len(makers)]()
            attempts += 1
            if s.prompt in seen:
                continue
            seen.add(s.prompt)
            out.append(s)
        return out


    def generate_weighted(self, n: int, weights: dict) -> list[Sample]:
        """Sample categories with per-category weights (>1 oversamples). Drives failure-driven
        enrichment: the flywheel raises weights on the categories the model is failing."""
        makers = self.makers()
        cats = [mk().category for mk in makers]      # one call each to read its category
        w = [max(0.05, weights.get(c, 1.0)) for c in cats]
        out, seen, tries = [], set(), 0
        while len(out) < n and tries < n * 100:
            tries += 1
            s = self.rng.choices(makers, weights=w, k=1)[0]()
            if s.prompt in seen:
                continue
            seen.add(s.prompt)
            out.append(s)
        return out

    def generate_balanced(self, per_category: int) -> list[Sample]:
        """Up to `per_category` unique samples per category — for meaningful per-group eval."""
        from collections import Counter
        makers = self.makers()
        out, seen, cnt = [], set(), Counter()
        attempts, cap = 0, per_category * len(makers) * 300
        while len(out) < per_category * len(makers) and attempts < cap:
            s = makers[attempts % len(makers)]()
            attempts += 1
            if cnt[s.category] >= per_category or s.prompt in seen:
                continue
            seen.add(s.prompt); cnt[s.category] += 1; out.append(s)
        return out


def synthesize(config_path: str) -> None:  # CLI entry retained
    raise NotImplementedError("Use scripts/flywheel.py — Generator drives data generation in-process")
