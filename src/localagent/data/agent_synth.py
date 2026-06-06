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


# A realistic usage distribution (not the old calc-dominated one): emphasize the two-call
# ("parallel") turns and productivity tools people actually want; down-weight the over-represented
# calculator/weather. Used as the base sampling weight in the flywheels.
REALISTIC_WEIGHTS = {
    "parallel": 2.5, "calc": 0.3, "weather": 0.6,
    "send_email": 1.4, "calendar_event": 1.4, "open_url": 1.4, "slack_send": 1.4,
    "notion_write": 1.4, "jira_issue": 1.4, "set_reminder": 1.2, "set_timer": 1.2,
    "read_file": 1.2, "write_file": 1.2, "run_command": 1.2, "git_commit": 1.2,
}


@dataclass
class Sample:
    category: str          # weather | calc | ... | parallel | text | no_tool
    group: str             # tool_call | web_search | planner | ... | parallel | text
    prompt: str            # user text (without framing markers)
    kind: str              # "tool" or "text"
    target: str            # canonical assistant body (no markers/EOS)
    ref_name: str = ""     # tool name (for tool kind; first call for parallel)
    ref_args: str = ""     # canonical sorted-key JSON args string (first call)
    calls: list = None     # [{"name","arguments"}, ...] when >1 call (parallel); else None


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

    # --- parallel / two-tool calls ("do X and Y" — what people actually want) ---
    _PARALLEL_POOL = ("weather", "web_search", "define", "play_music", "get_news", "read_file",
                      "run_tests", "set_reminder", "set_timer", "calendar_event", "send_email",
                      "open_url", "notion_write", "slack_send", "jira_issue", "grep_search",
                      "git_commit", "calc", "run_command")

    def parallel(self) -> Sample:
        """One user turn that needs TWO tool calls, joined by 'and'. Each clause is a standalone
        single-tool request so it splits/grounds cleanly (no value contains ' and ')."""
        for _ in range(20):
            a = getattr(self, self.rng.choice(self._PARALLEL_POOL))()
            b = getattr(self, self.rng.choice(self._PARALLEL_POOL))()
            p2 = b.prompt
            p2 = (p2[0].lower() + p2[1:]) if p2 else p2
            prompt = a.prompt.rstrip(".?! ") + " and " + p2
            if prompt.count(" and ") != 1:    # a value sneaked in an 'and' — retry
                continue
            calls = [{"name": a.ref_name, "arguments": json.loads(a.ref_args or "{}")},
                     {"name": b.ref_name, "arguments": json.loads(b.ref_args or "{}")}]
            target = " ".join(_tool_target(c["name"], c["arguments"]) for c in calls)
            return Sample("parallel", "parallel", prompt, "tool", target,
                          a.ref_name, a.ref_args, calls)
        return a  # fallback (rare)

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
             self.slack_send, self.jira_issue, self.parallel, self.text]
        if self.level >= 2:
            m.append(self.no_tool)
        return m

    # --- multi-turn trajectory helpers ---------------------------------------------------
    @staticmethod
    def _A(name, args):
        return Message(role=Role.assistant, tool_calls=[ToolCall(name, args)])

    @staticmethod
    def _T(resp):
        return Message(role=Role.tool, tool_response=resp)

    @staticmethod
    def _U(content):
        return Message(role=Role.user, content=content)

    @staticmethod
    def _F(content):  # final / intermediate assistant text turn
        return Message(role=Role.assistant, content=content)

    # Episode-type registries. Each builder returns a list[Message]. Builders whose name is in
    # `_PTR_*` ground at least one follow-up argument from a TOOL RESPONSE (pointer-head case).
    _CODING_TYPES = ("debug", "grep_read", "test_commit", "implement_feature", "fix_failing_test",
                     "refactor_rename", "review_pr", "dependency_bump", "trace_import",
                     "lint_and_fix")
    _PRODUCTIVITY_TYPES = ("research_summarize", "schedule_and_notify", "triage_issue",
                           "news_brief", "open_and_note", "remind_and_slack")
    _PLANNER_TYPES = ("plan_read_test_commit", "plan_research", "plan_fix_test")

    # ---- coding episode builders (return list[Message]) ----
    def _ep_debug(self):
        path = self.rng.choice(self.paths)
        return [
            self._U(f"There's a bug in {path}, investigate it."),
            self._A("read_file", {"path": path}),
            self._T("def handler():\n    return None  # suspicious"),
            self._A("run_tests", {}),
            self._T("FAILED tests/test_handler.py::test_ok"),
            self._F(f"The bug is in {path}: handler() returns None."),
        ]

    def _ep_grep_read(self):  # follow-up path grounded from the tool response (pointer case)
        path = self.rng.choice(self.paths)
        pat = self.rng.choice(self.patterns)
        return [
            self._U(f"Where is '{pat}' used? Open that file."),
            self._A("grep_search", {"pattern": pat}),
            self._T(f"{path}:12:    {pat} appears here"),
            self._A("read_file", {"path": path}),
            self._T("<file contents>"),
            self._F(f"'{pat}' is used in {path}."),
        ]

    def _ep_test_commit(self):
        msg = self.rng.choice(self.commits)
        return [
            self._U(f"Run the tests and commit with message '{msg}'."),
            self._A("run_tests", {}),
            self._T("All tests passed."),
            self._A("git_commit", {"message": msg}),
            self._T("Committed abc123."),
            self._F(f"Done — committed '{msg}'."),
        ]

    def _ep_implement_feature(self):  # grep -> read -> write -> test -> commit (pointer case)
        path = self.rng.choice(self.paths)
        pat = self.rng.choice(self.patterns)
        msg = self.rng.choice(self.commits)
        return [
            self._U(f"Implement '{msg}'; the relevant code uses '{pat}'."),
            self._A("grep_search", {"pattern": pat}),
            self._T(f"{path}:20:    {pat} ..."),
            self._A("read_file", {"path": path}),
            self._T("<current implementation>"),
            self._A("write_file", {"path": path}),
            self._T("written."),
            self._A("run_tests", {}),
            self._T("All tests passed."),
            self._A("git_commit", {"message": msg}),
            self._T("Committed def456."),
            self._F(f"Implemented and committed '{msg}'."),
        ]

    def _ep_fix_failing_test(self):  # run_tests -> read -> write -> run_tests
        path = self.rng.choice(self.paths)
        return [
            self._U(f"A test is failing in {path}, please fix it."),
            self._A("run_tests", {}),
            self._T(f"FAILED {path}::test_case"),
            self._A("read_file", {"path": path}),
            self._T("def f(): return None"),
            self._A("write_file", {"path": path}),
            self._T("patched."),
            self._A("run_tests", {}),
            self._T("All tests passed."),
            self._F(f"Fixed the failing test in {path}."),
        ]

    def _ep_refactor_rename(self):  # grep -> (response has path) -> read -> write -> tests (pointer)
        path = self.rng.choice(self.paths)
        pat = self.rng.choice(self.patterns)
        return [
            self._U(f"Rename the symbol '{pat}'; find where it lives and update it."),
            self._A("grep_search", {"pattern": pat}),
            self._T(f"{path}:34:    {pat} = ...  # definition"),
            self._A("read_file", {"path": path}),
            self._T("<current source>"),
            self._A("write_file", {"path": path}),
            self._T("written."),
            self._A("run_tests", {}),
            self._T("All tests passed."),
            self._F(f"Renamed '{pat}' in {path} and tests pass."),
        ]

    def _ep_review_pr(self):  # read -> (response = diff) -> comment text turn
        path = self.rng.choice(self.paths)
        return [
            self._U(f"Review the changes in {path}."),
            self._A("read_file", {"path": path}),
            self._T(f"@@ -1,3 +1,4 @@\n+    log.debug('x')\n     return value  # in {path}"),
            self._F(f"Looks good: {path} adds a debug log; no blocking issues."),
        ]

    def _ep_dependency_bump(self):  # read config -> write config -> install -> tests -> commit
        path = self.rng.choice(self.paths)
        cmd = self.rng.choice(self.commands)
        msg = self.rng.choice(self.commits)
        return [
            self._U(f"Bump a dependency in {path}, install, test, then commit '{msg}'."),
            self._A("read_file", {"path": path}),
            self._T("requests==2.0.0"),
            self._A("write_file", {"path": path}),
            self._T("written."),
            self._A("run_command", {"command": cmd}),
            self._T("install complete."),
            self._A("run_tests", {}),
            self._T("All tests passed."),
            self._A("git_commit", {"message": msg}),
            self._T("Committed 9f8e7d."),
            self._F(f"Bumped the dependency and committed '{msg}'."),
        ]

    def _ep_trace_import(self):  # grep -> (response has path) -> read -> text (pointer)
        path = self.rng.choice(self.paths)
        pat = self.rng.choice(self.patterns)
        return [
            self._U(f"Trace where '{pat}' is imported and open that module."),
            self._A("grep_search", {"pattern": pat}),
            self._T(f"{path}:1:from x import {pat}"),
            self._A("read_file", {"path": path}),
            self._T("<module source>"),
            self._F(f"'{pat}' is imported in {path}."),
        ]

    def _ep_lint_and_fix(self):  # run_command(lint) -> (response has path) -> read -> write (pointer)
        path = self.rng.choice(self.paths)
        cmd = self.rng.choice(self.commands)
        return [
            self._U(f"Run '{cmd}' and fix whatever file it flags."),
            self._A("run_command", {"command": cmd}),
            self._T(f"{path}:7: E501 line too long"),
            self._A("read_file", {"path": path}),
            self._T("<offending line>"),
            self._A("write_file", {"path": path}),
            self._T("written."),
            self._F(f"Fixed the lint issue in {path}."),
        ]

    # ---- productivity / computer-use episode builders ----
    def _ep_research_summarize(self):  # web_search -> (response has url) -> open_url -> text (pointer)
        query = self.rng.choice(self.queries)
        url = self.rng.choice(self.urls)
        return [
            self._U(f"Research {query} and open the top result."),
            self._A("web_search", {"query": query}),
            self._T(f"1. {url} — a relevant overview of {query}"),
            self._A("open_url", {"url": url}),
            self._T("<page loaded>"),
            self._F(f"Top result for {query} is {url}."),
        ]

    def _ep_schedule_and_notify(self):  # calendar_event -> send_email -> text
        title = self.rng.choice(self.events)
        nm = self.rng.choice(self.names)
        return [
            self._U(f"Schedule '{title}' and email {nm} about it."),
            self._A("calendar_event", {"title": title}),
            self._T("event created."),
            self._A("send_email", {"recipient": nm}),
            self._T("email sent."),
            self._F(f"Scheduled '{title}' and emailed {nm}."),
        ]

    def _ep_triage_issue(self):  # jira_issue -> slack_send -> text
        summary = self.rng.choice(self.jira)
        msg = self.rng.choice(self.slack)
        return [
            self._U(f"File a Jira issue for '{summary}' and ping the team on Slack '{msg}'."),
            self._A("jira_issue", {"summary": summary}),
            self._T("created PROJ-123."),
            self._A("slack_send", {"message": msg}),
            self._T("message posted."),
            self._F(f"Filed '{summary}' and notified the team."),
        ]

    def _ep_news_brief(self):  # get_news -> (response has url) -> open_url -> text (pointer)
        topic = self.rng.choice(self.topics)
        url = self.rng.choice(self.urls)
        return [
            self._U(f"Get the latest news on {topic} and open the top story."),
            self._A("get_news", {"topic": topic}),
            self._T(f"Top story: {url} covers {topic}"),
            self._A("open_url", {"url": url}),
            self._T("<page loaded>"),
            self._F(f"Top {topic} story is at {url}."),
        ]

    def _ep_open_and_note(self):  # open_url -> notion_write -> text
        url = self.rng.choice(self.urls)
        content = self.rng.choice(self.notion)
        return [
            self._U(f"Open {url} and note '{content}' in Notion."),
            self._A("open_url", {"url": url}),
            self._T("<page loaded>"),
            self._A("notion_write", {"content": content}),
            self._T("note saved."),
            self._F(f"Opened {url} and saved '{content}' to Notion."),
        ]

    def _ep_remind_and_slack(self):  # set_reminder -> slack_send -> text
        task = self.rng.choice(self.tasks)
        msg = self.rng.choice(self.slack)
        return [
            self._U(f"Remind me to {task} and tell the team on Slack '{msg}'."),
            self._A("set_reminder", {"task": task}),
            self._T("reminder set."),
            self._A("slack_send", {"message": msg}),
            self._T("message posted."),
            self._F(f"Reminder set to {task} and team notified."),
        ]

    # ---- planner-then-execute episode builders ----
    # The plan text is deterministic/canonical so it is exactly learnable and scorable.
    @staticmethod
    def _plan(steps):
        return "Plan: " + " ".join(f"{i + 1}) {s}" for i, s in enumerate(steps))

    def _ep_plan_read_test_commit(self):
        path = self.rng.choice(self.paths)
        msg = self.rng.choice(self.commits)
        return [
            self._U(f"Read {path}, run the tests, then commit '{msg}'."),
            self._F(self._plan(["read the file", "run tests", "commit"])),
            self._A("read_file", {"path": path}),
            self._T("<file contents>"),
            self._A("run_tests", {}),
            self._T("All tests passed."),
            self._A("git_commit", {"message": msg}),
            self._T("Committed 1a2b3c."),
            self._F(f"Read {path}, tests pass, committed '{msg}'."),
        ]

    def _ep_plan_research(self):  # plan -> search -> (response url) -> open_url -> summary (pointer)
        query = self.rng.choice(self.queries)
        url = self.rng.choice(self.urls)
        return [
            self._U(f"Look into {query}: search, open the best link, then summarize."),
            self._F(self._plan(["search the web", "open the top result", "summarize"])),
            self._A("web_search", {"query": query}),
            self._T(f"1. {url} — the best overview of {query}"),
            self._A("open_url", {"url": url}),
            self._T("<page loaded>"),
            self._F(f"Summary: {url} is the best source on {query}."),
        ]

    def _ep_plan_fix_test(self):  # plan -> run_tests -> read -> write -> run_tests -> summary
        path = self.rng.choice(self.paths)
        return [
            self._U(f"A test in {path} is broken — make a plan and fix it."),
            self._F(self._plan(["run tests", "read the file", "fix it", "run tests again"])),
            self._A("run_tests", {}),
            self._T(f"FAILED {path}::test_case"),
            self._A("read_file", {"path": path}),
            self._T("def f(): return None"),
            self._A("write_file", {"path": path}),
            self._T("patched."),
            self._A("run_tests", {}),
            self._T("All tests passed."),
            self._F(f"Fixed the failing test in {path}."),
        ]

    def _coding_builders(self):
        return {
            "debug": self._ep_debug, "grep_read": self._ep_grep_read,
            "test_commit": self._ep_test_commit, "implement_feature": self._ep_implement_feature,
            "fix_failing_test": self._ep_fix_failing_test, "refactor_rename": self._ep_refactor_rename,
            "review_pr": self._ep_review_pr, "dependency_bump": self._ep_dependency_bump,
            "trace_import": self._ep_trace_import, "lint_and_fix": self._ep_lint_and_fix,
        }

    def _productivity_builders(self):
        return {
            "research_summarize": self._ep_research_summarize,
            "schedule_and_notify": self._ep_schedule_and_notify,
            "triage_issue": self._ep_triage_issue, "news_brief": self._ep_news_brief,
            "open_and_note": self._ep_open_and_note, "remind_and_slack": self._ep_remind_and_slack,
        }

    def _planner_builders(self):
        return {
            "plan_read_test_commit": self._ep_plan_read_test_commit,
            "plan_research": self._ep_plan_research, "plan_fix_test": self._ep_plan_fix_test,
        }

    def coding_episode(self) -> Conversation:
        """A short multi-turn coding tool-use trajectory: tool call -> tool response -> follow-up.
        Some follow-up args are grounded in the *tool response*, not the user turn (the case
        only a learned pointer head can handle)."""
        which = self.rng.choice(self._CODING_TYPES)
        msgs = self._coding_builders()[which]()
        return Conversation(messages=msgs, meta={"kind": "coding_episode", "type": which})

    def productivity_episode(self) -> Conversation:
        """A multi-turn computer-use / productivity trajectory (calendar, email, browser, Slack,
        Notion, Jira). Several ground a follow-up arg (a URL) from the tool response."""
        which = self.rng.choice(self._PRODUCTIVITY_TYPES)
        msgs = self._productivity_builders()[which]()
        return Conversation(messages=msgs, meta={"kind": "productivity_episode", "type": which})

    def planner_episode(self) -> Conversation:
        """A planner-then-execute trajectory: a canonical text plan turn, then each step as a
        tool-call turn with tool responses between, then a final summary (teaches plan->act)."""
        which = self.rng.choice(self._PLANNER_TYPES)
        msgs = self._planner_builders()[which]()
        return Conversation(messages=msgs, meta={"kind": "planner_episode", "type": which})

    def coding_episodes(self, n: int) -> list[Conversation]:
        return [self.coding_episode() for _ in range(n)]

    def productivity_episodes(self, n: int) -> list[Conversation]:
        return [self.productivity_episode() for _ in range(n)]

    def planner_episodes(self, n: int) -> list[Conversation]:
        return [self.planner_episode() for _ in range(n)]

    def episodes(self, n: int, mix: bool = True) -> list[Conversation]:
        """Sample `n` multi-turn episodes. With ``mix=True`` (default) the pool spans coding +
        productivity + planner trajectories so the flywheel/eval see the full diversity; with
        ``mix=False`` it returns coding-only episodes (the original behaviour)."""
        if not mix:
            return [self.coding_episode() for _ in range(n)]
        builders = [self.coding_episode, self.productivity_episode, self.planner_episode]
        # weight coding a bit higher (it has the most types), then productivity, then planner
        weights = [0.5, 0.3, 0.2]
        return [self.rng.choices(builders, weights=weights, k=1)[0]() for _ in range(n)]

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
            seen.add(s.prompt)
            cnt[s.category] += 1
            out.append(s)
        return out


def synthesize(config_path: str) -> None:  # CLI entry retained
    raise NotImplementedError("Use scripts/flywheel.py — Generator drives data generation in-process")
