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

# --- computer-use surface. All values are quoted in the prompt so they ground as literal
# substrings (the byte model has no vision; targets are semantic element descriptions, not pixels).
# UI element targets (click/double_click/move_cursor). Train/eval disjoint. ---
UI_TARGETS_TRAIN = ["the Submit button", "the Login button", "the Save icon", "the Search box",
                    "the Settings menu", "the OK button", "the Cancel button", "the Next button",
                    "the Profile tab", "the hamburger menu", "the Download link", "the Send button",
                    "the Close button", "the Add button", "the Edit pencil", "the Filter dropdown",
                    "the Share button", "the Upload area", "the Refresh icon", "the New Tab button",
                    "the username field", "the password field", "the Continue button",
                    "the dropdown arrow", "the checkbox", "the Reply button", "the Compose button"]
UI_TARGETS_EVAL = ["the Delete button", "the Confirm button", "the Back arrow", "the email field",
                   "the Sign Up button", "the Apply button", "the gear icon", "the notification bell",
                   "the Done button", "the Bookmark star"]
# text typed into fields (type_text / write_clipboard). Train/eval disjoint.
TYPED_TEXT_TRAIN = ["hello world", "admin@example.com", "my search query", "first last name",
                    "the quick brown fox", "January 2026", "password123", "remote work policy",
                    "quarterly report", "weekly standup notes", "order number 4521",
                    "shipping address", "draft invoice", "project alpha", "status update"]
TYPED_TEXT_EVAL = ["good afternoon", "test@mail.com", "annual budget", "release notes",
                   "meeting agenda", "customer feedback", "invoice 9087"]
# desktop apps (open_app). Train/eval disjoint.
APPS_TRAIN = ["Chrome", "Safari", "Slack", "Spotify", "Terminal", "VS Code", "Notion", "Figma",
              "Calculator", "Calendar", "Mail", "Photoshop", "Finder", "Discord", "Zoom",
              "Excel", "Word", "Preview", "Notes", "Messages"]
APPS_EVAL = ["Firefox", "Postman", "Docker Desktop", "iTerm", "Obsidian", "Outlook", "Telegram",
             "PowerPoint"]
# wait durations in seconds (wait.seconds integer extractor). Train/eval disjoint.
WAIT_SECONDS_TRAIN = [2, 3, 5, 7, 10, 15, 20, 30, 45, 60, 90, 120]
WAIT_SECONDS_EVAL = [1, 4, 8, 12, 25, 40, 75]

# --- modern dev / agentic surface ---
# Python snippets (run_python) — quoted so they ground exactly. Train/eval disjoint.
PYCODE_TRAIN = ["print('hi')", "import numpy as np", "x = sum(range(10))", "len(data)",
                "df.head()", "os.getcwd()", "2 ** 16", "sorted(items)", "model.eval()",
                "json.loads(s)", "np.zeros(5)", "open('f.txt').read()", "time.sleep(1)",
                "requests.get(url)", "plt.show()"]
PYCODE_EVAL = ["print('done')", "math.sqrt(2)", "list(map(str, xs))", "pd.read_csv('a.csv')",
               "random.seed(0)", "min(values)", "torch.randn(3)"]
# glob patterns (find_files) — quoted. Train/eval disjoint.
GLOBS_TRAIN = ["*.py", "**/*.js", "test_*.py", "*.log", "src/**/*.ts", "*.yaml", "Dockerfile",
               "*.md", "**/*.go", "*.csv", "config/*.json", "*.rs", "*.sql", "*.png"]
GLOBS_EVAL = ["*.txt", "**/*.tsx", "*.toml", "spec_*.rb", "*.html", "lib/**/*.py", "*.env"]
# SQL queries (sql_query) — quoted. Train/eval disjoint.
SQL_TRAIN = ["SELECT * FROM users", "SELECT count(*) FROM orders", "DELETE FROM logs",
             "SELECT name FROM products", "UPDATE users SET active = 1", "SELECT * FROM events",
             "INSERT INTO tags VALUES (1)", "SELECT id FROM sessions", "SELECT * FROM payments",
             "SELECT email FROM accounts", "DROP TABLE temp", "SELECT * FROM inventory"]
SQL_EVAL = ["SELECT * FROM customers", "SELECT max(price) FROM items", "TRUNCATE cache",
            "SELECT title FROM posts", "UPDATE orders SET paid = 1", "SELECT * FROM metrics"]
# package names (install_package) — quoted. Train/eval disjoint.
PACKAGES_TRAIN = ["numpy", "pandas", "requests", "flask", "pytest", "torch", "fastapi", "redis",
                  "django", "scipy", "click", "pydantic", "uvicorn", "aiohttp", "sqlalchemy",
                  "matplotlib", "pillow", "boto3", "celery", "jinja2"]
PACKAGES_EVAL = ["scikit-learn", "transformers", "httpx", "rich", "typer", "polars", "ruff",
                 "lxml"]
# process names (kill_process) — quoted. Train/eval disjoint.
PROCESSES_TRAIN = ["node", "python", "chrome", "nginx", "postgres", "redis-server", "java",
                   "docker", "vite", "webpack", "gunicorn", "mysqld", "ssh", "code"]
PROCESSES_EVAL = ["firefox", "rabbitmq", "mongod", "celery", "uvicorn", "elasticsearch"]
# env var names (env_get) — quoted. Train/eval disjoint.
ENVVARS_TRAIN = ["PATH", "HOME", "API_KEY", "DATABASE_URL", "PORT", "AWS_REGION", "NODE_ENV",
                 "PYTHONPATH", "SECRET_KEY", "REDIS_URL", "LOG_LEVEL", "USER", "SHELL",
                 "OPENAI_API_KEY", "DEBUG"]
ENVVARS_EVAL = ["LANG", "TZ", "GITHUB_TOKEN", "S3_BUCKET", "JAVA_HOME", "VIRTUAL_ENV", "EDITOR"]
# docker images (docker_run) — quoted. Train/eval disjoint.
IMAGES_TRAIN = ["nginx", "postgres", "redis", "ubuntu", "python:3.12", "node:20", "alpine",
                "mysql", "mongo", "busybox", "golang", "rust", "httpd", "memcached"]
IMAGES_EVAL = ["debian", "elasticsearch", "rabbitmq", "grafana", "prometheus", "traefik"]
# archive paths (unzip) — file paths. Train/eval disjoint.
ARCHIVES_TRAIN = ["data.zip", "release.zip", "assets.zip", "backup.tar.gz", "dataset.zip",
                  "logs.zip", "build/output.zip", "models.zip", "images.tar.gz", "dist.zip",
                  "src/bundle.zip", "exports.zip"]
ARCHIVES_EVAL = ["archive.zip", "vendor.tar.gz", "photos.zip", "downloads/pack.zip", "weights.zip"]

# --- implicit factual-question entities -> web_search. The query arg is grounded to the ENTITY,
# which is a literal substring of every templated factual question. Train/eval disjoint. ---
ENTITIES_TRAIN = ["Mount Everest", "the Eiffel Tower", "the Great Wall of China", "Lake Baikal",
                  "the Amazon River", "the Sahara Desert", "the Pacific Ocean", "Mount Fuji",
                  "the Nile", "the Colosseum", "the Empire State Building", "the Golden Gate Bridge",
                  "the Grand Canyon", "Niagara Falls", "the Burj Khalifa", "the Statue of Liberty",
                  "the Leaning Tower of Pisa", "the Sydney Opera House", "Mount Kilimanjaro",
                  "the Mississippi River", "the Andes", "the Dead Sea", "the Sahara", "Big Ben",
                  "the Hoover Dam", "the Panama Canal", "the Taj Mahal", "Stonehenge",
                  "the Suez Canal", "the Mariana Trench", "Mount Rushmore", "the Rocky Mountains"]
ENTITIES_EVAL = ["the Matterhorn", "Lake Victoria", "the Yangtze River", "the Gobi Desert",
                 "the Arctic Ocean", "Mount Etna", "the Danube", "the Acropolis",
                 "the Chrysler Building", "the Brooklyn Bridge", "Angkor Wat", "the Petronas Towers"]
# "What's the {ATTR} of {PLACE}?" factual questions -> web_search (query grounded to PLACE).
PLACES_TRAIN = ["Peru", "Mongolia", "Iceland", "Portugal", "Kenya", "Vietnam", "Norway", "Chile",
                "Morocco", "Nepal", "Bolivia", "Finland", "Ireland", "Greece", "Croatia", "Ghana",
                "Ecuador", "Sweden", "Romania", "Hungary", "Tunisia", "Jordan", "Latvia", "Senegal",
                "Uruguay", "Slovenia", "Estonia", "Armenia", "Georgia", "Cambodia", "Laos", "Oman"]
PLACES_EVAL = ["Paraguay", "Lithuania", "Slovakia", "Namibia", "Botswana", "Bhutan", "Moldova",
               "Albania", "Tanzania", "Zambia", "Belize", "Brunei"]
# "Who {invented/wrote/founded/discovered} {THING}?" -> web_search (query grounded to THING).
INVENTIONS_TRAIN = ["the telephone", "the light bulb", "the printing press", "the steam engine",
                    "the airplane", "the telescope", "the World Wide Web", "the radio",
                    "the periodic table", "the theory of relativity", "the polio vaccine",
                    "the cotton gin", "the sewing machine", "the microscope", "dynamite",
                    "the assembly line", "the transistor", "penicillin", "the barometer",
                    "Hamlet", "Moby Dick", "War and Peace", "the Mona Lisa", "Pride and Prejudice"]
INVENTIONS_EVAL = ["the camera", "the typewriter", "the thermometer", "the compass",
                   "the seismograph", "Frankenstein", "Don Quixote", "the Starry Night"]
# "When did {EVENT} happen?" / "What year was {EVENT}?" -> web_search (query grounded to EVENT).
EVENTS_TRAIN = ["World War II", "the French Revolution", "the moon landing", "the Renaissance",
                "the Industrial Revolution", "the fall of the Berlin Wall", "the Cold War",
                "the American Civil War", "the Great Depression", "the Roman Empire",
                "the Battle of Hastings", "the Boston Tea Party", "the gold rush",
                "the Cuban Missile Crisis", "the signing of the Magna Carta", "the Black Death"]
EVENTS_EVAL = ["the Spanish Inquisition", "the Wright brothers' first flight", "the Boston Marathon",
               "the invention of the internet", "the discovery of America", "the Russian Revolution"]


# A realistic usage distribution (not the old calc-dominated one): emphasize the two-call
# ("parallel") turns and productivity tools people actually want; down-weight the over-represented
# calculator/weather. Used as the base sampling weight in the flywheels.
REALISTIC_WEIGHTS = {
    "parallel": 2.5, "calc": 0.3, "weather": 0.6,
    "send_email": 1.4, "calendar_event": 1.4, "open_url": 1.4, "slack_send": 1.4,
    "notion_write": 1.4, "jira_issue": 1.4, "set_reminder": 1.2, "set_timer": 1.2,
    "read_file": 1.2, "write_file": 1.2, "run_command": 1.2, "git_commit": 1.2,
    # computer-use family — the headline new capability, weight it up
    "click": 1.6, "type_text": 1.5, "key_press": 1.3, "scroll": 1.2, "screenshot": 1.1,
    "double_click": 1.2, "drag": 1.2, "wait": 1.0, "move_cursor": 1.1, "open_app": 1.5,
    # modern dev / agentic tools
    "run_python": 1.4, "edit_file": 1.3, "apply_patch": 1.1, "http_request": 1.3,
    "sql_query": 1.3, "list_dir": 1.2, "find_files": 1.2, "git_diff": 1.1, "git_status": 1.1,
    "install_package": 1.3, "kill_process": 1.1, "read_clipboard": 0.9, "write_clipboard": 1.0,
    "download_file": 1.2, "unzip": 1.0, "env_get": 1.0, "make_dir": 1.0, "list_processes": 0.9,
    "docker_run": 1.2,
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
        self.entities = ENTITIES_TRAIN if tr else ENTITIES_EVAL
        self.places = PLACES_TRAIN if tr else PLACES_EVAL
        self.inventions = INVENTIONS_TRAIN if tr else INVENTIONS_EVAL
        self.history = EVENTS_TRAIN if tr else EVENTS_EVAL  # historical events for web_search
        # computer-use surface
        self.ui_targets = UI_TARGETS_TRAIN if tr else UI_TARGETS_EVAL
        self.typed_text = TYPED_TEXT_TRAIN if tr else TYPED_TEXT_EVAL
        self.apps = APPS_TRAIN if tr else APPS_EVAL
        self.wait_seconds = WAIT_SECONDS_TRAIN if tr else WAIT_SECONDS_EVAL
        # modern dev / agentic surface
        self.pycode = PYCODE_TRAIN if tr else PYCODE_EVAL
        self.globs = GLOBS_TRAIN if tr else GLOBS_EVAL
        self.sql = SQL_TRAIN if tr else SQL_EVAL
        self.packages = PACKAGES_TRAIN if tr else PACKAGES_EVAL
        self.processes = PROCESSES_TRAIN if tr else PROCESSES_EVAL
        self.envvars = ENVVARS_TRAIN if tr else ENVVARS_EVAL
        self.images = IMAGES_TRAIN if tr else IMAGES_EVAL
        self.archives = ARCHIVES_TRAIN if tr else ARCHIVES_EVAL

    # ---- per-category sample makers ----
    def weather(self) -> Sample:
        city = self.rng.choice(self.cities)
        phr = self.rng.choice([
            f"What's the weather in {city}?",
            f"Tell me the weather for {city}.",
            f"How is the weather in {city} right now?",
            f"I wonder what the weather's like in {city}.",
            f"Is it raining in {city}?",
            f"What's it like outside in {city}?",
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
        # Natural arithmetic phrasings -> calculator. The arithmetic schema extractor grounds the
        # expression from the digit/operator span (whitespace-stripped), so we keep the SYMBOL form
        # (e.g. "7*8") inside the prompt rather than word forms ("7 times 8") that wouldn't ground.
        if self.rng.random() < 0.5 and expr == f"{a}{op}{b}":
            q = self.rng.choice([
                f"How much is {a}{op}{b}?",
                f"Can you compute {a}{op}{b}?",
                f"What's {a}{op}{b}?",
                f"Calculate {a}{op}{b}.",
                f"Work out {a}{op}{b} for me.",
            ])
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

    def web_search_implicit(self) -> Sample:
        """Implicit factual questions (bare questions, NOT explicit "search" commands) that should
        still map to ``web_search``. The ``query`` arg is grounded to an ENTITY/PLACE/THING/EVENT
        that is a literal substring of the chosen phrasing. This is the headline coverage gap: the
        model previously only saw imperative "search for X" prompts and mis-routed real questions."""
        kind = self.rng.choice(["measure", "attr", "who", "when"])
        if kind == "measure":
            e = self.rng.choice(self.entities)
            adj = self.rng.choice(["tall", "high", "far away", "old", "long", "deep", "heavy",
                                   "big", "wide"])
            phr = self.rng.choice([
                f"How {adj} is {e}?",
                f"Do you know how {adj} {e} is?",
                f"I wonder how {adj} {e} is.",
            ])
            query = e
        elif kind == "attr":
            p = self.rng.choice(self.places)
            attr = self.rng.choice(["capital", "population", "currency", "national language",
                                    "area", "time zone", "flag"])
            phr = self.rng.choice([
                f"What's the {attr} of {p}?",
                f"What is the {attr} of {p}?",
                f"Tell me the {attr} of {p}.",
                f"Do you know the {attr} of {p}?",
            ])
            query = p
        elif kind == "who":
            t = self.rng.choice(self.inventions)
            verb = self.rng.choice(["invented", "wrote", "founded", "discovered", "designed",
                                    "built", "painted", "created"])
            phr = self.rng.choice([
                f"Who {verb} {t}?",
                f"Do you know who {verb} {t}?",
                f"Any idea who {verb} {t}?",
            ])
            query = t
        else:  # when
            ev = self.rng.choice(self.history)
            phr = self.rng.choice([
                f"When did {ev} happen?",
                f"What year was {ev}?",
                f"When was {ev}?",
                f"What year did {ev} take place?",
            ])
            query = ev
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
        # Meaning questions -> define (distinct from web_search facts and get_news current events).
        return self._string_tool("define", "define", "define", "term", t,
                                 [f"Definition of {t}.", f"Define {t}.", f"Explain {t}.",
                                  f"Tell me about {t}.", f"Describe {t}.",
                                  f"Give me the definition of {t}.",
                                  f"What does {t} mean?", f"What is the meaning of {t}?",
                                  f"What's the definition of {t}?", f"What does the word {t} mean?",
                                  f"Can you define {t}?"])

    def play_music(self) -> Sample:
        s = self.rng.choice(self.songs)
        return self._string_tool("play_music", "music", "play_music", "song", s,
                                 [f"Play {s}.", f"Put on {s}.", f"Start playing {s}.",
                                  f"Queue up {s}.", f"I want to hear {s}.", f"Play the song {s}.",
                                  f"Can you put on {s}?", f"I'm in the mood for {s}.",
                                  f"Let's listen to {s}."])

    def get_news(self) -> Sample:
        t = self.rng.choice(self.topics)
        # Current-events questions -> get_news (distinct from define meaning & web_search facts).
        return self._string_tool("get_news", "news", "get_news", "topic", t,
                                 [f"Show the news about {t}.", f"Latest news on {t}.",
                                  f"What's the news about {t}.", f"Any news about {t}?",
                                  f"Give me news on {t}.", f"Show news about {t}.",
                                  f"What's the latest on {t}?", f"What's happening with {t}?",
                                  f"Any updates on {t}?", f"What's new with {t}?",
                                  f"Catch me up on {t}."])

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
                                  f"Remind me to {t}.", f"Add a reminder to {t}.",
                                  f"Can you remind me to {t}?", f"Don't let me forget to {t}.",
                                  f"I need to remember to {t}."])

    def set_timer(self) -> Sample:
        d = self.rng.choice(self.durations)
        return self._string_tool("set_timer", "tool_call", "set_timer", "duration", d,
                                 [f"Set a timer for {d}.", f"Start a timer for {d}.",
                                  f"Set a countdown for {d}.", f"Wake me in {d}.",
                                  f"Can you set a timer for {d}?", f"Let me know in {d}.",
                                  f"Ping me in {d}."])

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
                                  f"Write an email to {nm}.", f"Compose an email to {nm}.",
                                  f"Can you email {nm}?", f"Shoot an email over to {nm}.",
                                  f"Drop {nm} an email."])

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

    # --- computer-use family (text-grounded GUI actions) -------------------------------------
    # All string args are wrapped in single quotes inside the prompt so the `quoted` extractor
    # grounds them as exact substrings (the byte model has no vision — targets are semantic
    # element descriptions, never pixel coordinates). Enums/ints ground via schema extractors.
    def _tool_sample(self, category, group, name, args) -> Sample:
        ref_args = json.dumps(args, separators=(",", ":"), sort_keys=True)
        return Sample(category, group, "", "tool", _tool_target(name, args), name, ref_args)

    def screenshot(self) -> Sample:
        phr = self.rng.choice(["Take a screenshot.", "Capture the screen.",
                               "Grab a screenshot of the screen.", "Screenshot the current screen.",
                               "Snap a picture of what's on screen."])
        s = self._tool_sample("screenshot", "computer_use", "screenshot", {})
        s.prompt = phr
        return s

    def click(self) -> Sample:
        t = self.rng.choice(self.ui_targets)
        s = self._tool_sample("click", "computer_use", "click", {"target": t})
        s.prompt = self.rng.choice([
            f"Click '{t}'.", f"Click on '{t}'.", f"Press '{t}'.",
            f"Tap '{t}'.", f"Hit '{t}'.", f"Select '{t}'.", f"Go ahead and click '{t}'."])
        return s

    def double_click(self) -> Sample:
        t = self.rng.choice(self.ui_targets)
        s = self._tool_sample("double_click", "computer_use", "double_click", {"target": t})
        s.prompt = self.rng.choice([
            f"Double-click '{t}'.", f"Double click '{t}'.", f"Double-click on '{t}'.",
            f"Open '{t}' by double-clicking.", f"Double tap '{t}'."])
        return s

    def type_text(self) -> Sample:
        txt = self.rng.choice(self.typed_text)
        s = self._tool_sample("type_text", "computer_use", "type_text", {"text": txt})
        s.prompt = self.rng.choice([
            f"Type '{txt}'.", f"Type '{txt}' into the field.", f"Enter '{txt}'.",
            f"Input '{txt}'.", f"Write '{txt}' in the box.", f"Fill in '{txt}'."])
        return s

    def key_press(self) -> Sample:
        key = self.rng.choice(["Enter", "Tab", "Escape", "Backspace", "Space", "Delete",
                               "ArrowUp", "ArrowDown", "ArrowLeft", "ArrowRight"])
        s = self._tool_sample("key_press", "computer_use", "key_press", {"key": key})
        s.prompt = self.rng.choice([
            f"Press {key}.", f"Hit the {key} key.", f"Press the {key} key.",
            f"Tap {key}.", f"Send a {key} keypress."])
        return s

    def scroll(self) -> Sample:
        d = self.rng.choice(["up", "down", "left", "right"])
        s = self._tool_sample("scroll", "computer_use", "scroll", {"direction": d})
        s.prompt = self.rng.choice([
            f"Scroll {d}.", f"Scroll {d} a bit.", f"Scroll the page {d}.",
            f"Please scroll {d}.", f"Keep scrolling {d}."])
        return s

    def drag(self) -> Sample:
        src, dst = self.rng.sample(self.ui_targets, 2)
        s = self._tool_sample("drag", "computer_use", "drag", {"source": src, "dest": dst})
        s.prompt = self.rng.choice([
            f"Drag '{src}' to '{dst}'.", f"Drag '{src}' onto '{dst}'.",
            f"Move '{src}' over to '{dst}' by dragging.", f"Drag and drop '{src}' to '{dst}'."])
        return s

    def wait(self) -> Sample:
        sec = self.rng.choice(self.wait_seconds)
        s = self._tool_sample("wait", "computer_use", "wait", {"seconds": int(sec)})
        s.prompt = self.rng.choice([
            f"Wait {sec} seconds.", f"Wait for {sec} seconds.", f"Pause for {sec} seconds.",
            f"Hold on {sec} seconds.", f"Give it {sec} seconds."])
        return s

    def move_cursor(self) -> Sample:
        t = self.rng.choice(self.ui_targets)
        s = self._tool_sample("move_cursor", "computer_use", "move_cursor", {"target": t})
        s.prompt = self.rng.choice([
            f"Move the cursor to '{t}'.", f"Hover over '{t}'.", f"Move the mouse to '{t}'.",
            f"Point at '{t}'.", f"Bring the cursor to '{t}'."])
        return s

    def open_app(self) -> Sample:
        nm = self.rng.choice(self.apps)
        s = self._tool_sample("open_app", "computer_use", "open_app", {"name": nm})
        s.prompt = self.rng.choice([
            f"Open '{nm}'.", f"Launch '{nm}'.", f"Open the '{nm}' app.",
            f"Start '{nm}'.", f"Fire up '{nm}'.", f"Bring up '{nm}'."])
        return s

    # --- modern dev / agentic tools ----------------------------------------------------------
    def run_python(self) -> Sample:
        code = self.rng.choice(self.pycode)
        s = self._tool_sample("run_python", "code", "run_python", {"code": code})
        s.prompt = self.rng.choice([
            f"Run the Python code '{code}'.", f"Execute '{code}' in Python.",
            f"Run '{code}'.", f"Evaluate '{code}' in a Python shell."])
        return s

    def edit_file(self) -> Sample:
        p = self.rng.choice(self.paths)
        return self._string_tool("edit_file", "code", "edit_file", "path", p,
                                 [f"Edit {p}.", f"Make changes to {p}.", f"Modify {p}.",
                                  f"Update the file {p}.", f"Open {p} for editing."])

    def apply_patch(self) -> Sample:
        p = self.rng.choice(self.paths)
        return self._string_tool("apply_patch", "code", "apply_patch", "path", p,
                                 [f"Apply the patch to {p}.", f"Patch {p}.",
                                  f"Apply a patch to the file {p}.", f"Patch the file {p}."])

    def http_request(self) -> Sample:
        u = self.rng.choice(self.urls)
        args = {"url": u}
        if self.rng.random() < 0.5:
            method = self.rng.choice(["GET", "POST", "PUT", "DELETE", "PATCH"])
            args["method"] = method
            phr = self.rng.choice([
                f"Make a {method} request to {u}.", f"Send a {method} request to {u}.",
                f"Do a {method} on {u}."])
        else:
            phr = self.rng.choice([
                f"Make an HTTP request to {u}.", f"Hit the endpoint {u}.",
                f"Send a request to {u}.", f"Call the API at {u}."])
        s = self._tool_sample("http_request", "code", "http_request", args)
        s.prompt = phr
        return s

    def sql_query(self) -> Sample:
        q = self.rng.choice(self.sql)
        return self._string_tool("sql_query", "code", "sql_query", "query", q,
                                 [f"Run the SQL query '{q}'.", f"Execute '{q}' on the database.",
                                  f"Query the database with '{q}'.", f"Run '{q}' against the db."])

    def list_dir(self) -> Sample:
        p = self.rng.choice(self.paths)
        return self._string_tool("list_dir", "code", "list_dir", "path", p,
                                 [f"List the directory {p}.", f"List the contents of {p}.",
                                  f"Show what's in {p}.", f"ls {p}.", f"What files are in {p}?"])

    def find_files(self) -> Sample:
        g = self.rng.choice(self.globs)
        return self._string_tool("find_files", "code", "find_files", "pattern", g,
                                 [f"Find files matching '{g}'.", f"Find all '{g}' files.",
                                  f"Search for files matching '{g}'.", f"Locate '{g}' files."])

    def git_diff(self) -> Sample:
        s = self._tool_sample("git_diff", "code", "git_diff", {})
        s.prompt = self.rng.choice(["Show the git diff.", "What's the diff?", "Git diff.",
                                    "Show me the changes.", "Display the current diff."])
        return s

    def git_status(self) -> Sample:
        s = self._tool_sample("git_status", "code", "git_status", {})
        s.prompt = self.rng.choice(["Show the git status.", "What's the git status?", "Git status.",
                                    "Check the repo status.", "Show me the working tree status."])
        return s

    def install_package(self) -> Sample:
        nm = self.rng.choice(self.packages)
        return self._string_tool("install_package", "code", "install_package", "name", nm,
                                 [f"Install '{nm}'.", f"Install the package '{nm}'.",
                                  f"Add the dependency '{nm}'.", f"pip install '{nm}'.",
                                  f"Set up '{nm}'."])

    def kill_process(self) -> Sample:
        nm = self.rng.choice(self.processes)
        return self._string_tool("kill_process", "code", "kill_process", "name", nm,
                                 [f"Kill the process '{nm}'.", f"Kill '{nm}'.",
                                  f"Stop the '{nm}' process.", f"Terminate '{nm}'."])

    def read_clipboard(self) -> Sample:
        s = self._tool_sample("read_clipboard", "computer_use", "read_clipboard", {})
        s.prompt = self.rng.choice(["Read the clipboard.", "What's on the clipboard?",
                                    "Get the clipboard contents.", "Paste the clipboard.",
                                    "Show me what's copied."])
        return s

    def write_clipboard(self) -> Sample:
        txt = self.rng.choice(self.typed_text)
        return self._string_tool("write_clipboard", "computer_use", "write_clipboard", "text", txt,
                                 [f"Copy '{txt}' to the clipboard.", f"Put '{txt}' on the clipboard.",
                                  f"Copy '{txt}'.", f"Set the clipboard to '{txt}'."])

    def download_file(self) -> Sample:
        u = self.rng.choice(self.urls)
        return self._string_tool("download_file", "code", "download_file", "url", u,
                                 [f"Download the file from {u}.", f"Download {u}.",
                                  f"Fetch the file at {u}.", f"Grab the file from {u}."])

    def unzip(self) -> Sample:
        p = self.rng.choice(self.archives)
        return self._string_tool("unzip", "code", "unzip", "path", p,
                                 [f"Unzip {p}.", f"Extract {p}.", f"Unpack the archive {p}.",
                                  f"Decompress {p}."])

    def env_get(self) -> Sample:
        nm = self.rng.choice(self.envvars)
        return self._string_tool("env_get", "code", "env_get", "name", nm,
                                 [f"Get the env variable '{nm}'.", f"Read the environment variable '{nm}'.",
                                  f"What is '{nm}' set to?", f"Show the value of '{nm}'."])

    def make_dir(self) -> Sample:
        p = self.rng.choice(self.paths)
        return self._string_tool("make_dir", "code", "make_dir", "path", p,
                                 [f"Create the directory {p}.", f"Make a directory {p}.",
                                  f"mkdir {p}.", f"Create folder {p}."])

    def list_processes(self) -> Sample:
        s = self._tool_sample("list_processes", "code", "list_processes", {})
        s.prompt = self.rng.choice(["List the running processes.", "Show running processes.",
                                    "What processes are running?", "List all processes.",
                                    "Show me the process list."])
        return s

    def docker_run(self) -> Sample:
        img = self.rng.choice(self.images)
        return self._string_tool("docker_run", "code", "docker_run", "image", img,
                                 [f"Run a Docker container from '{img}'.", f"Run the '{img}' image.",
                                  f"Start a container from '{img}'.", f"docker run '{img}'."])

    # --- parallel / two-tool calls ("do X and Y" — what people actually want) ---
    _PARALLEL_POOL = ("weather", "web_search", "web_search_implicit", "define", "play_music",
                      "get_news", "read_file",
                      "run_tests", "set_reminder", "set_timer", "calendar_event", "send_email",
                      "open_url", "notion_write", "slack_send", "jira_issue", "grep_search",
                      "git_commit", "calc", "run_command",
                      # computer-use + modern tools (single groundable clause each)
                      "screenshot", "click", "type_text", "key_press", "scroll", "wait",
                      "open_app", "move_cursor", "double_click",
                      "run_python", "edit_file", "git_diff", "git_status", "install_package",
                      "find_files", "list_dir", "kill_process", "sql_query", "env_get",
                      "make_dir", "unzip", "docker_run")

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
        m = [self.weather, self.calc, self.web_search, self.web_search_implicit, self.planner,
             self.define, self.play_music, self.get_news,
             self.read_file, self.write_file, self.grep_search, self.run_command,
             self.git_commit, self.run_tests, self.set_reminder, self.set_timer,
             self.calendar_event, self.send_email, self.open_url, self.notion_write,
             self.slack_send, self.jira_issue,
             # computer-use family
             self.screenshot, self.click, self.double_click, self.type_text, self.key_press,
             self.scroll, self.drag, self.wait, self.move_cursor, self.open_app,
             # modern dev / agentic tools
             self.run_python, self.edit_file, self.apply_patch, self.http_request, self.sql_query,
             self.list_dir, self.find_files, self.git_diff, self.git_status, self.install_package,
             self.kill_process, self.read_clipboard, self.write_clipboard, self.download_file,
             self.unzip, self.env_get, self.make_dir, self.list_processes, self.docker_run,
             self.parallel, self.text]
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
    # All plan-episode types (length-bucketed). Backwards-compatible: the original three
    # (plan_read_test_commit, plan_research, plan_fix_test) are a subset. The gold "plan" of any
    # episode is recoverable via ``episode_plan`` (ordered tool names). Keep this tuple in sync
    # with ``_PLAN_BUILDERS`` below — ``test_planner_data`` asserts they match exactly.
    _PLANNER_TYPES = (
        "plan_no_tool_thanks", "plan_no_tool_greet",
        "plan_single_read", "plan_single_search", "plan_single_timer",
        "plan_research_record", "plan_research_share", "plan_schedule_notify",
        "plan_remind_timer", "plan_read_run",
        "plan_read_test_commit", "plan_research", "plan_locate_inspect_edit",
        "plan_search_open_note", "plan_gui_open_click",
        "plan_fix_test", "plan_implement", "plan_gui_login",
    )

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

    # ---- computer-use (GUI) episode builders ----
    # Realistic desktop/browser flows the planner can learn. Every arg is a literal substring of an
    # earlier turn (the user request or a tool response). Screenshots/keypresses interleave sensibly.
    def _ep_cu_login(self):  # screenshot -> click(field) -> type_text -> key_press(Enter)
        field = self.rng.choice([t for t in self.ui_targets if "field" in t] or self.ui_targets)
        text = self.rng.choice(self.typed_text)
        return [
            self._U(f"Log in: click '{field}', type '{text}', then press Enter."),
            self._A("screenshot", {}),
            self._T("<screen captured>"),
            self._A("click", {"target": field}),
            self._T("focused."),
            self._A("type_text", {"text": text}),
            self._T("typed."),
            self._A("key_press", {"key": "Enter"}),
            self._T("submitted."),
            self._F(f"Logged in: typed '{text}' into '{field}'."),
        ]

    def _ep_cu_open_and_click(self):  # open_app -> click -> screenshot
        app = self.rng.choice(self.apps)
        target = self.rng.choice(self.ui_targets)
        return [
            self._U(f"Open '{app}', click '{target}', and take a screenshot."),
            self._A("open_app", {"name": app}),
            self._T("app launched."),
            self._A("click", {"target": target}),
            self._T("clicked."),
            self._A("screenshot", {}),
            self._T("<screen captured>"),
            self._F(f"Opened '{app}' and clicked '{target}'."),
        ]

    def _ep_cu_search_box(self):  # click(search box) -> type_text -> key_press(Enter) -> screenshot
        text = self.rng.choice(self.typed_text)
        return [
            self._U(f"Click 'the Search box', search for '{text}', and capture the results."),
            self._A("click", {"target": "the Search box"}),
            self._T("focused."),
            self._A("type_text", {"text": text}),
            self._T("typed."),
            self._A("key_press", {"key": "Enter"}),
            self._T("searching."),
            self._A("screenshot", {}),
            self._T("<results captured>"),
            self._F(f"Searched for '{text}'."),
        ]

    def _ep_cu_scroll_click(self):  # scroll(down) -> screenshot -> click
        target = self.rng.choice(self.ui_targets)
        return [
            self._U(f"Scroll down, take a screenshot, then click '{target}'."),
            self._A("scroll", {"direction": "down"}),
            self._T("scrolled."),
            self._A("screenshot", {}),
            self._T("<screen captured>"),
            self._A("click", {"target": target}),
            self._T("clicked."),
            self._F(f"Scrolled down and clicked '{target}'."),
        ]

    def _ep_cu_drag_drop(self):  # screenshot -> drag(source, dest)
        src, dst = self.rng.sample(self.ui_targets, 2)
        return [
            self._U(f"Take a screenshot then drag '{src}' to '{dst}'."),
            self._A("screenshot", {}),
            self._T("<screen captured>"),
            self._A("drag", {"source": src, "dest": dst}),
            self._T("dropped."),
            self._F(f"Dragged '{src}' onto '{dst}'."),
        ]

    def _computer_use_builders(self):
        return {
            "cu_login": self._ep_cu_login, "cu_open_and_click": self._ep_cu_open_and_click,
            "cu_search_box": self._ep_cu_search_box, "cu_scroll_click": self._ep_cu_scroll_click,
            "cu_drag_drop": self._ep_cu_drag_drop,
        }

    _COMPUTER_USE_TYPES = ("cu_login", "cu_open_and_click", "cu_search_box", "cu_scroll_click",
                           "cu_drag_drop")

    def computer_use_episode(self) -> Conversation:
        """A short multi-turn GUI trajectory (open app / click / type / key / scroll / drag /
        screenshot). All args are text-grounded substrings; no pixel coordinates."""
        which = self.rng.choice(self._COMPUTER_USE_TYPES)
        msgs = self._computer_use_builders()[which]()
        return Conversation(messages=msgs, meta={"kind": "computer_use_episode", "type": which})

    def computer_use_episodes(self, n: int) -> list[Conversation]:
        return [self.computer_use_episode() for _ in range(n)]

    # ---- planner-then-execute episode builders ----
    # The plan text is deterministic/canonical so it is exactly learnable and scorable.
    @staticmethod
    def _plan(steps):
        return "Plan: " + " ".join(f"{i + 1}) {s}" for i, s in enumerate(steps))

    # ---- richer multi-step PLAN episodes (planner -> action decomposition, stage 1) ----
    # Each builder returns list[Message]. The gold "plan" is the ordered projection of the
    # episode's tool-call turns onto tool NAMES (see ``episode_plan`` below); each step's args
    # are groundable — copyable either from the composite user request or from an EARLIER tool
    # response (the pointer case, where a path/url is "returned" then consumed downstream).
    # Plan lengths span 1..4 so the planner learns plan-LENGTH control, not just a fixed shape.

    # --- 2-step plans ---
    def _ep_plan_research_record(self):  # web_search -> notion_write (research then record)
        query = self.rng.choice(self.queries)
        content = self.rng.choice(self.notion)
        return [
            self._U(f"Research {query} then note '{content}' in Notion."),
            self._F(self._plan(["search the web", "save a Notion note"])),
            self._A("web_search", {"query": query}),
            self._T(f"Results for {query}."),
            self._A("notion_write", {"content": content}),
            self._T("note saved."),
            self._F(f"Researched {query} and noted '{content}'."),
        ]

    def _ep_plan_research_share(self):  # web_search -> send_email (research then share)
        query = self.rng.choice(self.queries)
        nm = self.rng.choice(self.names)
        return [
            self._U(f"Look up {query} and email {nm} about it."),
            self._F(self._plan(["search the web", "email the result"])),
            self._A("web_search", {"query": query}),
            self._T(f"Results for {query}."),
            self._A("send_email", {"recipient": nm}),
            self._T("email sent."),
            self._F(f"Looked up {query} and emailed {nm}."),
        ]

    def _ep_plan_schedule_notify(self):  # calendar_event -> send_email (schedule then notify)
        title = self.rng.choice(self.events)
        nm = self.rng.choice(self.names)
        return [
            self._U(f"Schedule '{title}' then notify {nm} by email."),
            self._F(self._plan(["create a calendar event", "send an email"])),
            self._A("calendar_event", {"title": title}),
            self._T("event created."),
            self._A("send_email", {"recipient": nm}),
            self._T("email sent."),
            self._F(f"Scheduled '{title}' and notified {nm}."),
        ]

    def _ep_plan_remind_timer(self):  # set_reminder -> set_timer
        task = self.rng.choice(self.tasks)
        dur = self.rng.choice(self.durations)
        return [
            self._U(f"Remind me to {task} and set a timer for {dur}."),
            self._F(self._plan(["set a reminder", "set a timer"])),
            self._A("set_reminder", {"task": task}),
            self._T("reminder set."),
            self._A("set_timer", {"duration": dur}),
            self._T("timer started."),
            self._F(f"Reminder to {task} set and timer for {dur} started."),
        ]

    def _ep_plan_read_run(self):  # read_file -> run_command
        path = self.rng.choice(self.paths)
        cmd = self.rng.choice(self.commands)
        return [
            self._U(f"Read {path} then run '{cmd}'."),
            self._F(self._plan(["read the file", "run the command"])),
            self._A("read_file", {"path": path}),
            self._T("<file contents>"),
            self._A("run_command", {"command": cmd}),
            self._T("command finished."),
            self._F(f"Read {path} and ran '{cmd}'."),
        ]

    # --- 3-step plans ---
    def _ep_plan_locate_inspect_edit(self):  # grep_search -> read_file -> write_file (pointer)
        path = self.rng.choice(self.paths)
        pat = self.rng.choice(self.patterns)
        return [
            self._U(f"Find '{pat}', inspect that file, then edit it."),
            self._F(self._plan(["search the code", "read the file", "edit the file"])),
            self._A("grep_search", {"pattern": pat}),
            self._T(f"{path}:9:    {pat} appears here"),   # path "returned" by grep
            self._A("read_file", {"path": path}),          # grounded from tool response
            self._T("<current source>"),
            self._A("write_file", {"path": path}),         # grounded from tool response
            self._T("written."),
            self._F(f"Located '{pat}' in {path} and edited it."),
        ]

    def _ep_plan_search_open_note(self):  # web_search -> open_url -> notion_write (pointer)
        query = self.rng.choice(self.queries)
        url = self.rng.choice(self.urls)
        content = self.rng.choice(self.notion)
        return [
            self._U(f"Research {query}, open the top link, then note '{content}'."),
            self._F(self._plan(["search the web", "open the top result", "save a Notion note"])),
            self._A("web_search", {"query": query}),
            self._T(f"1. {url} — overview of {query}"),    # url "returned" by search
            self._A("open_url", {"url": url}),             # grounded from tool response
            self._T("<page loaded>"),
            self._A("notion_write", {"content": content}),
            self._T("note saved."),
            self._F(f"Researched {query}, opened {url}, noted '{content}'."),
        ]

    # --- 4-step plans ---
    def _ep_plan_read_test_commit(self):  # 3-step, existing one (kept, lightly varied phrasing)
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

    def _ep_plan_research(self):  # 3-step plan -> search -> (url) -> open_url -> summary (pointer)
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

    def _ep_plan_fix_test(self):  # 4-step: run_tests -> read -> write -> run_tests
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

    def _ep_plan_implement(self):  # 4-step: grep -> (path) -> read -> write -> git_commit (pointer)
        path = self.rng.choice(self.paths)
        pat = self.rng.choice(self.patterns)
        msg = self.rng.choice(self.commits)
        return [
            self._U(f"Implement '{msg}' where '{pat}' lives, then commit."),
            self._F(self._plan(["search the code", "read the file", "edit the file", "commit"])),
            self._A("grep_search", {"pattern": pat}),
            self._T(f"{path}:20:    {pat} ..."),           # path "returned" by grep
            self._A("read_file", {"path": path}),
            self._T("<current implementation>"),
            self._A("write_file", {"path": path}),
            self._T("written."),
            self._A("git_commit", {"message": msg}),
            self._T("Committed def456."),
            self._F(f"Implemented and committed '{msg}'."),
        ]

    # --- GUI plan episodes (computer-use planner flows) ---
    def _ep_plan_gui_login(self):  # 4-step: click -> type_text -> key_press -> screenshot
        field = self.rng.choice([t for t in self.ui_targets if "field" in t] or self.ui_targets)
        text = self.rng.choice(self.typed_text)
        return [
            self._U(f"Log in: click '{field}', type '{text}', press Enter, then screenshot."),
            self._F(self._plan(["click the field", "type the text", "press Enter",
                                "take a screenshot"])),
            self._A("click", {"target": field}),
            self._T("focused."),
            self._A("type_text", {"text": text}),
            self._T("typed."),
            self._A("key_press", {"key": "Enter"}),
            self._T("submitted."),
            self._A("screenshot", {}),
            self._T("<screen captured>"),
            self._F("Logged in and captured the screen."),
        ]

    def _ep_plan_gui_open_click(self):  # 3-step: open_app -> click -> screenshot
        app = self.rng.choice(self.apps)
        target = self.rng.choice(self.ui_targets)
        return [
            self._U(f"Open '{app}', click '{target}', then take a screenshot."),
            self._F(self._plan(["open the app", "click the element", "take a screenshot"])),
            self._A("open_app", {"name": app}),
            self._T("app launched."),
            self._A("click", {"target": target}),
            self._T("clicked."),
            self._A("screenshot", {}),
            self._T("<screen captured>"),
            self._F(f"Opened '{app}' and clicked '{target}'."),
        ]

    # --- 1-step plans (teach the planner to STOP after one tool) ---
    def _ep_plan_single_read(self):
        path = self.rng.choice(self.paths)
        return [
            self._U(f"Just read {path}."),
            self._F(self._plan(["read the file"])),
            self._A("read_file", {"path": path}),
            self._T("<file contents>"),
            self._F(f"Read {path}."),
        ]

    def _ep_plan_single_search(self):
        query = self.rng.choice(self.queries)
        return [
            self._U(f"Only search the web for {query}."),
            self._F(self._plan(["search the web"])),
            self._A("web_search", {"query": query}),
            self._T(f"Results for {query}."),
            self._F(f"Searched for {query}."),
        ]

    def _ep_plan_single_timer(self):
        dur = self.rng.choice(self.durations)
        return [
            self._U(f"Set a timer for {dur}, nothing else."),
            self._F(self._plan(["set a timer"])),
            self._A("set_timer", {"duration": dur}),
            self._T("timer started."),
            self._F(f"Timer for {dur} started."),
        ]

    # --- 0-step "plans": a trivial request answered with text, no tool (don't over-plan) ---
    def _ep_plan_no_tool_thanks(self):
        return [
            self._U("Thanks for your help!"),
            self._F("Plan: just reply."),
            self._F("You're welcome!"),
        ]

    def _ep_plan_no_tool_greet(self):
        nm = self.rng.choice(self.names)
        return [
            self._U(f"Just say hello to {nm}."),
            self._F("Plan: just reply."),
            self._F(f"Hello, {nm}!"),
        ]

    # length-bucketed registry; ``plan_episode`` samples across buckets for plan-length variety
    _PLAN_BUILDERS = {
        # 0-step (text only, no tool)
        "plan_no_tool_thanks": ("_ep_plan_no_tool_thanks", 0),
        "plan_no_tool_greet": ("_ep_plan_no_tool_greet", 0),
        # 1-step
        "plan_single_read": ("_ep_plan_single_read", 1),
        "plan_single_search": ("_ep_plan_single_search", 1),
        "plan_single_timer": ("_ep_plan_single_timer", 1),
        # 2-step
        "plan_research_record": ("_ep_plan_research_record", 2),
        "plan_research_share": ("_ep_plan_research_share", 2),
        "plan_schedule_notify": ("_ep_plan_schedule_notify", 2),
        "plan_remind_timer": ("_ep_plan_remind_timer", 2),
        "plan_read_run": ("_ep_plan_read_run", 2),
        # 3-step
        "plan_read_test_commit": ("_ep_plan_read_test_commit", 3),
        "plan_research": ("_ep_plan_research", 3),
        "plan_locate_inspect_edit": ("_ep_plan_locate_inspect_edit", 3),
        "plan_search_open_note": ("_ep_plan_search_open_note", 3),
        "plan_gui_open_click": ("_ep_plan_gui_open_click", 3),
        # 4-step
        "plan_fix_test": ("_ep_plan_fix_test", 4),
        "plan_implement": ("_ep_plan_implement", 4),
        "plan_gui_login": ("_ep_plan_gui_login", 4),
    }

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
        return {name: getattr(self, attr) for name, (attr, _len) in self._PLAN_BUILDERS.items()}

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

    def _build_plan_episode(self, which: str) -> Conversation:
        """Build a named plan episode and tag meta with its gold plan (ordered tool names) and
        plan length, both *derived* from the episode's tool-call sequence (no schema change)."""
        msgs = self._planner_builders()[which]()
        plan = episode_plan(Conversation(messages=msgs))
        return Conversation(messages=msgs, meta={"kind": "planner_episode", "type": which,
                                                 "plan": plan, "plan_len": len(plan)})

    def planner_episode(self) -> Conversation:
        """A planner-then-execute trajectory: a canonical NUMBERED text plan turn, then each step
        as a tool-call turn with tool responses between, then a final summary (teaches plan->act).
        Backwards-compatible: only multi-step types (>=1 tool call, plan starts ``Plan: 1)``); the
        0-step "don't over-plan" cases are reached via ``plan_episode``/``plan_episodes`` instead.
        ``meta['plan']`` is the gold ordered tool-name list; ``meta['plan_len']`` its length."""
        multi = [t for t, (_a, ln) in self._PLAN_BUILDERS.items() if ln >= 1]
        which = self.rng.choice(multi)
        return self._build_plan_episode(which)

    def plan_episode(self) -> Conversation:
        """Sample a plan episode with *plan-length variety*: pick a length bucket (0..4) uniformly,
        then a type within it, so the planner sees STOP-after-one and don't-over-plan cases, not
        only long plans. Same Conversation shape as ``planner_episode`` (alias-friendly)."""
        by_len: dict[int, list[str]] = {}
        for name, (_attr, ln) in self._PLAN_BUILDERS.items():
            by_len.setdefault(ln, []).append(name)
        length = self.rng.choice(sorted(by_len))
        which = self.rng.choice(by_len[length])
        return self._build_plan_episode(which)

    def coding_episodes(self, n: int) -> list[Conversation]:
        return [self.coding_episode() for _ in range(n)]

    def productivity_episodes(self, n: int) -> list[Conversation]:
        return [self.productivity_episode() for _ in range(n)]

    def planner_episodes(self, n: int) -> list[Conversation]:
        return [self.planner_episode() for _ in range(n)]

    def plan_episodes(self, n: int) -> list[Conversation]:
        """`n` plan episodes sampled for plan-length variety. Entry point for stage-1 planner
        training/eval: each Conversation's gold plan is ``meta['plan']`` (== ``episode_plan(ep)``)
        and its per-step grounded ToolCalls are the assistant tool-call turns, in order."""
        return [self.plan_episode() for _ in range(n)]

    def episodes(self, n: int, mix: bool = True) -> list[Conversation]:
        """Sample `n` multi-turn episodes. With ``mix=True`` (default) the pool spans coding +
        productivity + planner trajectories so the flywheel/eval see the full diversity; with
        ``mix=False`` it returns coding-only episodes (the original behaviour)."""
        if not mix:
            return [self.coding_episode() for _ in range(n)]
        builders = [self.coding_episode, self.productivity_episode, self.planner_episode,
                    self.computer_use_episode]
        # weight coding a bit higher (it has the most types), then productivity, planner, GUI
        weights = [0.4, 0.25, 0.2, 0.15]
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


# ---- plan helpers (free functions; no schema change — operate on the existing episode shape) --
def episode_plan(ep: Conversation) -> list[str]:
    """The gold PLAN of an episode: the ordered projection of its assistant tool-call turns onto
    tool NAMES. This is the stage-1 planner target — the planner emits this list, and the existing
    action decoder grounds each name into a concrete ToolCall. A 0-step plan (a trivial text-only
    request) yields ``[]``."""
    return [m.tool_calls[0].name
            for m in ep.messages
            if m.role == Role.assistant and m.tool_calls]


def episode_steps(ep: Conversation) -> list[ToolCall]:
    """The per-step grounded ToolCalls of an episode, in order — one per planned step. Zipped with
    ``episode_plan(ep)`` they give (tool_name, grounded_args) for each step."""
    return [m.tool_calls[0]
            for m in ep.messages
            if m.role == Role.assistant and m.tool_calls]


# ---- curriculum ordering (LFM2-style easy->hard) ---------------------------------------------
# LFM2 orders pretraining data by empirical success probability (easy first, hard later). We port
# the *principle* to tool-calling SFT with a transparent, deterministic proxy for difficulty built
# from signals already present in each Sample — no schema change, no model in the loop:
#
#   score = (W_PARALLEL  * (n_tool_calls - 1)     # >1 call (the "X and Y" parallel turns) is hard
#          + W_ARGS      * max(0, n_required_args - 1)  # multi-arg (e.g. weather city+unit) is hard
#          + W_HAS_ARG   * has_any_arg             # a single copy-arg call is harder than no-arg/text
#          + W_ABSTAIN   * is_abstention           # "don't call a tool" negatives are subtle
#          + W_PROMPT    * prompt_len_bucket)      # longer phrasings, mild tie-breaker
#
# A bare text turn or a no-arg tool call (run_tests) scores ~0 (easiest); a two-call parallel turn
# with copy args scores highest (hardest). Ties broken by a stable hash of the prompt so the order
# is fully deterministic and independent of the input list order.
CURRICULUM_WEIGHTS = {
    "parallel": 3.0,   # per extra tool call beyond the first
    "args": 1.5,       # per required arg beyond the first
    "has_arg": 0.6,    # single copy-arg call vs no-arg/text
    "abstain": 1.0,    # abstention / irrelevance negative
    "prompt": 0.25,    # per ~40-char bucket of prompt length (tie-breaker scale)
}


def difficulty_score(s: "Sample", weights: dict | None = None) -> float:
    """Transparent easy->hard difficulty score for one SFT `Sample` (higher = harder).

    Uses only signals already on the Sample (number of tool calls, number/copy of args, abstention,
    prompt length). Deterministic and side-effect free. See ``CURRICULUM_WEIGHTS`` for the recipe.
    """
    w = weights or CURRICULUM_WEIGHTS
    # number of tool calls in the turn: parallel samples carry `calls`; single tool/text => 1.
    n_calls = len(s.calls) if s.calls else 1
    # required args across the call(s).
    if s.calls:
        n_args = sum(len(c.get("arguments", {})) for c in s.calls)
        has_arg = 1.0 if n_args > 0 else 0.0
    elif s.kind == "tool":
        try:
            args = json.loads(s.ref_args) if s.ref_args else {}
        except (json.JSONDecodeError, TypeError):
            args = {}
        n_args = len(args)
        has_arg = 1.0 if n_args > 0 else 0.0
    else:
        n_args, has_arg = 0, 0.0
    is_abstain = 1.0 if s.category == "no_tool" else 0.0
    prompt_bucket = len(s.prompt) / 40.0
    return (w["parallel"] * (n_calls - 1)
            + w["args"] * max(0, n_args - 1)
            + w["has_arg"] * has_arg
            + w["abstain"] * is_abstain
            + w["prompt"] * prompt_bucket)


def curriculum_order(samples: list, weights: dict | None = None) -> list:
    """Return `samples` reordered easy->hard by ``difficulty_score`` (ascending). Deterministic:
    ties are broken by a stable hash of (target, prompt), so the result does not depend on the
    input order. Opt-in — callers choose this over a shuffle. Does NOT mutate the input list."""
    import hashlib

    def _key(s):
        h = hashlib.sha1(f"{s.target}\x00{s.prompt}".encode()).hexdigest()
        return (difficulty_score(s, weights), h)

    return sorted(samples, key=_key)


def synthesize(config_path: str) -> None:  # CLI entry retained
    raise NotImplementedError("Use scripts/flywheel.py — Generator drives data generation in-process")
