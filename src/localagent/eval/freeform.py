"""Hand-authored free-form dispatch eval — the honest out-of-distribution test.

These queries are written in natural, varied phrasing that deliberately does NOT follow the synthetic
templates, so accuracy here measures whether tool *selection* generalizes (not template memorization).
Each entry is (query, gold_tool). Arg values are intentionally embedded as literal substrings so the
pointer-copy path is still scorable, but the primary metric this set drives is SELECTION accuracy.
"""

from __future__ import annotations

# (query, gold tool name). Gold reflects the most appropriate single tool in STANDARD_TOOLS.
FREEFORM_EVAL: list[tuple[str, str]] = [
    # --- web_search / knowledge ---
    ("What is the color of a monkey?", "web_search"),
    ("Look up who invented the telephone.", "web_search"),
    ("Google the tallest mountain in Africa.", "web_search"),
    ("Find out when the Eiffel Tower was built.", "web_search"),
    ("Can you search for the best ramen in Tokyo?", "web_search"),
    # --- define ---
    ("What does the word ephemeral mean?", "define"),
    ("Define photosynthesis for me.", "define"),
    ("Give me the definition of entropy.", "define"),
    # --- open_url / http ---
    ("Open https://github.com/pytorch/pytorch in the browser.", "open_url"),
    ("Go to example.com and load the homepage.", "open_url"),
    ("Fetch the JSON from https://api.github.com/repos/torvalds/linux.", "http_request"),
    ("Make a GET request to https://httpbin.org/get.", "http_request"),
    # --- download_file ---
    ("Download the dataset from https://example.com/data.zip.", "download_file"),
    ("Grab the file at https://example.com/report.pdf and save it.", "download_file"),
    # --- get_news ---
    ("Any recent news about electric vehicles?", "get_news"),
    ("What's the latest headlines on climate policy?", "get_news"),
    # --- list_dir / find_files ---
    ("List the files in the src directory.", "list_dir"),
    ("Show me what's inside the build folder.", "list_dir"),
    ("ls the current directory.", "list_dir"),
    ("Find all python files under tests.", "find_files"),
    # --- grep_search ---
    ("Search the codebase for the function train_step.", "grep_search"),
    ("Grep for TODO comments in the repo.", "grep_search"),
    ("Where is the string 'eos_id' used?", "grep_search"),
    # --- make_dir ---
    ("Make a directory called build.", "make_dir"),
    ("Create a new folder named output.", "make_dir"),
    ("mkdir logs please.", "make_dir"),
    # --- run_command / run_python / run_tests ---
    ("Run the test suite.", "run_tests"),
    ("Execute the unit tests.", "run_tests"),
    ("Run the python script train.py.", "run_python"),
    ("Run the command echo hello in the shell.", "run_command"),
    # --- read/write/edit file ---
    ("Read the file data/loader.py.", "read_file"),
    ("Open and show me the contents of README.md.", "read_file"),
    ("Write a config to settings.yaml.", "write_file"),
    # --- git ---
    ("What's the git status of the repo?", "git_status"),
    ("Show me the git diff.", "git_diff"),
    ("Commit the staged changes with message fix bug.", "git_commit"),
    # --- app_action ---
    ("Email Dana the quarterly report.", "send_email"),
    ("Send a slack message to the team channel saying deploy done.", "slack_send"),
    ("Add a calendar event for the standup at 9am.", "calendar_event"),
    ("Remind me to water the plants.", "set_reminder"),
    # --- compute ---
    ("What is 18 * 24?", "calculator"),
    ("Compute 144 divided by 12.", "calculator"),
    # --- weather / music ---
    ("What's the weather like in Oslo?", "get_weather"),
    ("Play Bohemian Rhapsody.", "play_music"),
]
