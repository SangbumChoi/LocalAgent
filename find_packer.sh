cd /home/jovyan/sbchoi/localagent
grep -rln "generations" scripts/*.py | head -6
grep -rn "def .*pack\|corpus-staging\|staging.sqlite" scripts/prepare_corpus.py scripts/freeze_corpus.py src/localagent/data/pretrain_corpus.py 2>/dev/null | head -8
.venv/bin/localagent --help 2>&1 | head -20
