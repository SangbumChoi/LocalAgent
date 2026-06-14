"""Colab bootstrap — runs ON the Colab runtime via `colab exec -f scripts/colab_bootstrap.py`.

`colab exec` transmits this file to the remote kernel and runs it (Python only, no shell), so we
clone the repo, install it, and run the same GPU pipeline as the HF Job. Config via the constants
below (the launcher `scripts/launch_colab.sh` can sed them, or edit here). HF push works if the
runtime is HF-authenticated (the launcher uploads a token first).
"""
import os
import subprocess

# --- config (launch_colab.sh overrides these) ---
REPO = "https://github.com/sangbumchoi/localagent"
BRANCH = "claude/tiny-llm-agents-5C3c0"
PUSH_REPO = "danelcsb/localagent-30m-v2"
STAGES = "pretrain,sft,grpo"
WORKDIR = "/content/localagent"


def sh(cmd):
    print(f"$ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True)


# 1. code: clone (or update) the repo at the branch
if os.path.isdir(WORKDIR):
    sh(f"cd {WORKDIR} && git fetch --depth 1 origin {BRANCH} && git checkout {BRANCH} && git pull")
else:
    sh(f"git clone --depth 1 -b {BRANCH} {REPO} {WORKDIR}")
os.chdir(WORKDIR)

# 2. deps + the package
sh("pip install -q -e . 'datasets>=2.19' 'huggingface_hub>=0.24' safetensors pyyaml")

# 3. HF auth for the push (token uploaded to /content/hf_token by the launcher, optional)
if os.path.exists("/content/hf_token"):
    from huggingface_hub import login
    login(open("/content/hf_token").read().strip())

# 4. run the full pipeline (CUDA auto-detected by train_job.py)
push = f"--push {PUSH_REPO}" if PUSH_REPO else ""
sh(f"python scripts/train_job.py --real --stages {STAGES} "
   f"--pretrain-steps 6000 --sft-steps 3000 --grpo-steps 300 --batch 64 --seq-len 512 {push}")
print("COLAB_BOOTSTRAP_DONE", flush=True)
