cd /home/jovyan/sbchoi/localagent
mkdir -p experiments
# thor2 queue: skeleton factorial first (fresh-pool controls already exist/running there),
# then the la64k-16k variant once pool-64k exists.
cat > experiments/queue-thor2.txt <<'Q'
bash skeleton_arm.sh skel-96m skeleton
bash skeleton_arm.sh anti-96m ffn
bash skeleton_arm.sh fullproj-96m blocks
until [ -f data/shards/pool-64k/manifest.json ]; do sleep 300; done; LOCALAGENT_PARAM_BUDGET=140000000 bash run_la64k.sh 16k
Q
# face queue: embed-only skeleton arm + the ToolBench replicates for the 95.3M floor claim.
cat > experiments/queue-face.txt <<'Q'
bash skeleton_arm.sh embed-96m embed
cd /home/jovyan/sbchoi/localagent && export PYTHONPATH=src && .venv/bin/python scripts/eval_suite.py --model catalog:runs/sft-arm3-96m/latest.pt --rows 200 --device cuda --suites toolbench --out runs/evalsuite/arm3-96m-tbr.json
Q
echo QUEUES_SEEDED; wc -l experiments/queue-*.txt
