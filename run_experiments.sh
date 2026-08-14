#!/usr/bin/env bash
# Drive the documented LocalAgent experiment battery (docs/EXPERIMENTS.md + figures/README.md)
# on the thor2-h100 GPU. Three lanes run concurrently; within a lane, steps are ordered by their
# artifact dependencies (logit_analysis needs the 30M checkpoint from analyze_loop).
cd /home/jovyan/sbchoi/localagent || exit 1
export PYTHONUNBUFFERED=1
PY=.venv/bin/python
mkdir -p explog
STATUS=explog/STATUS.txt

step() { # <name> <command...>
  local name="$1"; shift
  local start=$SECONDS
  echo "START $name $(date -Is)" >> "$STATUS"
  "$@" > "explog/$name.log" 2>&1
  local rc=$?
  echo "END   $name rc=$rc secs=$((SECONDS-start)) $(date -Is)" >> "$STATUS"
  return $rc
}

lane_a() {
  step 01-06_18_flywheel $PY scripts/flywheel.py --rounds 5
}

lane_b() {
  step 07_analyze_loop_1m $PY scripts/analyze_loop.py --rounds 5
  step 09_analyze_loop_30m $PY scripts/analyze_loop.py --rounds 5 --model configs/model/tiny-30m-byte.yaml
  step 17_logit_analysis $PY scripts/logit_analysis.py
}

lane_c() {
  step 10-11_benchmark $PY scripts/benchmark.py --prompt-len 64 --decode 64
  step 12_tool_scale $PY scripts/tool_scale_analysis.py
  step 13_distill_demo $PY scripts/distill_demo.py
  step 14_codebench $PY scripts/codebench_eval.py
  step 15_example_scaling $PY scripts/example_scaling.py
  step 16_scenarios $PY scripts/scenarios_eval.py
  step xx_toolcall_eval $PY scripts/toolcall_eval.py
}

lane_a & lane_b & lane_c &
wait
echo "ALL_EXPERIMENTS_DONE $(date -Is)" >> "$STATUS"
