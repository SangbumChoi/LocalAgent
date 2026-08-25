#!/usr/bin/env bash
cd /home/jovyan/sbchoi/localagent
export PYTHONPATH=src PYTHONUNBUFFERED=1
.venv/bin/localagent export onnx runs/flywheel/ultra-tiny.pt demos/web/onnx/localagent.onnx 2>&1 | tail -20
echo "EXPORT_RC=$?"
ls -lh demos/web/onnx/
