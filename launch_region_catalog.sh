#!/usr/bin/env bash
cd /home/jovyan/sbchoi/localagent || exit 1
setsid nohup bash run_region_catalog.sh attn ffn embed norms early late no_embed full scratch \
  </dev/null > explog/region_catalog_driver.log 2>&1 &
echo LAUNCHED $!
