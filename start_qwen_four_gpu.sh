#!/usr/bin/env bash
set -euo pipefail
repo=/data/cxr25/zhx/Proactive_AI_Agent
py="$repo/zhx/miniforge3/envs/agent/bin/python"
mkdir -p "$repo/logs" "$repo/data/egolife/runs/qwen_services"
for pair in 2:8001 4:8002 6:8003 7:8004; do
  gpu=${pair%:*}; port=${pair#*:}
  log="$repo/logs/qwen_api_gpu${gpu}_${port}_20261103.log"
  pidfile="$repo/data/egolife/runs/qwen_services/gpu${gpu}_${port}.pid"
  if curl -fsS --max-time 2 "http://127.0.0.1:${port}/health" >/dev/null; then
    echo "port ${port} already healthy"; continue
  fi
  CUDA_VISIBLE_DEVICES="$gpu" WHISPER_DEVICE=cpu PORT="$port" nohup "$py" -u "$repo/code/api_server.py" >"$log" 2>&1 < /dev/null &
  echo $! > "$pidfile"
  echo "started gpu=${gpu} port=${port} pid=$! log=${log}"
done
