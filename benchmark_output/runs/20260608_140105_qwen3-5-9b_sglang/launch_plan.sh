#!/usr/bin/env bash
set -euo pipefail
docker run --rm --name llm-bench-sglang-1780927265 -p 30000:30000 -v /root/.cache/huggingface:/root/.cache/huggingface -e HF_HOME=/root/.cache/huggingface -v /root/.cache/modelscope:/root/.cache/modelscope --gpus all --shm-size 16g --ipc=host sglang:latest sglang serve --model-path /root/.cache/modelscope/hub/models/Qwen/Qwen3.5-9B --host 0.0.0.0 --port 30000 --served-model-name Qwen3.5-9B --tp 2 --mem-fraction-static 0.9 --context-length 4096
