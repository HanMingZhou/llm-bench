#!/usr/bin/env bash
set -euo pipefail
docker run --rm --name llm-bench-vllm-1780927805 -p 8000:8000 -v /root/.cache/huggingface:/root/.cache/huggingface -e HF_HOME=/root/.cache/huggingface -v /root/.cache/modelscope:/root/.cache/modelscope --gpus all --shm-size 16g --ipc=host vllm-openai:latest /root/.cache/modelscope/hub/models/Qwen/Qwen3.5-9B --host 0.0.0.0 --port 8000 --served-model-name Qwen3.5-9B --tensor-parallel-size 2 --gpu-memory-utilization 0.9 --max-model-len 4096
