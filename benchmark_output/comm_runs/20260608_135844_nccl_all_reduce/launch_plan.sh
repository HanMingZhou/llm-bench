#!/usr/bin/env bash
set -euo pipefail
docker run --rm --name llm-bench-nccl-1780927124 --gpus all --shm-size 16g --ipc=host ghcr.io/coreweave/nccl-tests:12.2.2-cudnn8-devel-ubuntu22.04-nccl2.23.4-1-2ff05b2 /opt/nccl-tests/build/all_reduce_perf -b 8 -e 1G -f 2 -g 1 -n 100 -w 20
