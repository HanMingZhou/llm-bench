import os
import subprocess
import sys
from pathlib import Path

import pytest


pytestmark = pytest.mark.integration


def _run_enabled(name: str) -> bool:
    return os.environ.get("LLM_BENCH_RUN_INTEGRATION") == "1" and os.environ.get(name) == "1"


def _run_cmd(args: list[str], tmp_path: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "llm_bench", *args, "--output-dir", str(tmp_path / "runs")],
        text=True,
        capture_output=True,
        timeout=int(os.environ.get("LLM_BENCH_INTEGRATION_TIMEOUT", "1200")),
    )


def test_vllm_backend_real_service(tmp_path: Path):
    if not _run_enabled("LLM_BENCH_TEST_VLLM"):
        pytest.skip("set LLM_BENCH_RUN_INTEGRATION=1 and LLM_BENCH_TEST_VLLM=1")
    model_name = os.environ["LLM_BENCH_MODEL_NAME"]
    image = os.environ.get("LLM_BENCH_VLLM_IMAGE", "vllm/vllm-openai:latest")
    container_command = os.environ.get(
        "LLM_BENCH_VLLM_COMMAND",
        # vllm/vllm-openai entrypoint is already `vllm serve`; pass only model + args.
        f"{model_name} --host 0.0.0.0 --port 8000 --tensor-parallel-size 1",
    ).split()
    proc = _run_cmd(
        [
            "infer",
            "--backend", "vllm",
            "--image", image,
            "--model-name", model_name,
            "--port", "8000",
            "--input-tokens", "8",
            "--output-tokens", "4",
            "--concurrency", "1",
            "--total-requests", "1",
            "--warmup-requests", "0",
            "--",
            *container_command,
        ],
        tmp_path,
    )
    assert proc.returncode == 0, proc.stderr


def test_sglang_backend_real_service(tmp_path: Path):
    if not _run_enabled("LLM_BENCH_TEST_SGLANG"):
        pytest.skip("set LLM_BENCH_RUN_INTEGRATION=1 and LLM_BENCH_TEST_SGLANG=1")
    model_name = os.environ["LLM_BENCH_MODEL_NAME"]
    image = os.environ.get("LLM_BENCH_SGLANG_IMAGE", "lmsysorg/sglang:latest")
    container_command = os.environ.get(
        "LLM_BENCH_SGLANG_COMMAND",
        f"python -m sglang.launch_server --model-path {model_name} --host 0.0.0.0 --port 8000 --tp 1",
    ).split()
    proc = _run_cmd(
        [
            "infer",
            "--backend", "sglang",
            "--image", image,
            "--model-name", model_name,
            "--port", "8000",
            "--input-tokens", "8",
            "--output-tokens", "4",
            "--concurrency", "1",
            "--total-requests", "1",
            "--warmup-requests", "0",
            "--",
            *container_command,
        ],
        tmp_path,
    )
    assert proc.returncode == 0, proc.stderr
