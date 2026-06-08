import json

import pytest

from llm_bench.commands.infer import _run_inference, _validate_for_run
from llm_bench.config import BenchConfig


def test_validate_requires_command():
    config = BenchConfig()
    config.backend.name = "vllm"
    config.backend.image = "vllm/vllm-openai:latest"
    config.backend.model_name = "Qwen/Qwen2.5-7B-Instruct"
    with pytest.raises(ValueError, match="Missing container command"):
        _validate_for_run(config)


def test_validate_requires_model_name():
    config = BenchConfig()
    config.backend.name = "vllm"
    config.backend.image = "vllm/vllm-openai:latest"
    config.backend.command = ["vllm", "serve", "/model"]
    with pytest.raises(ValueError, match="--model-name is required"):
        _validate_for_run(config)


def test_validate_requires_image():
    config = BenchConfig()
    config.backend.name = "vllm"
    config.backend.image = ""
    config.backend.model_name = "Qwen/Qwen2.5-7B-Instruct"
    config.backend.command = ["vllm", "serve", "/model"]
    with pytest.raises(ValueError, match="--image is required"):
        _validate_for_run(config)


def test_validate_accepts_dry_run():
    config = BenchConfig()
    config.backend.name = "dry-run"
    _validate_for_run(config)


def test_validate_transformers_requires_model_path():
    config = BenchConfig()
    config.backend.name = "transformers"
    with pytest.raises(ValueError, match="--model-path is required"):
        _validate_for_run(config)


def test_validate_transformers_with_model_path():
    config = BenchConfig()
    config.backend.name = "transformers"
    config.transformers.model_path = "/mnt/models/qwen"
    _validate_for_run(config)


def test_backend_registry_returns_transformers():
    from llm_bench.backends.registry import get_backend
    from llm_bench.backends.transformers_backend import TransformersBackend

    config = BenchConfig()
    config.backend.name = "transformers"
    backend = get_backend(config)
    assert isinstance(backend, TransformersBackend)


def test_precheck_failure_is_archived(tmp_path):
    config = BenchConfig()
    config.backend.name = "vllm"
    config.backend.image = "missing-image-for-test:latest"
    config.backend.model_name = "missing-model"
    config.backend.command = ["vllm", "serve", "/missing"]
    config.report.output_dir = str(tmp_path / "runs")
    config.report.run_name = "precheck-failure"

    with pytest.raises(RuntimeError):
        _run_inference(config, {"backend": {"name": "vllm"}})

    run_dir = tmp_path / "runs" / "precheck-failure"
    assert (run_dir / "environment.json").exists()
    assert (run_dir / "reports" / "inference_report.md").exists()
    summary = json.loads((run_dir / "metrics.summary.json").read_text(encoding="utf-8"))
    assert summary["success_requests"] == 0
