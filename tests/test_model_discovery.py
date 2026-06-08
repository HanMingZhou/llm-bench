import pytest

from llm_bench.config import BenchConfig
from llm_bench.environment import (
    discover_docker_images,
    discover_model_paths,
    enforce_runtime_requirements,
    model_candidates,
)


def test_discover_huggingface_snapshot(tmp_path):
    snapshot = tmp_path / "huggingface" / "hub" / "models--Qwen--Qwen2.5-7B-Instruct" / "snapshots" / "abcdef123456"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}", encoding="utf-8")

    models = discover_model_paths(roots=[tmp_path / "huggingface" / "hub"])

    assert models == [
        {
            "source": "Hugging Face",
            "name": "Qwen/Qwen2.5-7B-Instruct@abcdef12",
            "path": str(snapshot),
        }
    ]


def test_discover_modelscope_model(tmp_path):
    model = tmp_path / "modelscope" / "hub" / "models" / "Qwen" / "Qwen2.5-7B-Instruct"
    model.mkdir(parents=True)
    (model / "model.safetensors").write_text("", encoding="utf-8")

    models = discover_model_paths(roots=[tmp_path / "modelscope" / "hub"])

    assert models == [
        {
            "source": "ModelScope",
            "name": "Qwen/Qwen2.5-7B-Instruct",
            "path": str(model),
        }
    ]


def test_model_candidates_prefers_huggingface_snapshot(monkeypatch, tmp_path):
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    snapshot = tmp_path / ".cache" / "huggingface" / "hub" / "models--Qwen--Qwen2.5-7B-Instruct" / "snapshots" / "abcdef123456"
    snapshot.mkdir(parents=True)
    (snapshot / "config.json").write_text("{}", encoding="utf-8")

    candidates = model_candidates("", "Qwen/Qwen2.5-7B-Instruct")
    existing = [candidate for candidate in candidates if candidate.exists()]

    assert existing[0] == snapshot


def test_enforce_skips_when_disabled():
    config = BenchConfig()
    config.skip_env_check = True
    # No assertion needed; just verify it doesn't raise.
    enforce_runtime_requirements(config, {})


def test_enforce_dry_run_skips_check():
    config = BenchConfig()
    config.backend.name = "dry-run"
    enforce_runtime_requirements(config, {})


def test_enforce_reports_docker_failure():
    config = BenchConfig()
    config.backend.name = "vllm"
    config.backend.image = "vllm/vllm-openai:latest"
    runtime = {
        "docker": {"installed": False, "daemon_ok": False, "image_exists": False, "error": "no docker"},
        "port": {"available": True},
        "gpu": {"gpu_available": True},
        "disk": {"free_gb": 10},
    }
    with pytest.raises(RuntimeError, match="docker is not installed"):
        enforce_runtime_requirements(config, runtime)


def test_enforce_reports_missing_image():
    config = BenchConfig()
    config.backend.name = "vllm"
    config.backend.image = "vllm/vllm-openai:latest"
    runtime = {
        "docker": {"installed": True, "daemon_ok": True, "image_exists": False},
        "port": {"available": True},
        "gpu": {"gpu_available": True},
        "disk": {"free_gb": 10},
    }
    with pytest.raises(RuntimeError, match="docker image does not exist locally"):
        enforce_runtime_requirements(config, runtime)


def test_enforce_reports_port_busy():
    config = BenchConfig()
    config.backend.name = "vllm"
    config.backend.image = "vllm:test"
    runtime = {
        "docker": {"installed": True, "daemon_ok": True, "image_exists": True},
        "port": {"available": False},
        "gpu": {"gpu_available": True},
        "disk": {"free_gb": 10},
    }
    with pytest.raises(RuntimeError, match="port is not available"):
        enforce_runtime_requirements(config, runtime)


def test_enforce_reports_no_gpu():
    config = BenchConfig()
    config.backend.name = "vllm"
    config.backend.image = "vllm:test"
    runtime = {
        "docker": {"installed": True, "daemon_ok": True, "image_exists": True},
        "port": {"available": True},
        "gpu": {"gpu_available": False},
        "disk": {"free_gb": 10},
    }
    with pytest.raises(RuntimeError, match="GPU is not visible"):
        enforce_runtime_requirements(config, runtime)


def test_discover_docker_images_prioritizes_backend(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/docker" if name == "docker" else None)

    def fake_run(_cmd):
        return {
            "returncode": 0,
            "stdout": "\n".join(
                [
                    "ubuntu:22.04\timg1\t80MB",
                    "lmsysorg/sglang:latest\timg2\t18GB",
                    "vllm/vllm-openai:latest\timg3\t20GB",
                    "<none>:<none>\timg4\t1GB",
                ]
            ),
            "stderr": "",
        }

    monkeypatch.setattr("llm_bench.environment._run", fake_run)
    images = discover_docker_images("vllm")
    assert images[0]["name"] == "vllm/vllm-openai:latest"
    assert {image["name"] for image in images} == {
        "vllm/vllm-openai:latest",
        "ubuntu:22.04",
        "lmsysorg/sglang:latest",
    }


def test_discover_docker_images_prioritizes_nccl(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda name: "/usr/bin/docker" if name == "docker" else None)

    def fake_run(_cmd):
        return {
            "returncode": 0,
            "stdout": "\n".join(
                [
                    "quay.io/jupyter/scipy-notebook:latest\timg1\t3.68GB",
                    "ghcr.io/coreweave/nccl-tests:12.2.2-cudnn8-devel-ubuntu22.04-nccl2.23.4-1-2ff05b2\timg2\t13.3GB",
                ]
            ),
            "stderr": "",
        }

    monkeypatch.setattr("llm_bench.environment._run", fake_run)
    images = discover_docker_images("nccl")
    assert "nccl-tests" in images[0]["name"]
