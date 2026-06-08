import pytest

from llm_bench.backends.docker_serving import DockerServingBackend
from llm_bench.config import BenchConfig


def _baseline(name: str = "vllm") -> BenchConfig:
    config = BenchConfig()
    config.backend.name = name
    config.backend.image = f"{name}:test"
    config.backend.model_name = "Qwen/Qwen2.5-7B-Instruct"
    config.backend.port = 8000
    config.backend.command = [
        "vllm" if name == "vllm" else "python",
        "serve" if name == "vllm" else "-m",
        "/models/qwen" if name == "vllm" else "sglang.launch_server",
    ]
    return config


def test_docker_command_contains_required_flags():
    config = _baseline("vllm")
    cmd = DockerServingBackend("vllm")._docker_run_command(config, "container-x")
    assert cmd[0:3] == ["docker", "run", "--rm"]
    assert "--name" in cmd and "container-x" in cmd
    assert "-p" in cmd
    port_idx = cmd.index("-p")
    assert cmd[port_idx + 1] == "8000:8000"
    image_idx = cmd.index("vllm:test")
    assert cmd[image_idx + 1:] == config.backend.command


def test_keep_container_skips_rm():
    config = _baseline("vllm")
    config.backend.keep_container = True
    cmd = DockerServingBackend("vllm")._docker_run_command(config, "container")
    assert "--rm" not in cmd


def test_hf_cache_and_token_injected_when_set():
    config = _baseline("vllm")
    config.backend.hf_cache = "/data/hf"
    config.backend.hf_token = "hf_abc"
    cmd = DockerServingBackend("vllm")._docker_run_command(config, "container")
    assert "-v" in cmd
    assert "/data/hf:/root/.cache/huggingface" in cmd
    assert "HF_HOME=/root/.cache/huggingface" in cmd
    assert "HF_TOKEN=hf_abc" in cmd
    assert "HUGGING_FACE_HUB_TOKEN=hf_abc" in cmd


def test_no_hf_cache_means_no_mount_or_env():
    config = _baseline("vllm")
    config.backend.hf_cache = ""
    config.backend.hf_token = ""
    cmd = DockerServingBackend("vllm")._docker_run_command(config, "container")
    assert "HF_HOME=/root/.cache/huggingface" not in cmd
    assert "HF_TOKEN" not in " ".join(cmd)


def test_docker_args_pass_through_unchanged():
    config = _baseline("vllm")
    config.backend.docker_args = ["--gpus", "all", "--shm-size", "16g", "--ipc=host"]
    cmd = DockerServingBackend("vllm")._docker_run_command(config, "container")
    # docker_args should appear immediately before the image name, untouched.
    image_idx = cmd.index("vllm:test")
    docker_args = cmd[image_idx - len(config.backend.docker_args):image_idx]
    assert docker_args == config.backend.docker_args


def test_container_command_is_appended_verbatim():
    config = _baseline("vllm")
    config.backend.command = [
        "vllm", "serve", "/models/qwen",
        "--tensor-parallel-size", "2",
        "--gpu-memory-utilization", "0.9",
        "--max-model-len", "4096",
        "--host", "0.0.0.0", "--port", "8000",
    ]
    cmd = DockerServingBackend("vllm")._docker_run_command(config, "container")
    image_idx = cmd.index("vllm:test")
    assert cmd[image_idx + 1:] == config.backend.command


def test_unsupported_backend_raises():
    with pytest.raises(ValueError, match="Unsupported docker serving backend"):
        DockerServingBackend("transformers")


def test_preview_command_uses_predictable_name():
    config = _baseline("vllm")
    cmd = DockerServingBackend("vllm").preview_command(config)
    assert "llm-bench-vllm-preview" in cmd


def test_extra_mounts_emit_v_flags():
    from llm_bench.backends.docker_serving import DockerServingBackend
    config = _baseline("vllm")
    config.backend.extra_mounts = [
        "/home/u/.cache/modelscope:/home/u/.cache/modelscope",
        "/mnt/models:/models:ro",
    ]
    cmd = DockerServingBackend("vllm")._docker_run_command(config, "name")
    joined = " ".join(cmd)
    assert "-v /home/u/.cache/modelscope:/home/u/.cache/modelscope" in joined
    assert "-v /mnt/models:/models:ro" in joined


def test_extra_mounts_empty_entries_are_skipped():
    from llm_bench.backends.docker_serving import DockerServingBackend
    config = _baseline("vllm")
    config.backend.extra_mounts = ["", "/m:/m"]
    cmd = DockerServingBackend("vllm")._docker_run_command(config, "name")
    assert cmd.count("-v") == 1


def test_read_new_returns_only_new_bytes(tmp_path):
    from llm_bench.backends.docker_serving import _read_new
    log = tmp_path / "x.log"
    log.write_text("line1\n")
    text, off = _read_new(log, 0)
    assert text == "line1\n"
    text2, off2 = _read_new(log, off)
    assert text2 == ""
    log.write_text("line1\nline2\n")
    text3, _off3 = _read_new(log, off2)
    assert text3 == "line2\n"


def test_print_diagnostics_matches_cuda_runtime_hint(tmp_path, capsys):
    from llm_bench.backends.docker_serving import _print_diagnostics
    log = tmp_path / "log"
    log.write_text("W0607 21:09:19.634000 1 No CUDA runtime is found, using CUDA_HOME='/usr/local/cuda'")
    _print_diagnostics(log, 1)
    out = capsys.readouterr().out
    assert "container exited" in out
    assert "CUDA runtime" in out
    assert "--gpus all" in out


def test_print_diagnostics_silent_when_no_known_hint(tmp_path, capsys):
    from llm_bench.backends.docker_serving import _print_diagnostics
    log = tmp_path / "log"
    log.write_text("INFO server started")
    _print_diagnostics(log, 0)
    assert capsys.readouterr().out == ""


def test_print_wait_status_skips_when_no_new_log(tmp_path, capsys):
    from llm_bench.backends.docker_serving import _print_wait_status, _WaitState

    class FakeProc:
        def poll(self):
            return None

    log = tmp_path / "log"
    log.write_text("hello\n")
    state = _WaitState()
    _print_wait_status(10, log, FakeProc(), state)
    first = capsys.readouterr().out
    # New format prints "[setup] waiting for server | elapsed=10s | container=running"
    # followed by indented log content. Both should be present on the first tick.
    assert "[setup] waiting for server" in first
    assert "elapsed=10s" in first
    assert "hello" in first
    # No further writes -> the next tick should NOT re-print the same tail.
    _print_wait_status(20, log, FakeProc(), state)
    second = capsys.readouterr().out
    assert "hello" not in second
    assert "elapsed=20s" in second
