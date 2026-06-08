import argparse

from llm_bench.config import BenchConfig, apply_profile, default_backend_image, from_mapping
from llm_bench.commands.infer import _apply_cli_overrides


def _ns(**kwargs) -> argparse.Namespace:
    """Build an argparse.Namespace defaulting every known infer flag to None."""
    defaults = dict(
        config=None, backend=None, image=None, port=None, model_name=None,
        hf_cache=None, hf_token=None, docker_arg=None, startup_timeout=None,
        keep_container=False, model_path=None, tokenizer_path=None,
        torch_dtype=None, device_map=None, trust_remote_code=None, revision=None,
        quantization=None, low_cpu_mem_usage=None, do_sample=None, top_k=None,
        repetition_penalty=None, num_beams=None, batch_size=None,
        workload_profile=None, api=None, concurrency=None, input_tokens=None,
        output_tokens=None, total_requests=None, warmup_requests=None,
        request_timeout=None, stream=None, temperature=None, top_p=None, seed=None,
        prompt_jsonl=None, prompt_dir=None, prompt_include=None, prompt_exclude=None,
        prompt_dir_recursive=None, output_dir=None, run_name=None, tag=None,
        save_request_metrics=None, save_gpu_metrics=None, save_logs=None,
        include_samples=None, skip_env_check=False, passthrough=[],
    )
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def test_passthrough_becomes_backend_command():
    args = _ns(
        backend="vllm",
        image="vllm:test",
        model_name="Qwen/Qwen2.5-7B-Instruct",
        passthrough=["vllm", "serve", "/m", "--tensor-parallel-size", "2"],
    )
    config, requested = _apply_cli_overrides(BenchConfig(), args, {})
    assert config.backend.command == ["vllm", "serve", "/m", "--tensor-parallel-size", "2"]
    assert requested["backend"]["command"] == config.backend.command
    assert config.backend.model_name == "Qwen/Qwen2.5-7B-Instruct"


def test_docker_args_override_replaces_list():
    args = _ns(backend="vllm", docker_arg=["--gpus", "all", "--shm-size=16g"])
    config, _ = _apply_cli_overrides(BenchConfig(), args, {})
    assert config.backend.docker_args == ["--gpus", "all", "--shm-size=16g"]


def test_int_list_parsing_for_workload():
    args = _ns(backend="vllm", concurrency="1,4,8", input_tokens="512,2048", output_tokens="128")
    config, _ = _apply_cli_overrides(BenchConfig(), args, {})
    assert config.workload.concurrency == [1, 4, 8]
    assert config.workload.input_tokens == [512, 2048]
    assert config.workload.output_tokens == [128]


def test_profile_then_cli_override_precedence():
    # profile sets concurrency, explicit --concurrency must win.
    args = _ns(backend="vllm", workload_profile="standard", concurrency="2")
    config, _ = _apply_cli_overrides(BenchConfig(), args, {})
    assert config.workload.profile == "standard"
    assert config.workload.concurrency == [2]


def test_prompt_jsonl_sets_mode():
    args = _ns(backend="vllm", prompt_jsonl="examples/workload.jsonl")
    config, _ = _apply_cli_overrides(BenchConfig(), args, {})
    assert config.workload.mode == "jsonl"


def test_prompt_dir_sets_mode():
    args = _ns(backend="vllm", prompt_dir="examples/prompts")
    config, _ = _apply_cli_overrides(BenchConfig(), args, {})
    assert config.workload.mode == "prompt-dir"


def test_transformers_options_routed_to_transformers_section():
    args = _ns(
        backend="transformers",
        model_path="/mnt/models/qwen",
        torch_dtype="bfloat16",
        device_map="cuda:0",
        trust_remote_code=True,
        quantization="4bit",
        do_sample=True,
        batch_size=2,
    )
    config, requested = _apply_cli_overrides(BenchConfig(), args, {})
    assert config.transformers.model_path == "/mnt/models/qwen"
    assert config.transformers.torch_dtype == "bfloat16"
    assert config.transformers.device_map == "cuda:0"
    assert config.transformers.trust_remote_code is True
    assert config.transformers.quantization == "4bit"
    assert config.transformers.do_sample is True
    assert config.transformers.batch_size == 2
    assert requested["transformers"]["model_path"] == "/mnt/models/qwen"


def test_default_image_filled_for_vllm():
    args = _ns(backend="vllm")
    config, _ = _apply_cli_overrides(BenchConfig(), args, {})
    assert config.backend.image == default_backend_image("vllm")


def test_default_image_not_filled_for_transformers():
    args = _ns(backend="transformers", model_path="/m")
    config, _ = _apply_cli_overrides(BenchConfig(), args, {})
    assert config.backend.image == ""


def test_keep_container_and_skip_env_check_flags():
    args = _ns(backend="vllm", keep_container=True, skip_env_check=True)
    config, requested = _apply_cli_overrides(BenchConfig(), args, {})
    assert config.backend.keep_container is True
    assert config.skip_env_check is True
    assert requested["skip_env_check"] is True


def test_config_roundtrip_via_to_dict_and_from_mapping():
    args = _ns(
        backend="vllm",
        image="vllm:test",
        model_name="Qwen/Qwen2.5-7B-Instruct",
        port=8001,
        docker_arg=["--gpus", "all"],
        passthrough=["vllm", "serve", "/m"],
    )
    config, _ = _apply_cli_overrides(BenchConfig(), args, {})
    rebuilt = from_mapping(config.to_dict())
    assert rebuilt.backend.name == "vllm"
    assert rebuilt.backend.image == "vllm:test"
    assert rebuilt.backend.model_name == "Qwen/Qwen2.5-7B-Instruct"
    assert rebuilt.backend.port == 8001
    assert rebuilt.backend.command == ["vllm", "serve", "/m"]
    assert rebuilt.backend.docker_args == ["--gpus", "all"]


def test_to_dict_strips_runtime_fields():
    config = BenchConfig()
    config.backend.stdout_log = "/tmp/x"
    config.backend.launch_command = ["docker", "run"]
    data = config.to_dict()
    assert "stdout_log" not in data["backend"]
    assert "launch_command" not in data["backend"]


def test_apply_profile_custom_is_noop_marker():
    config = BenchConfig()
    apply_profile(config, "custom")
    assert config.workload.profile == "custom"


def test_stream_defaults_to_true():
    # Regression guard: serving must default to streaming for accurate TTFT/TPOT.
    assert BenchConfig().workload.stream is True
