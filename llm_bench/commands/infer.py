from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path

from llm_bench.archive import create_run_dir, write_run_archive
from llm_bench.backends.base import BackendResult
from llm_bench.backends.registry import get_backend
from llm_bench.commands.common import existing_report_path
from llm_bench.config import (
    BenchConfig,
    apply_profile,
    default_backend_image,
    default_hf_cache,
    load_config,
    parse_int_list,
)
from llm_bench.environment import enforce_runtime_requirements, inspect_runtime
from llm_bench.interactive import run_infer_wizard, run_unified_wizard
from llm_bench.yaml_io import dump_yaml


def register_infer(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    infer = sub.add_parser(
        "infer",
        help="Run one inference benchmark.",
        description=(
            "Run one inference benchmark.\n"
            "\n"
            "vLLM (docker; if using the official vllm/vllm-openai image whose ENTRYPOINT\n"
            "is already `vllm serve`, pass ONLY the model path and args after `--`;\n"
            "for images without ENTRYPOINT, use the wizard `-i` or prepend `vllm serve`):\n"
            "  llm-bench infer --backend vllm --image vllm/vllm-openai:latest \\\n"
            "    --model-name Qwen/Qwen2.5-7B-Instruct --port 8000 -- \\\n"
            "    /models/qwen --tensor-parallel-size 2 \\\n"
            "      --gpu-memory-utilization 0.9 --max-model-len 4096 \\\n"
            "      --host 0.0.0.0 --port 8000\n"
            "\n"
            "SGLang (docker; the lmsysorg/sglang image has NO server entrypoint, so\n"
            "after `--` pass the full launcher command):\n"
            "  llm-bench infer --backend sglang --image lmsysorg/sglang:latest \\\n"
            "    --model-name Qwen/Qwen2.5-7B-Instruct --port 30000 -- \\\n"
            "    python3 -m sglang.launch_server --model-path /models/qwen \\\n"
            "      --host 0.0.0.0 --port 30000 --tp 2\n"
            "\n"
            "Transformers (in-process, no docker; uses HF kwargs directly, do NOT\n"
            "pass anything after `--`):\n"
            "  llm-bench infer --backend transformers \\\n"
            "    --model-path /models/qwen --torch-dtype bfloat16 \\\n"
            "    --device-map cuda:0 --trust-remote-code \\\n"
            "    --workload-profile quick"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    _add_common_infer_options(infer)
    infer.add_argument("-i", "--interactive", action="store_true")
    infer.set_defaults(func=cmd_infer)


def register_wizard(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    wizard = sub.add_parser("wizard", help="Open the interactive benchmark wizard.")
    wizard.add_argument("--config", type=Path)
    wizard.set_defaults(func=cmd_wizard)


def register_check(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    check = sub.add_parser(
        "check",
        help="Check docker / image / port / GPU / disk before running infer.",
    )
    check.add_argument("--config", type=Path)
    check.add_argument("--backend", choices=["vllm", "sglang", "dry-run"])
    check.add_argument("--image")
    check.add_argument("--port", type=int)
    check.add_argument("--output-dir")
    check.add_argument("--skip-env-check", action="store_true")
    check.set_defaults(func=cmd_check)


def register_report(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    report = sub.add_parser("report", help="Print the Markdown report path for a run.")
    report.add_argument("run_dir", type=Path)
    report.set_defaults(func=cmd_report)


def _add_common_infer_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, help="Load a YAML config file. CLI flags override file values.")

    # Backend selection
    parser.add_argument(
        "--backend",
        choices=["vllm", "sglang", "transformers", "dry-run"],
        help="vllm/sglang use docker + OpenAI HTTP; transformers calls from_pretrained directly.",
    )

    # vllm / sglang (docker serving) options
    parser.add_argument("--image", help="Docker image to run (vllm/sglang only).")
    parser.add_argument("--port", type=int, help="Host port (forwarded into the container). vllm/sglang only.")
    parser.add_argument(
        "--model-name",
        help="OpenAI API `model` field. Required for vllm/sglang.",
    )
    parser.add_argument("--hf-cache", help="HuggingFace cache dir to mount. Default: $HF_HOME or ~/.cache/huggingface.")
    parser.add_argument("--hf-token", help="HuggingFace token. Passed into the container as HF_TOKEN.")
    parser.add_argument(
        "--docker-arg",
        action="append",
        default=None,
        help="Extra docker run argument. Repeatable. Pass `--docker-arg=--shm-size=16g` for flags that start with `--`.",
    )
    parser.add_argument("--startup-timeout", type=int, help="Seconds to wait for the server. Default 900.")
    parser.add_argument("--keep-container", action="store_true", help="Do not --rm the container.")

    # transformers backend options. Names mirror transformers exactly.
    parser.add_argument("--model-path", help="Local path or HF repo id. For transformers backend only.")
    parser.add_argument("--tokenizer-path", help="Tokenizer path. Default = --model-path.")
    parser.add_argument(
        "--torch-dtype",
        choices=["float16", "bfloat16", "float32"],
        help="Maps to `torch_dtype` kwarg of from_pretrained.",
    )
    parser.add_argument("--device-map", help="Maps to `device_map` kwarg (e.g. auto, cuda:0, cpu).")
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--revision", help="Model revision (`revision` kwarg of from_pretrained).")
    parser.add_argument(
        "--quantization",
        help="`load_in_4bit` / `load_in_8bit`. Pass 4bit, int4, nf4, 8bit, or int8.",
    )
    parser.add_argument("--low-cpu-mem-usage", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--do-sample", action=argparse.BooleanOptionalAction, default=None, help="generate() do_sample.")
    parser.add_argument("--top-k", type=int, help="generate() top_k.")
    parser.add_argument("--repetition-penalty", type=float, help="generate() repetition_penalty.")
    parser.add_argument("--num-beams", type=int, help="generate() num_beams.")
    parser.add_argument("--batch-size", type=int, help="transformers per-call batch size.")

    # Workload (client-side) options.
    parser.add_argument(
        "--workload-profile",
        choices=["quick", "standard", "long-context", "custom"],
        help="Preset workload. `custom` uses your own concurrency/input/output values.",
    )
    parser.add_argument("--api", choices=["completions", "chat"], help="OpenAI API endpoint.")
    parser.add_argument("--concurrency", help="Comma-separated, e.g. 1,4,8.")
    parser.add_argument("--input-tokens", help="Comma-separated synthetic input token counts.")
    parser.add_argument("--output-tokens", help="Comma-separated max output tokens.")
    parser.add_argument("--total-requests", type=int)
    parser.add_argument("--duration", type=int, help="Benchmark by wall-clock seconds instead of a fixed request count.")
    parser.add_argument("--warmup-requests", type=int)
    parser.add_argument("--request-timeout", type=int, help="Seconds per request. Default 120.")
    parser.add_argument("--stream", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--prompt-jsonl", help="Use prompts from a JSONL file.")
    parser.add_argument("--prompt-dir", help="Use prompts from a directory of text files.")
    parser.add_argument("--prompt-include")
    parser.add_argument("--prompt-exclude")
    parser.add_argument("--prompt-dir-recursive", action=argparse.BooleanOptionalAction, default=None)

    # Report options.
    parser.add_argument("--output-dir", help="Where to write the run archive.")
    parser.add_argument("--run-name")
    parser.add_argument("--tag", action="append", default=None)
    parser.add_argument("--save-request-metrics", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--save-gpu-metrics", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--save-logs", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--include-samples", action=argparse.BooleanOptionalAction, default=None)

    parser.add_argument("--skip-env-check", action="store_true")


def cmd_infer(args: argparse.Namespace) -> None:
    config = load_config(getattr(args, "config", None))
    if getattr(args, "interactive", False):
        config, requested, should_start = run_infer_wizard(config)
        config, requested = _apply_cli_overrides(config, args, requested)
        if not should_start:
            _save_interactive_request(config, requested)
            return
    else:
        config, requested = _apply_cli_overrides(config, args, {})
    _validate_for_run(config)
    _run_inference(config, requested)


def cmd_wizard(args: argparse.Namespace) -> None:
    config = load_config(getattr(args, "config", None))
    task_type, result, requested, should_start = run_unified_wizard(config)
    if task_type == "infer":
        if not should_start:
            _save_interactive_request(result, requested)
            return
        _validate_for_run(result)
        _run_inference(result, requested)
    else:
        from llm_bench.commands.comm import cmd_comm_all_reduce
        cmd_comm_all_reduce(result)


def cmd_check(args: argparse.Namespace) -> None:
    config = load_config(getattr(args, "config", None))
    config, _ = _apply_cli_overrides(config, args, {})
    runtime = inspect_runtime(config)
    print(json.dumps(runtime, indent=2, ensure_ascii=False))
    enforce_runtime_requirements(config, runtime)


def cmd_report(args: argparse.Namespace) -> None:
    print(existing_report_path(args.run_dir))


def _run_inference(config: BenchConfig, requested: dict[str, object]) -> None:
    runtime = inspect_runtime(config)
    try:
        enforce_runtime_requirements(config, runtime)
    except RuntimeError as exc:
        run_dir = create_run_dir(config)
        result = BackendResult(
            backend=config.selected_backend,
            request_metrics=[],
            startup_seconds=0.0,
            errors=[str(exc)],
            gpu_metrics=[],
        )
        manifest = write_run_archive(run_dir, config, requested, result, runtime)
        print(f"run_id: {manifest['run_id']}")
        print(f"run_dir: {run_dir}")
        print(f"report: {run_dir / 'reports' / 'inference_report.md'}")
        raise
    backend = get_backend(config)
    _print_execution_plan(config, runtime, backend)
    run_dir = create_run_dir(config)
    result = backend.run(config)
    manifest = write_run_archive(run_dir, config, requested, result, runtime)
    print(f"run_id: {manifest['run_id']}")
    print(f"run_dir: {run_dir}")
    print(f"report: {run_dir / 'reports' / 'inference_report.md'}")


def _validate_for_run(config: BenchConfig) -> None:
    if config.backend.name in {"vllm", "sglang"}:
        if not config.backend.command:
            raise ValueError(
                "Missing container command. Append it after `--` on the CLI. "
                "For the official vllm/vllm-openai image (ENTRYPOINT is `vllm serve`), "
                "pass only the model path and args; for other images, prepend `vllm serve`. "
                "Use `-i` (interactive wizard) to select the launcher. Example:\n"
                "  llm-bench infer --backend vllm --image vllm/vllm-openai:latest \\\n"
                "    --model-name Qwen/Qwen2.5-7B-Instruct -- \\\n"
                "    /models/qwen --host 0.0.0.0 --port 8000 ..."
            )
        if not config.backend.model_name:
            raise ValueError("--model-name is required (used as the OpenAI API `model` field).")
        if not config.backend.image:
            raise ValueError("--image is required (or set backend.image in your YAML).")
    elif config.backend.name == "transformers":
        if not config.transformers.model_path:
            raise ValueError(
                "--model-path is required for transformers backend.\n"
                "Example:\n"
                "  llm-bench infer --backend transformers \\\n"
                "    --model-path /mnt/models/qwen --torch-dtype bfloat16 \\\n"
                "    --device-map cuda:0 --workload-profile quick"
            )


def _save_interactive_request(config: BenchConfig, requested: dict[str, object]) -> None:
    output = Path(config.report.output_dir).parent / "interactive.requested.yaml"
    output.parent.mkdir(parents=True, exist_ok=True)
    dump_yaml(output, requested)
    print(f"saved_config: {output}")


def _apply_cli_overrides(
    config: BenchConfig,
    args: argparse.Namespace,
    requested: dict[str, object],
) -> tuple[BenchConfig, dict[str, object]]:
    profile = getattr(args, "workload_profile", None)
    if profile:
        apply_profile(config, profile)
        requested.setdefault("workload", {})["profile"] = profile

    def set_attr(section: str, attr: str, value: object) -> None:
        if value is None:
            return
        getattr(config, section).__setattr__(attr, value)
        requested.setdefault(section, {})[attr] = value

    set_attr("backend", "name", getattr(args, "backend", None))
    set_attr("backend", "image", getattr(args, "image", None))
    set_attr("backend", "port", getattr(args, "port", None))
    set_attr("backend", "model_name", getattr(args, "model_name", None))
    set_attr("backend", "hf_cache", getattr(args, "hf_cache", None))
    set_attr("backend", "hf_token", getattr(args, "hf_token", None))
    set_attr("backend", "startup_timeout_seconds", getattr(args, "startup_timeout", None))

    if getattr(args, "keep_container", False):
        config.backend.keep_container = True
        requested.setdefault("backend", {})["keep_container"] = True

    docker_arg = getattr(args, "docker_arg", None)
    if docker_arg:
        config.backend.docker_args = list(docker_arg)
        requested.setdefault("backend", {})["docker_args"] = list(docker_arg)

    passthrough = getattr(args, "passthrough", None) or []
    if passthrough:
        config.backend.command = list(passthrough)
        requested.setdefault("backend", {})["command"] = list(passthrough)

    # transformers-only options
    set_attr("transformers", "model_path", getattr(args, "model_path", None))
    set_attr("transformers", "tokenizer_path", getattr(args, "tokenizer_path", None))
    set_attr("transformers", "torch_dtype", getattr(args, "torch_dtype", None))
    set_attr("transformers", "device_map", getattr(args, "device_map", None))
    set_attr("transformers", "revision", getattr(args, "revision", None))
    set_attr("transformers", "quantization", getattr(args, "quantization", None))
    set_attr("transformers", "top_k", getattr(args, "top_k", None))
    set_attr("transformers", "repetition_penalty", getattr(args, "repetition_penalty", None))
    set_attr("transformers", "num_beams", getattr(args, "num_beams", None))
    set_attr("transformers", "batch_size", getattr(args, "batch_size", None))
    if getattr(args, "trust_remote_code", None) is not None:
        config.transformers.trust_remote_code = bool(args.trust_remote_code)
        requested.setdefault("transformers", {})["trust_remote_code"] = bool(args.trust_remote_code)
    if getattr(args, "low_cpu_mem_usage", None) is not None:
        config.transformers.low_cpu_mem_usage = bool(args.low_cpu_mem_usage)
        requested.setdefault("transformers", {})["low_cpu_mem_usage"] = bool(args.low_cpu_mem_usage)
    if getattr(args, "do_sample", None) is not None:
        config.transformers.do_sample = bool(args.do_sample)
        requested.setdefault("transformers", {})["do_sample"] = bool(args.do_sample)

    set_attr("workload", "api", getattr(args, "api", None))
    set_attr("workload", "total_requests", getattr(args, "total_requests", None))
    set_attr("workload", "duration_seconds", getattr(args, "duration", None))
    set_attr("workload", "warmup_requests", getattr(args, "warmup_requests", None))
    set_attr("workload", "request_timeout_seconds", getattr(args, "request_timeout", None))
    set_attr("workload", "temperature", getattr(args, "temperature", None))
    set_attr("workload", "top_p", getattr(args, "top_p", None))
    set_attr("workload", "seed", getattr(args, "seed", None))
    set_attr("workload", "prompt_jsonl", getattr(args, "prompt_jsonl", None))
    set_attr("workload", "prompt_dir", getattr(args, "prompt_dir", None))
    set_attr("workload", "prompt_include", getattr(args, "prompt_include", None))
    set_attr("workload", "prompt_exclude", getattr(args, "prompt_exclude", None))

    if getattr(args, "stream", None) is not None:
        config.workload.stream = bool(args.stream)
        requested.setdefault("workload", {})["stream"] = bool(args.stream)
    if getattr(args, "prompt_dir_recursive", None) is not None:
        config.workload.prompt_dir_recursive = bool(args.prompt_dir_recursive)
        requested.setdefault("workload", {})["prompt_dir_recursive"] = bool(args.prompt_dir_recursive)

    for name in ("input_tokens", "output_tokens", "concurrency"):
        parsed = parse_int_list(getattr(args, name, None))
        if parsed is not None:
            setattr(config.workload, name, parsed)
            requested.setdefault("workload", {})[name] = parsed

    set_attr("report", "output_dir", getattr(args, "output_dir", None))
    set_attr("report", "run_name", getattr(args, "run_name", None))
    if getattr(args, "tag", None):
        config.report.tags.extend(args.tag)
        requested.setdefault("report", {})["tags"] = list(config.report.tags)
    for attr in ("save_request_metrics", "save_gpu_metrics", "save_logs", "include_samples"):
        value = getattr(args, attr, None)
        if value is not None:
            setattr(config.report, attr, bool(value))
            requested.setdefault("report", {})[attr] = bool(value)

    if getattr(args, "skip_env_check", False):
        config.skip_env_check = True
        requested["skip_env_check"] = True

    if config.workload.prompt_jsonl:
        config.workload.mode = "jsonl"
        requested.setdefault("workload", {})["mode"] = "jsonl"
    elif config.workload.prompt_dir:
        config.workload.mode = "prompt-dir"
        requested.setdefault("workload", {})["mode"] = "prompt-dir"

    if not config.backend.image:
        config.backend.image = default_backend_image(config.backend.name)
    if not config.backend.hf_cache:
        config.backend.hf_cache = default_hf_cache()

    return config, requested


def _print_execution_plan(config: BenchConfig, runtime: dict[str, object], backend) -> None:
    print("", flush=True)
    print("Execution plan", flush=True)
    print(f"- backend: {config.backend.name}", flush=True)
    if config.backend.name in {"vllm", "sglang"}:
        print(f"- image: {config.backend.image}", flush=True)
        print(f"- model_name (API): {config.backend.model_name}", flush=True)
        print(f"- port: {config.backend.port}", flush=True)
    elif config.backend.name == "transformers":
        print(f"- model_path: {config.transformers.model_path}", flush=True)
        print(f"- torch_dtype: {config.transformers.torch_dtype}", flush=True)
        print(f"- device_map: {config.transformers.device_map}", flush=True)
        print(f"- quantization: {config.transformers.quantization or '(none)'}", flush=True)
        print(f"- batch_size: {config.transformers.batch_size}", flush=True)
    gpu = runtime.get("gpu") or {}
    gpus = gpu.get("gpus") or []
    if gpus:
        print(f"- detected GPUs: {gpu.get('gpu_count')}", flush=True)
        for item in gpus:
            print(f"  [{item.get('index')}] {item.get('name')} ({item.get('memory_total_mb')} MiB)", flush=True)
    else:
        print(f"- detected GPUs: 0 ({gpu.get('error') or 'not available'})", flush=True)
    if hasattr(backend, "preview_command"):
        command = backend.preview_command(config)
        print("- docker command:", flush=True)
        print(f"  {shlex.join(command)}", flush=True)
    print("", flush=True)
