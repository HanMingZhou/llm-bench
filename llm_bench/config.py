from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .yaml_io import load_yaml


@dataclass
class BackendConfig:
    """Thin wrapper around a docker run + a real serving framework command.

    The tool no longer translates framework parameters. Everything inside the
    container is whatever the user wrote in `command` (the argv after `--` on
    the CLI). The fields here describe what the *tool* itself needs: which
    image, which port to forward, which OpenAI model id to put in the request
    body, plus optional HF cache passthrough and arbitrary docker args.
    """

    name: str = "vllm"  # vllm | sglang | dry-run
    image: str = ""
    port: int = 8000
    model_name: str = ""
    hf_cache: str = ""
    hf_token: str = ""
    docker_args: list[str] = field(default_factory=list)
    # Extra bind mounts the wizard auto-adds (e.g. ModelScope cache, local
    # /mnt/models). Each entry is the literal `host:container` (or
    # `host:container:ro`) string passed to `docker run -v`.
    extra_mounts: list[str] = field(default_factory=list)
    command: list[str] = field(default_factory=list)
    startup_timeout_seconds: int = 900
    keep_container: bool = False

    # Runtime-populated, not serialized as user config.
    stdout_log: str = ""
    stderr_log: str = ""
    launch_command: list[str] = field(default_factory=list)


@dataclass
class TransformersConfig:
    """Native HuggingFace transformers backend parameters.

    Field names mirror `AutoModelForCausalLM.from_pretrained` and
    `model.generate(...)` keyword arguments. No translation happens; e.g.
    `torch_dtype` here is passed straight as `torch_dtype=` to
    `from_pretrained`.
    """

    model_path: str = ""
    tokenizer_path: str = ""
    torch_dtype: str = "bfloat16"
    device_map: str = "auto"
    trust_remote_code: bool = False
    revision: str = "main"
    quantization: str = ""
    low_cpu_mem_usage: bool = True
    do_sample: bool = False
    top_k: int = 50
    repetition_penalty: float = 1.0
    num_beams: int = 1
    batch_size: int = 1


@dataclass
class WorkloadConfig:
    profile: str = "custom"
    mode: str = "fixed"
    api: str = "completions"
    input_tokens: list[int] = field(default_factory=lambda: [512])
    output_tokens: list[int] = field(default_factory=lambda: [128])
    concurrency: list[int] = field(default_factory=lambda: [1])
    total_requests: int = 16
    duration_seconds: int | None = None
    warmup_requests: int = 2
    request_timeout_seconds: int = 120
    # Default to streaming so TTFT / TPOT are measured accurately for vLLM /
    # SGLang. Non-streaming cannot separate the first token from the rest.
    stream: bool = True
    seed: int = 42
    prompt_jsonl: str = ""
    prompt_dir: str = ""
    prompt_dir_recursive: bool = True
    prompt_include: str = "*.txt,*.md,*.json,*.jsonl"
    prompt_exclude: str = ""
    temperature: float = 0.0
    top_p: float = 1.0


@dataclass
class ReportConfig:
    output_dir: str = "benchmark_output/runs"
    run_name: str = ""
    tags: list[str] = field(default_factory=list)
    include_samples: bool = False
    save_request_metrics: bool = True
    save_gpu_metrics: bool = True
    save_logs: bool = True


@dataclass
class RetentionConfig:
    request_metrics_days: int | None = None
    gpu_metrics_days: int | None = None
    logs_days: int | None = None
    keep_summary_forever: bool = True
    keep_output_samples: bool = True


@dataclass
class BenchConfig:
    backend: BackendConfig = field(default_factory=BackendConfig)
    transformers: TransformersConfig = field(default_factory=TransformersConfig)
    workload: WorkloadConfig = field(default_factory=WorkloadConfig)
    report: ReportConfig = field(default_factory=ReportConfig)
    retention: RetentionConfig = field(default_factory=RetentionConfig)
    skip_env_check: bool = False

    @property
    def selected_backend(self) -> str:
        return self.backend.name

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        # Strip runtime-only fields from the persisted view.
        backend = data.get("backend") or {}
        for key in ("stdout_log", "stderr_log", "launch_command"):
            backend.pop(key, None)
        return data


PROFILES: dict[str, dict[str, Any]] = {
    "quick": {
        "input_tokens": [512],
        "output_tokens": [128],
        "concurrency": [1, 4],
        "total_requests": 32,
        "warmup_requests": 4,
        "request_timeout_seconds": 120,
        "mode": "fixed",
    },
    "standard": {
        "input_tokens": [512, 2048],
        "output_tokens": [128, 256],
        "concurrency": [1, 4, 8, 16],
        "total_requests": 256,
        "warmup_requests": 16,
        "request_timeout_seconds": 180,
        "mode": "fixed",
    },
    "long-context": {
        "input_tokens": [4096, 8192, 16384],
        "output_tokens": [128],
        "concurrency": [1, 2, 4],
        "total_requests": 128,
        "warmup_requests": 8,
        "request_timeout_seconds": 300,
        "mode": "long-context",
    },
}


def parse_int_list(value: str | None) -> list[int] | None:
    if value is None:
        return None
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if not parts:
        return []
    return [int(p) for p in parts]


def default_hf_cache() -> str:
    """Return the default Hugging Face cache directory on this host."""
    for candidate in (
        os.environ.get("HF_HOME"),
        os.environ.get("HUGGINGFACE_HUB_CACHE"),
        os.environ.get("TRANSFORMERS_CACHE"),
    ):
        if candidate:
            return str(Path(candidate).expanduser())
    return str(Path("~/.cache/huggingface").expanduser())


def default_backend_image(backend: str) -> str:
    images = {
        "vllm": "vllm/vllm-openai:latest",
        "sglang": "lmsysorg/sglang:latest",
        "dry-run": "",
    }
    return images.get(backend, "")


def from_mapping(data: dict[str, Any]) -> BenchConfig:
    backend_data = dict(data.get("backend") or {})
    backend = BackendConfig(**_filter_keys(backend_data, BackendConfig))

    transformers_data = dict(data.get("transformers") or {})
    transformers = TransformersConfig(**_filter_keys(transformers_data, TransformersConfig))

    workload_data = _filter_keys(data.get("workload") or {}, WorkloadConfig)
    for key in ("input_tokens", "output_tokens", "concurrency"):
        if key in workload_data and isinstance(workload_data[key], int):
            workload_data[key] = [workload_data[key]]
    workload = WorkloadConfig(**workload_data)

    report = ReportConfig(**_filter_keys(data.get("report") or {}, ReportConfig))
    retention = RetentionConfig(**_filter_keys(data.get("retention") or {}, RetentionConfig))

    if not backend.image and backend.name in {"vllm", "sglang"}:
        backend.image = default_backend_image(backend.name)
    if not backend.hf_cache:
        backend.hf_cache = default_hf_cache()

    return BenchConfig(
        backend=backend,
        transformers=transformers,
        workload=workload,
        report=report,
        retention=retention,
        skip_env_check=bool(data.get("skip_env_check", False)),
    )


def load_config(path: Path | None) -> BenchConfig:
    data = load_yaml(path) if path else {}
    return from_mapping(data)


def apply_profile(config: BenchConfig, profile: str) -> None:
    if profile == "custom":
        config.workload.profile = "custom"
        return
    if profile not in PROFILES:
        raise ValueError(f"Unknown profile: {profile}")
    config.workload.profile = profile
    for key, value in PROFILES[profile].items():
        setattr(config.workload, key, value)


def _filter_keys(data: dict[str, Any], cls: Any) -> dict[str, Any]:
    allowed = set(cls.__dataclass_fields__.keys())
    return {k: v for k, v in data.items() if k in allowed}
