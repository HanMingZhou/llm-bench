from __future__ import annotations

from llm_bench.backends.docker_serving import DockerServingBackend
from llm_bench.backends.dry_run import DryRunBackend
from llm_bench.backends.transformers_backend import TransformersBackend
from llm_bench.config import BenchConfig


def get_backend(config: BenchConfig):
    name = config.backend.name
    if name == "dry-run":
        return DryRunBackend()
    if name in {"vllm", "sglang"}:
        return DockerServingBackend(name)
    if name == "transformers":
        return TransformersBackend()
    raise ValueError(f"Unknown backend: {name}. Supported: vllm, sglang, transformers, dry-run.")
