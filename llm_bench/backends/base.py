from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any
from typing import Protocol

from llm_bench.config import BenchConfig


@dataclass
class RequestMetric:
    request_id: str
    backend: str
    concurrency: int
    input_tokens: int
    output_tokens: int
    requested_output_tokens: int
    ttft_ms: float
    tpot_ms: float
    e2e_latency_ms: float
    success: bool = True
    error: str | None = None
    error_category: str | None = None
    metadata: dict[str, Any] | None = None
    prompt_sample: str | None = None
    output_sample: str | None = None
    output_valid: bool | None = None
    validation_error: str | None = None
    # Absolute wall-clock timestamps (set by backends that issue real requests,
    # e.g. the HTTP client) so throughput can use true elapsed time.
    start_unix: float | None = None
    end_unix: float | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class BackendResult:
    backend: str
    request_metrics: list[RequestMetric]
    startup_seconds: float = 0.0
    errors: list[str] | None = None
    gpu_metrics: list[dict[str, object]] | None = None
    peak_memory_mb: float | None = None


class InferenceBackend(Protocol):
    name: str

    def run(self, config: BenchConfig) -> BackendResult:
        ...
