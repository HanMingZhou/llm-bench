from __future__ import annotations

import random

from llm_bench.backends.base import BackendResult, RequestMetric
from llm_bench.config import BenchConfig
from llm_bench.workload import build_workload_requests


class DryRunBackend:
    name = "dry-run"

    def run(self, config: BenchConfig) -> BackendResult:
        rng = random.Random(config.workload.seed)
        metrics: list[RequestMetric] = []
        request_index = 1
        workload_requests = build_workload_requests(config)

        for concurrency in config.workload.concurrency:
            count = max(config.workload.total_requests, 1)
            if config.workload.duration_seconds:
                count = max(concurrency * config.workload.duration_seconds, 1)
            index = 0
            while index < count:
                workload = workload_requests[index % len(workload_requests)]
                input_tokens = workload.input_tokens
                output_tokens = workload.output_tokens
                ttft = 25.0 + input_tokens * 0.018 + concurrency * 1.7 + rng.uniform(-3.0, 4.0)
                tpot = 6.0 + output_tokens * 0.006 + concurrency * 0.22 + rng.uniform(-0.7, 0.9)
                e2e = ttft + max(output_tokens - 1, 1) * tpot
                metrics.append(
                    RequestMetric(
                        request_id=f"req_{request_index:06d}",
                        backend=config.selected_backend,
                        concurrency=concurrency,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        requested_output_tokens=output_tokens,
                        ttft_ms=round(max(ttft, 1.0), 3),
                        tpot_ms=round(max(tpot, 0.1), 3),
                        e2e_latency_ms=round(max(e2e, 1.0), 3),
                        metadata=workload.metadata,
                        prompt_sample=workload.prompt[:500],
                        output_sample=f"dry-run response for {output_tokens} requested tokens",
                        output_valid=output_tokens > 0,
                        validation_error=None if output_tokens > 0 else "empty output",
                    )
                )
                request_index += 1
                index += 1

        return BackendResult(
            backend=config.selected_backend,
            request_metrics=metrics,
            startup_seconds=0.0,
            errors=[],
        )
