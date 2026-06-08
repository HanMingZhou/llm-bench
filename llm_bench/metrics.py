from __future__ import annotations

from collections import Counter, defaultdict
from statistics import mean
from collections.abc import Iterable

from llm_bench.backends.base import RequestMetric


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * pct
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _wall_clock_seconds(rows: list[RequestMetric], fallback: float) -> float:
    """Real wall-clock span (max end - min start) for a group of requests.

    Uses absolute timestamps when the backend records them (HTTP client). Falls
    back to the sum(e2e)/concurrency estimate for backends that do not record
    them (dry-run, transformers batch mode).
    """
    starts = [m.start_unix for m in rows if m.start_unix is not None]
    ends = [m.end_unix for m in rows if m.end_unix is not None]
    if starts and ends:
        span = max(ends) - min(starts)
        if span > 0:
            return span
    return fallback


def _safe_div(numerator: float, denominator: float) -> float:
    """Zero-safe division. Returns 0.0 when denominator is zero/None."""
    if not denominator:
        return 0.0
    return numerator / denominator


def _decode_tps_from_tpot_ms(tpot_ms: float) -> float:
    """Per-request decode speed (tokens/sec) derived from TPOT.

    Decode TPS = 1000 / TPOT(ms). With streaming each output token's average
    cost (TPOT) is real; non-streaming amortizes E2E across all output tokens,
    which is still a meaningful per-request "tokens per second" reading.

    Important: when we pass `tpot_p99` here, the result is NOT the "99th
    percentile of decode TPS"; it is the decode TPS *corresponding to* the
    99th percentile (slowest) TPOT, i.e. the lower-bound user experience
    (`worst-case decode speed` rather than "fastest 99% of requests"). Field
    naming kept (`decode_tps_p99`) for backwards compatibility — interpret
    it as "decode TPS at TPOT p99".
    """
    return round(1000.0 / tpot_ms, 3) if tpot_ms > 0 else 0.0


def _prefill_metrics_per_req(rows: list[RequestMetric]) -> tuple[float, float, float]:
    """Per-request prefill throughput aggregates: (mean, p50, p99) tok/s.

    Prefill TPS per request = input_tokens / TTFT_seconds. Streamed runs give
    a true TTFT; non-streamed runs collapse TTFT to E2E so this number becomes
    "tokens spent during the entire request", which is still useful as a
    loose proxy for long-context behaviour.
    """
    if not rows:
        return 0.0, 0.0, 0.0
    per_req = []
    for m in rows:
        if m.ttft_ms <= 0 or m.input_tokens <= 0:
            continue
        per_req.append(m.input_tokens / (m.ttft_ms / 1000.0))
    if not per_req:
        return 0.0, 0.0, 0.0
    return (
        round(mean(per_req), 3),
        round(percentile(per_req, 0.50), 3),
        round(percentile(per_req, 0.99), 3),
    )


def summarize_requests(
    metrics: Iterable[RequestMetric],
    gpu_metrics: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    rows = list(metrics)
    success = [m for m in rows if m.success]
    failed = [m for m in rows if not m.success]
    grouped: dict[tuple[int, int, int], list[RequestMetric]] = defaultdict(list)
    for row in rows:
        output_bucket = row.requested_output_tokens or row.output_tokens
        grouped[(row.input_tokens, output_bucket, row.concurrency)].append(row)

    backend_results = []
    for (input_tokens, output_tokens, concurrency), group in sorted(grouped.items()):
        ok = [m for m in group if m.success]
        e2e = [m.e2e_latency_ms for m in ok]
        ttft = [m.ttft_ms for m in ok]
        tpot = [m.tpot_ms for m in ok]
        total_wall_seconds = _wall_clock_seconds(ok, sum(e2e) / 1000.0 / max(concurrency, 1))
        output_token_count = sum(m.output_tokens for m in ok)
        input_token_count = sum(m.input_tokens for m in ok)
        tpot_p50 = round(percentile(tpot, 0.50), 3)
        tpot_p99 = round(percentile(tpot, 0.99), 3)
        prefill_mean, prefill_p50, prefill_p99 = _prefill_metrics_per_req(ok)
        backend_results.append(
            {
                "workload": {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "concurrency": concurrency,
                },
                "metrics": {
                    # System throughput (across all concurrent requests).
                    "qps": round(_safe_div(len(ok), total_wall_seconds), 3),
                    "input_tokens_per_sec": round(_safe_div(input_token_count, total_wall_seconds), 3),
                    "output_tokens_per_sec": round(_safe_div(output_token_count, total_wall_seconds), 3),
                    "total_tokens_per_sec": round(
                        _safe_div(input_token_count + output_token_count, total_wall_seconds), 3
                    ),
                    # Per-request token speeds the user actually feels.
                    "decode_tps_p50": _decode_tps_from_tpot_ms(tpot_p50),
                    "decode_tps_p99": _decode_tps_from_tpot_ms(tpot_p99),
                    "prefill_tps_per_req_mean": prefill_mean,
                    "prefill_tps_per_req_p99": prefill_p99,
                    # Latencies.
                    "ttft_p50_ms": round(percentile(ttft, 0.50), 3),
                    "ttft_p90_ms": round(percentile(ttft, 0.90), 3),
                    "ttft_p99_ms": round(percentile(ttft, 0.99), 3),
                    "tpot_p50_ms": tpot_p50,
                    "tpot_p90_ms": round(percentile(tpot, 0.90), 3),
                    "tpot_p99_ms": tpot_p99,
                    "e2e_p50_ms": round(percentile(e2e, 0.50), 3),
                    "e2e_p90_ms": round(percentile(e2e, 0.90), 3),
                    "e2e_p99_ms": round(percentile(e2e, 0.99), 3),
                    "success_requests": len(ok),
                    "failed_requests": len(group) - len(ok),
                },
            }
        )

    e2e_all = [m.e2e_latency_ms for m in success]
    error_categories = Counter(m.error_category or "unknown" for m in failed)

    # Error-category counts.
    timeout_requests = sum(1 for m in rows if m.error_category == "timeout")
    oom_count = sum(1 for m in rows if m.error_category == "oom")
    service_crash_categories = {"container_exit", "health_check_timeout", "docker_daemon"}
    service_crash_count = sum(1 for m in rows if m.error_category in service_crash_categories)

    # Global aggregates over all successful requests.
    all_e2e = [m.e2e_latency_ms for m in success]
    all_ttft = [m.ttft_ms for m in success]
    all_tpot = [m.tpot_ms for m in success]
    all_output_tokens = sum(m.output_tokens for m in success)
    all_input_tokens = sum(m.input_tokens for m in success)
    all_concurrencies = [m.concurrency for m in success]
    avg_concurrency = mean(all_concurrencies) if all_concurrencies else 1
    total_wall_seconds = _wall_clock_seconds(success, sum(all_e2e) / 1000.0 / max(avg_concurrency, 1))

    global_qps = round(_safe_div(len(success), total_wall_seconds), 3)
    global_input_tps = round(_safe_div(all_input_tokens, total_wall_seconds), 3)
    global_output_tps = round(_safe_div(all_output_tokens, total_wall_seconds), 3)
    global_total_tps = round(_safe_div(all_input_tokens + all_output_tokens, total_wall_seconds), 3)
    # Per-request "speed the user feels": decode TPS from median TPOT, plus
    # mean / p99 prefill TPS per request.
    global_tpot_p50 = round(percentile(all_tpot, 0.50), 3)
    global_tpot_p99 = round(percentile(all_tpot, 0.99), 3)
    decode_tps_p50 = _decode_tps_from_tpot_ms(global_tpot_p50)
    decode_tps_p99 = _decode_tps_from_tpot_ms(global_tpot_p99)
    prefill_mean, prefill_p50, prefill_p99 = _prefill_metrics_per_req(success)

    # GPU aggregates.
    gpu_avg_utilization: float = 0.0
    gpu_max_utilization: float = 0.0
    gpu_avg_memory_used_mb: float = 0.0
    gpu_peak_memory_used_mb: float = 0.0
    gpu_avg_temperature: float = 0.0
    gpu_max_temperature: float = 0.0
    gpu_avg_power_w: float = 0.0
    gpu_max_power_w: float = 0.0

    if gpu_metrics:
        utilizations = [g["utilization"] for g in gpu_metrics if "utilization" in g]
        memory_used = [g["memory_used_mb"] for g in gpu_metrics if "memory_used_mb" in g]
        temperatures = [g["temperature"] for g in gpu_metrics if "temperature" in g]
        powers = [g["power_w"] for g in gpu_metrics if "power_w" in g]

        if utilizations:
            gpu_avg_utilization = round(mean(utilizations), 3)
            gpu_max_utilization = round(max(utilizations), 3)
        if memory_used:
            gpu_avg_memory_used_mb = round(mean(memory_used), 3)
            gpu_peak_memory_used_mb = round(max(memory_used), 3)
        if temperatures:
            gpu_avg_temperature = round(mean(temperatures), 3)
            gpu_max_temperature = round(max(temperatures), 3)
        if powers:
            gpu_avg_power_w = round(mean(powers), 3)
            gpu_max_power_w = round(max(powers), 3)

    return {
        "success_requests": len(success),
        "failed_requests": len(failed),
        "timeout_requests": timeout_requests,
        "oom_count": oom_count,
        "service_crash_count": service_crash_count,
        "e2e_avg_ms": round(mean(e2e_all), 3) if e2e_all else 0.0,
        "e2e_p50_ms": round(percentile(e2e_all, 0.50), 3),
        "e2e_p90_ms": round(percentile(e2e_all, 0.90), 3),
        "e2e_p99_ms": round(percentile(e2e_all, 0.99), 3),
        "qps": global_qps,
        "input_tokens_per_sec": global_input_tps,
        "output_tokens_per_sec": global_output_tps,
        "total_tokens_per_sec": global_total_tps,
        "decode_tps_p50": decode_tps_p50,
        "decode_tps_p99": decode_tps_p99,
        "prefill_tps_per_req_mean": prefill_mean,
        "prefill_tps_per_req_p50": prefill_p50,
        "prefill_tps_per_req_p99": prefill_p99,
        "ttft_p50_ms": round(percentile(all_ttft, 0.50), 3),
        "ttft_p90_ms": round(percentile(all_ttft, 0.90), 3),
        "ttft_p99_ms": round(percentile(all_ttft, 0.99), 3),
        "tpot_p50_ms": global_tpot_p50,
        "tpot_p90_ms": round(percentile(all_tpot, 0.90), 3),
        "tpot_p99_ms": global_tpot_p99,
        "gpu_avg_utilization": gpu_avg_utilization,
        "gpu_max_utilization": gpu_max_utilization,
        "gpu_avg_memory_used_mb": gpu_avg_memory_used_mb,
        "gpu_peak_memory_used_mb": gpu_peak_memory_used_mb,
        "gpu_avg_temperature": gpu_avg_temperature,
        "gpu_max_temperature": gpu_max_temperature,
        "gpu_avg_power_w": gpu_avg_power_w,
        "gpu_max_power_w": gpu_max_power_w,
        "error_categories": dict(error_categories),
        "backend_results": backend_results,
    }
