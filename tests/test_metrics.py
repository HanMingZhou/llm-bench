from llm_bench.backends.base import RequestMetric
from llm_bench.metrics import (
    _decode_tps_from_tpot_ms,
    _prefill_metrics_per_req,
    _safe_div,
    _wall_clock_seconds,
    percentile,
    summarize_requests,
)


def _ok(req_id="r", start=None, end=None, **kw):
    defaults = dict(
        request_id=req_id, backend="vllm", concurrency=1, input_tokens=8, output_tokens=4,
        requested_output_tokens=4, ttft_ms=1.0, tpot_ms=1.0, e2e_latency_ms=4.0,
    )
    defaults.update(kw)
    return RequestMetric(start_unix=start, end_unix=end, **defaults)


def test_percentile_basic():
    assert percentile([], 0.5) == 0.0
    assert percentile([1, 2, 3, 4, 5], 0.5) == 3.0
    assert percentile([1, 2, 3, 4, 5], 1.0) == 5.0


def test_wall_clock_prefers_real_timestamps():
    rows = [_ok(start=1000.0, end=1002.0), _ok(start=1001.0, end=1004.0)]
    # max(end)-min(start) = 1004 - 1000 = 4
    assert _wall_clock_seconds(rows, fallback=99.0) == 4.0


def test_wall_clock_falls_back_when_no_timestamps():
    rows = [_ok(), _ok()]
    assert _wall_clock_seconds(rows, fallback=7.5) == 7.5


def test_summarize_aggregates_gpu_metrics_with_friendly_keys():
    # Regression for the silent "GPU 平均利用率 0.0%" bug: gpu.py used to write
    # `utilization.gpu` while metrics.py read `utilization`, so all GPU values
    # were always 0. The keys must be aligned.
    gpu_rows = [
        {"utilization": 80, "memory_used_mb": 18000, "temperature": 70, "power_w": 280},
        {"utilization": 90, "memory_used_mb": 19500, "temperature": 75, "power_w": 295},
    ]
    summary = summarize_requests([_ok(start=1000.0, end=1001.0)], gpu_metrics=gpu_rows)
    assert summary["gpu_avg_utilization"] == 85.0
    assert summary["gpu_max_utilization"] == 90.0
    assert summary["gpu_peak_memory_used_mb"] == 19500
    assert summary["gpu_avg_temperature"] == 72.5
    assert summary["gpu_max_power_w"] == 295


def test_safe_div_zero_denominator_returns_zero():
    assert _safe_div(100, 0) == 0.0
    assert _safe_div(100, None) == 0.0  # type: ignore[arg-type]
    assert _safe_div(100, 5) == 20.0


def test_decode_tps_inverts_tpot():
    # Decode TPS = 1000 / TPOT(ms). 20ms TPOT → 50 tok/s per-request decode.
    assert _decode_tps_from_tpot_ms(20.0) == 50.0
    assert _decode_tps_from_tpot_ms(28.849) == round(1000.0 / 28.849, 3)
    # Zero / negative TPOT (no decode samples) → 0, not divide-by-zero.
    assert _decode_tps_from_tpot_ms(0) == 0.0
    assert _decode_tps_from_tpot_ms(-1) == 0.0


def test_prefill_tps_per_req_aggregates_mean_p50_p99():
    # Prefill TPS per request = input_tokens / TTFT(s).
    # 512 / 0.064s = 8000 tok/s, 512 / 0.128s = 4000 tok/s.
    rows = [
        RequestMetric("r1", "vllm", 1, 512, 4, 4, ttft_ms=64.0, tpot_ms=10, e2e_latency_ms=100),
        RequestMetric("r2", "vllm", 1, 512, 4, 4, ttft_ms=128.0, tpot_ms=10, e2e_latency_ms=100),
    ]
    mean_val, p50, p99 = _prefill_metrics_per_req(rows)
    assert mean_val == 6000.0
    # p50/p99 lie inside [4000, 8000], with p99 close to 8000.
    assert 4000.0 <= p50 <= 8000.0
    assert p99 >= 7000.0


def test_prefill_tps_skips_zero_inputs_or_ttft():
    rows = [
        RequestMetric("a", "vllm", 1, 0, 4, 4, ttft_ms=64.0, tpot_ms=10, e2e_latency_ms=100),
        RequestMetric("b", "vllm", 1, 512, 4, 4, ttft_ms=0, tpot_ms=10, e2e_latency_ms=100),
    ]
    # Both rows are skipped → all-zero result instead of div/0 crash.
    assert _prefill_metrics_per_req(rows) == (0.0, 0.0, 0.0)


def test_summarize_emits_per_request_token_speeds():
    # 16 requests, single concurrency, deterministic TPOT=20ms, TTFT=64ms,
    # 512 input / 128 output, 2.624s e2e.
    rows = [
        RequestMetric(
            f"r{i}", "vllm", 1, 512, 128, 128,
            ttft_ms=64.0, tpot_ms=20.0, e2e_latency_ms=2624.0,
            start_unix=1000.0 + i * 0.1, end_unix=1002.624 + i * 0.1,
        )
        for i in range(16)
    ]
    summary = summarize_requests(rows)

    # Global aggregates.
    assert summary["decode_tps_p50"] == 50.0   # 1000 / 20
    assert summary["decode_tps_p99"] == 50.0
    assert summary["prefill_tps_per_req_mean"] == 8000.0   # 512 / 0.064
    assert summary["prefill_tps_per_req_p99"] == 8000.0
    # input_tokens_per_sec should also be exposed (was missing in old report).
    assert summary["input_tokens_per_sec"] > 0
    assert summary["output_tokens_per_sec"] > 0

    # Per-workload row gets the same fields so the report can split per (i/o/c).
    wl = summary["backend_results"][0]["metrics"]
    assert wl["decode_tps_p50"] == 50.0
    assert wl["prefill_tps_per_req_mean"] == 8000.0


def test_summarize_counts_errors_by_category():
    rows = [
        _ok(),
        _ok(req_id="r2"),
    ]
    # Manually build failure rows
    fail_oom = RequestMetric(
        request_id="oom", backend="vllm", concurrency=1, input_tokens=8, output_tokens=0,
        requested_output_tokens=4, ttft_ms=0.0, tpot_ms=0.0, e2e_latency_ms=100.0,
        success=False, error="CUDA OOM", error_category="oom",
    )
    fail_timeout = RequestMetric(
        request_id="to", backend="vllm", concurrency=1, input_tokens=8, output_tokens=0,
        requested_output_tokens=4, ttft_ms=0.0, tpot_ms=0.0, e2e_latency_ms=120000.0,
        success=False, error="timed out", error_category="timeout",
    )
    summary = summarize_requests([*rows, fail_oom, fail_timeout])
    assert summary["success_requests"] == 2
    assert summary["failed_requests"] == 2
    assert summary["oom_count"] == 1
    assert summary["timeout_requests"] == 1
    assert summary["error_categories"] == {"oom": 1, "timeout": 1}
