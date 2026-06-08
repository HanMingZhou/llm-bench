from llm_bench.backends.base import BackendResult, RequestMetric
from llm_bench.config import BenchConfig
from llm_bench.metrics import summarize_requests
from llm_bench.report import render_markdown_report


def _sample_summary(stream: bool = True) -> tuple[BenchConfig, BackendResult, dict]:
    cfg = BenchConfig()
    cfg.backend.name = "vllm"
    cfg.backend.image = "vllm/vllm-openai:latest"
    cfg.backend.model_name = "Qwen3.5-9B"
    cfg.backend.port = 8000
    cfg.backend.command = ["/m", "--host", "0.0.0.0"]
    cfg.workload.profile = "quick"
    cfg.workload.stream = stream

    rows = []
    for c in (1, 4):
        for i in range(8):
            rows.append(RequestMetric(
                f"r_{c}_{i}", "vllm", c, 512, 128, 128,
                ttft_ms=64.0 * c, tpot_ms=20.0 * c, e2e_latency_ms=64.0 * c + 19 * 20.0 * c,
                start_unix=1000.0 + i * 0.1, end_unix=1000.5 + i * 0.1,
            ))
    gpu = [{"utilization": 82, "memory_used_mb": 18000, "temperature": 72, "power_w": 280}]
    summary = summarize_requests(rows, gpu_metrics=gpu)
    return cfg, BackendResult("vllm", rows, gpu_metrics=gpu), summary


def _runtime() -> dict:
    return {
        "gpu": {
            "gpu_count": 2,
            "gpus": [
                {"name": "NVIDIA GeForce RTX 3090", "memory_total_mb": "24576"},
                {"name": "NVIDIA GeForce RTX 3090", "memory_total_mb": "24576"},
            ],
        },
        "docker": {"installed": True, "daemon_ok": True, "image": "vllm/vllm-openai:latest"},
        "port": {"available": True},
        "disk": {"free_gb": 200},
    }


def _manifest() -> dict:
    return {
        "run_id": "test-run",
        "backend": {
            "container_command": ["/m"],
            "docker_args": ["--gpus", "all"],
            "launch_command": ["docker", "run", "vllm/vllm-openai:latest", "/m"],
        },
    }


def test_render_markdown_report_contains_headline_sections():
    cfg, result, summary = _sample_summary()
    md = render_markdown_report(_manifest(), summary, cfg, result, _runtime(), [])
    # TL;DR comes first and surfaces the new token-speed metrics.
    assert md.startswith("# 推理压测报告 · Qwen3.5-9B")
    assert "## TL;DR" in md
    for key in (
        "Output TPS",
        "Decode TPS",
        "Prefill TPS",
        "Input TPS",
        "TTFT p99",
    ):
        assert key in md, f"missing {key} in report"
    # GPU number table exists (regression: previously only an image was shown).
    for key in ("Util Avg", "Util Max", "Mem Peak", "Power Max"):
        assert key in md, f"missing GPU stat {key} in report"
    # Glossary explains Output TPS vs QPS (now under "四、名词解释").
    assert "名词解释" in md
    assert "QPS" in md  # QPS is still mentioned but downgraded from headline.
    # Section order: 配置 (env/backend/workload) MUST come before 性能指标.
    assert md.index("# 一、配置") < md.index("# 二、性能指标")
    assert md.index("# 二、性能指标") < md.index("# 三、错误与建议")


def test_render_markdown_report_per_workload_tables_have_decode_prefill():
    cfg, result, summary = _sample_summary()
    md = render_markdown_report(_manifest(), summary, cfg, result, _runtime(), [])
    # Per-workload section has its own throughput / decode-prefill / latency tables.
    assert "## 分 workload 明细" in md
    assert "Decode p50" in md and "Prefill mean" in md
    assert "TTFT p50" in md and "TTFT p99" in md
    assert "i512/o128/c1" in md and "i512/o128/c4" in md


def test_render_markdown_report_transformers_keeps_caveat():
    cfg, result, summary = _sample_summary()
    cfg.backend.name = "transformers"
    cfg.transformers.model_path = "/models/qwen"
    md = render_markdown_report(_manifest(), summary, cfg, result, _runtime(), [])
    # Transformers backend section + the per-request-latency caveat at the top.
    assert "## 后端配置 (transformers)" in md
    assert "ℹ transformers 后端把 concurrency 当作 batch_size" in md


def test_render_markdown_report_dry_run_short_circuits_perf_summary():
    cfg, result, summary = _sample_summary()
    cfg.backend.name = "dry-run"
    md = render_markdown_report(_manifest(), summary, cfg, result, _runtime(), [])
    # dry-run shows the warning banner and skips the "性能摘要" section.
    assert "自检 run (dry-run)" in md
    assert "## 性能摘要" not in md
