from llm_bench.gpu import _coerce


def test_query_gpu_metrics_aliases_friendly_keys(monkeypatch):
    """Regression test: gpu.py must produce both nvidia-smi dotted names AND
    snake_case friendly aliases, otherwise summarize_requests reads back 0.0
    for every GPU stat (the silent "GPU 平均利用率 0.0%" bug)."""
    from llm_bench import gpu

    csv = "2026-06-07 21:18:21,0,NVIDIA GeForce RTX 3090,82,18000,24576,71,275\n"

    class FakeProc:
        returncode = 0
        stdout = csv
        stderr = ""

    monkeypatch.setattr(gpu.shutil, "which", lambda _name: "/usr/bin/nvidia-smi")
    monkeypatch.setattr(gpu.subprocess, "run", lambda *a, **kw: FakeProc())

    rows = gpu.query_gpu_metrics()
    assert rows, "expected at least one sampled row"
    row = rows[0]
    # Friendly aliases must be present (these are what metrics.py / report.py read).
    assert row["utilization"] == 82
    assert row["memory_used_mb"] == 18000
    assert row["memory_total_mb"] == 24576
    assert row["temperature"] == 71
    assert row["power_w"] == 275
    # Original dotted names also retained for back-compat with archived jsonl.
    assert row["utilization.gpu"] == 82
    assert row["memory.used"] == 18000
    assert row["power.draw"] == 275
    assert "sample_time_unix" in row


def test_coerce_handles_int_float_string():
    assert _coerce("42") == 42
    assert _coerce("3.14") == 3.14
    assert _coerce("NVIDIA GeForce RTX 3090") == "NVIDIA GeForce RTX 3090"
