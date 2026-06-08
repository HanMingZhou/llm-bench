import json
import os
import time
from pathlib import Path

from llm_bench.cleanup import build_cleanup_plan
from llm_bench.compare import compare_manifests
from llm_bench.regression import RegressionThresholds, evaluate_regression


def test_compare_manifests_basic():
    baseline = {
        "run_id": "a",
        "model": {"name": "m", "dtype": "bfloat16"},
        "backend": {"name": "dry-run", "tensor_parallel_size": 1},
        "workload": {"input_tokens": [1], "output_tokens": [1], "concurrency": [1], "stream": False},
        "summary": {"e2e_p99_ms": 100, "success_requests": 1, "failed_requests": 0, "backend_results": []},
    }
    candidate = json.loads(json.dumps(baseline))
    candidate["run_id"] = "b"
    candidate["summary"]["e2e_p99_ms"] = 110
    comparison = compare_manifests(baseline, candidate)
    assert comparison["comparability"]["level"] == "strictly_comparable"
    assert comparison["summary_deltas"][2]["delta_pct"] == 10.0


def test_cleanup_plan(tmp_path: Path):
    run = tmp_path / "run"
    run.mkdir()
    req = run / "metrics.requests.jsonl"
    req.write_text("x", encoding="utf-8")
    old = time.time() - 10 * 86400
    os.utime(req, (old, old))
    plan = build_cleanup_plan(tmp_path, request_metrics_days=1)
    assert req in plan.delete_files


def test_regression_gate_detects_latency_increase():
    baseline = {
        "run_id": "a",
        "model": {"name": "m", "dtype": "bfloat16"},
        "backend": {"name": "dry-run", "tensor_parallel_size": 1},
        "workload": {"input_tokens": [1], "output_tokens": [1], "concurrency": [1], "stream": False},
        "summary": {
            "e2e_p99_ms": 100,
            "success_requests": 1,
            "failed_requests": 0,
            "backend_results": [
                {
                    "workload": {"input_tokens": 1, "output_tokens": 1, "concurrency": 1},
                    "metrics": {"e2e_p99_ms": 100, "output_tokens_per_sec": 100, "failed_requests": 0},
                }
            ],
        },
    }
    candidate = json.loads(json.dumps(baseline))
    candidate["run_id"] = "b"
    candidate["summary"]["e2e_p99_ms"] = 150
    candidate["summary"]["backend_results"][0]["metrics"]["e2e_p99_ms"] = 150
    result = evaluate_regression(
        baseline,
        candidate,
        RegressionThresholds(max_e2e_p99_increase_pct=20, max_output_tps_drop_pct=None),
    )
    assert result["status"] == "fail"
    assert any(v["metric"] == "e2e_p99_ms" for v in result["violations"])
