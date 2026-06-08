import json

from llm_bench.archive import write_run_archive
from llm_bench.backends.dry_run import DryRunBackend
from llm_bench.backends.base import BackendResult, RequestMetric
from llm_bench.config import BenchConfig


def test_archive_respects_save_flags_and_writes_samples(tmp_path):
    config = BenchConfig()
    config.report.save_request_metrics = False
    config.report.save_gpu_metrics = False
    config.report.save_logs = False
    config.report.include_samples = True
    metric = RequestMetric(
        request_id="req_1",
        backend="dry-run",
        concurrency=1,
        input_tokens=8,
        output_tokens=4,
        requested_output_tokens=4,
        ttft_ms=1.0,
        tpot_ms=1.0,
        e2e_latency_ms=4.0,
        prompt_sample="hello",
        output_sample="world",
        output_valid=True,
    )
    runtime = {
        "gpu": {
            "gpu_count": 2,
            "gpus": [
                {"index": "0", "name": "NVIDIA A100", "memory_total_mb": "81920"},
                {"index": "1", "name": "NVIDIA A100", "memory_total_mb": "81920"},
            ],
        }
    }

    manifest = write_run_archive(tmp_path, config, {}, BackendResult("dry-run", [metric]), runtime)

    assert manifest["hardware"]["gpu_model"] == "NVIDIA A100"
    assert manifest["hardware"]["gpu_count"] == 2
    assert not (tmp_path / "metrics.requests.jsonl").exists()
    assert not (tmp_path / "metrics.gpu.jsonl").exists()
    assert not (tmp_path / "logs" / "backend.stderr.log").exists()
    assert not (tmp_path / "logs" / "backend.log").exists()
    assert not (tmp_path / "model.json").exists()
    assert not (tmp_path / "backend.json").exists()
    assert not (tmp_path / "workload.json").exists()
    assert not (tmp_path / "reports" / "inference_metrics.json").exists()
    samples = [json.loads(line) for line in (tmp_path / "samples.jsonl").read_text(encoding="utf-8").splitlines()]
    assert samples[0]["prompt_sample"] == "hello"
    assert samples[0]["output_sample"] == "world"


def test_archive_writes_merged_backend_log(tmp_path):
    # docker_serving 把 stdout+stderr 写到一个文件，archive 应把它落到 logs/backend.log
    src = tmp_path / "active.log"
    src.write_text("INFO booting\nINFO ready\n")

    config = BenchConfig()
    config.backend.name = "vllm"
    config.backend.image = "vllm:test"
    config.backend.model_name = "m"
    config.backend.stdout_log = str(src)
    config.backend.stderr_log = ""
    config.report.save_logs = True

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    metric = RequestMetric(
        request_id="r", backend="vllm", concurrency=1, input_tokens=1, output_tokens=1,
        requested_output_tokens=1, ttft_ms=1.0, tpot_ms=1.0, e2e_latency_ms=1.0,
    )
    write_run_archive(run_dir, config, {}, BackendResult("vllm", [metric]),
                      {"gpu": {"gpu_count": 0, "gpus": []}})

    merged = run_dir / "logs" / "backend.log"
    assert merged.exists()
    text = merged.read_text(encoding="utf-8")
    assert "INFO booting" in text and "INFO ready" in text
    # 旧的双文件不再生成
    assert not (run_dir / "logs" / "backend.stdout.log").exists()
    assert not (run_dir / "logs" / "backend.stderr.log").exists()
    # 临时日志被搬走
    assert not src.exists()


def test_archive_save_logs_false_cleans_temp_log_file(tmp_path):
    src = tmp_path / "leak.log"
    src.write_text("temp")
    config = BenchConfig()
    config.backend.name = "vllm"
    config.backend.stdout_log = str(src)
    config.backend.stderr_log = ""
    config.report.save_logs = False
    config.report.save_request_metrics = False
    config.report.save_gpu_metrics = False

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    metric = RequestMetric("r", "vllm", 1, 1, 1, 1, 1.0, 1.0, 1.0)
    write_run_archive(run_dir, config, {}, BackendResult("vllm", [metric]),
                      {"gpu": {"gpu_count": 0, "gpus": []}})

    assert not src.exists(), "temp log should be removed even when save_logs=False"
    assert not (run_dir / "logs" / "backend.log").exists()


def test_dry_run_duration_generates_bounded_rows():
    config = BenchConfig()
    config.workload.concurrency = [2]
    config.workload.duration_seconds = 3
    config.workload.total_requests = 100

    result = DryRunBackend().run(config)

    assert len(result.request_metrics) == 6
