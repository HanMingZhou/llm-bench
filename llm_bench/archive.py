from __future__ import annotations

import json
import platform
import shutil
import shlex
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from llm_bench.backends.base import BackendResult
from llm_bench.charts import render_summary_charts
from llm_bench.config import BenchConfig
from llm_bench.gpu import query_gpu_topology
from llm_bench.metrics import summarize_requests
from llm_bench.textutil import slug
from llm_bench.yaml_io import dump_yaml


def create_run_dir(config: BenchConfig) -> Path:
    created = datetime.now().astimezone()
    label = config.backend.model_name or config.transformers.model_path or "unknown-model"
    model_name = slug(label)
    backend = slug(config.backend.name)
    run_name = config.report.run_name or f"{created:%Y%m%d_%H%M%S}_{model_name}_{backend}"
    return _make_run_dir(Path(config.report.output_dir), run_name)


def _make_run_dir(parent: Path, run_name: str) -> Path:
    # Avoid crashing when two runs collide within the same second.
    candidate = parent / run_name
    suffix = 1
    while True:
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        except FileExistsError:
            candidate = parent / f"{run_name}_{suffix}"
            suffix += 1


def write_run_archive(
    run_dir: Path,
    config: BenchConfig,
    requested: dict[str, Any],
    result: BackendResult,
    runtime: dict[str, Any] | None = None,
) -> dict[str, Any]:
    reports_dir = run_dir / "reports"
    images_dir = reports_dir / "images"
    logs_dir = run_dir / "logs"
    reports_dir.mkdir(exist_ok=True)
    logs_dir.mkdir(exist_ok=True)

    summary = summarize_requests(result.request_metrics, gpu_metrics=result.gpu_metrics)
    created_at = datetime.now().astimezone().isoformat(timespec="seconds")

    if config.backend.name == "transformers":
        model_block = {
            "name": config.backend.model_name or config.transformers.model_path,
            "path": config.transformers.model_path,
        }
    else:
        model_block = {"name": config.backend.model_name}

    backend_block: dict[str, Any] = {
        "name": config.backend.name,
        "image": config.backend.image,
        "port": config.backend.port,
        "container_command": list(config.backend.command),
        "docker_args": list(config.backend.docker_args),
        "startup_seconds": result.startup_seconds,
        "launch_command": list(config.backend.launch_command),
    }
    if config.backend.name == "transformers":
        backend_block["transformers"] = asdict(config.transformers)
    if result.peak_memory_mb is not None:
        backend_block["peak_memory_mb"] = result.peak_memory_mb

    manifest = {
        "run_id": run_dir.name,
        "created_at": created_at,
        "model": model_block,
        "hardware": _hardware_from_runtime(runtime),
        "backend": backend_block,
        "workload": asdict(config.workload),
        "summary": summary,
        "tags": config.report.tags,
    }

    dump_yaml(run_dir / "config.requested.yaml", requested)
    dump_yaml(run_dir / "config.resolved.yaml", config.to_dict())
    _write_json(run_dir / "run_manifest.json", manifest)
    _write_json(run_dir / "environment.json", collect_environment(runtime))
    _write_json(run_dir / "metrics.summary.json", summary)
    if config.report.save_request_metrics:
        _write_jsonl(run_dir / "metrics.requests.jsonl", [m.to_dict() for m in result.request_metrics])
    if config.report.save_gpu_metrics:
        _write_jsonl(run_dir / "metrics.gpu.jsonl", result.gpu_metrics or [])
    if config.report.include_samples:
        _write_jsonl(run_dir / "samples.jsonl", _sample_rows(result))
    if config.report.save_logs:
        # stdout_log holds the merged stdout+stderr stream (vllm/sglang/uvicorn
        # write INFO via stderr, so we union both into one timeline). Keep
        # stderr_log handling for backwards-compat configs that still set it.
        merged_src = Path(config.backend.stdout_log) if config.backend.stdout_log else None
        fallback = "\n".join(result.errors or [])
        _move_or_write(merged_src, logs_dir / "backend.log", fallback)
        if config.backend.stderr_log:
            stderr_src = Path(config.backend.stderr_log)
            if stderr_src.exists():
                _move_or_write(stderr_src, logs_dir / "backend.stderr.log", "")
    else:
        # Even when logs aren't archived, drop the temp backend log files the
        # docker serving backend created so they don't pile up in output_dir.
        _cleanup_temp_logs(config)
    if config.backend.launch_command:
        plan = run_dir / "launch_plan.sh"
        plan.write_text(
            "#!/usr/bin/env bash\nset -euo pipefail\n"
            + shlex.join(config.backend.launch_command)
            + "\n",
            encoding="utf-8",
        )
        plan.chmod(0o755)

    from llm_bench.report import render_markdown_report

    chart_paths = render_summary_charts(summary, images_dir, result.gpu_metrics or [])
    (reports_dir / "inference_report.md").write_text(
        render_markdown_report(manifest, summary, config, result, runtime or {}, chart_paths),
        encoding="utf-8",
    )
    return manifest


def collect_environment(runtime: dict[str, Any] | None = None) -> dict[str, Any]:
    data = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "machine": platform.machine(),
        "processor": platform.processor(),
    }
    if runtime:
        data["runtime_checks"] = runtime
    data["gpu_topology"] = query_gpu_topology()
    return data


def _hardware_from_runtime(runtime: dict[str, Any] | None) -> dict[str, Any]:
    gpu = (runtime or {}).get("gpu") or {}
    gpus = gpu.get("gpus") or []
    gpu_models = sorted({str(item.get("name")) for item in gpus if item.get("name")})
    return {
        "gpu_model": ", ".join(gpu_models) if gpu_models else None,
        "gpu_count": gpu.get("gpu_count", 0),
        "interconnect": _interconnect_summary(),
    }


def _interconnect_summary() -> str | None:
    topo = query_gpu_topology()
    if not topo.get("available"):
        return None
    links = topo.get("links") or []
    return ",".join(str(link) for link in links) if links else "unknown"


def _sample_rows(result: BackendResult) -> list[dict[str, Any]]:
    rows = []
    for metric in result.request_metrics:
        data = metric.to_dict()
        if data.get("prompt_sample") or data.get("output_sample"):
            rows.append(
                {
                    "request_id": data["request_id"],
                    "backend": data["backend"],
                    "prompt_sample": data.get("prompt_sample"),
                    "output_sample": data.get("output_sample"),
                    "output_valid": data.get("output_valid"),
                    "validation_error": data.get("validation_error"),
                    "metadata": data.get("metadata"),
                }
            )
    return rows


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _move_or_write(src: Path | None, dst: Path, fallback: str) -> None:
    if src and src.exists():
        shutil.move(str(src), str(dst))
    else:
        dst.write_text(fallback, encoding="utf-8")


def _cleanup_temp_logs(config: BenchConfig) -> None:
    for raw in (config.backend.stdout_log, config.backend.stderr_log):
        if not raw:
            continue
        try:
            path = Path(raw)
            if path.exists():
                path.unlink()
        except OSError:
            pass
