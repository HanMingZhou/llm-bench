from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from llm_bench.charts import render_compare_chart
from llm_bench.textutil import slug


SUMMARY_METRICS = [
    "e2e_p50_ms",
    "e2e_p90_ms",
    "e2e_p99_ms",
    "ttft_p99_ms",
    "tpot_p99_ms",
    "output_tokens_per_sec",
    "qps",
    "success_requests",
    "failed_requests",
    "timeout_requests",
    "oom_count",
]

WORKLOAD_METRICS = [
    "qps",
    "output_tokens_per_sec",
    "total_tokens_per_sec",
    "ttft_p99_ms",
    "tpot_p99_ms",
    "e2e_p99_ms",
    "success_requests",
    "failed_requests",
]


def compare_manifests(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    comparability = compare_fields(baseline, candidate)
    return {
        "baseline_run_id": baseline.get("run_id"),
        "candidate_run_id": candidate.get("run_id"),
        "comparability": comparability,
        "summary_deltas": [
            _metric_delta(metric, (baseline.get("summary") or {}).get(metric), (candidate.get("summary") or {}).get(metric))
            for metric in SUMMARY_METRICS
        ],
        "workload_deltas": _workload_deltas(baseline, candidate),
    }


def write_compare_report(
    baseline_run_dir: Path,
    candidate_run_dir: Path,
    output_dir: Path | None = None,
) -> Path:
    baseline = load_manifest(baseline_run_dir)
    candidate = load_manifest(candidate_run_dir)
    comparison = compare_manifests(baseline, candidate)
    output = output_dir or candidate_run_dir / "reports" / "compare"
    images = output / "images"
    output.mkdir(parents=True, exist_ok=True)
    images.mkdir(parents=True, exist_ok=True)
    chart = render_compare_chart(comparison, images)
    report_path = output / "compare_report.md"
    report_path.write_text(_render_markdown(comparison, chart), encoding="utf-8")
    (output / "compare_metrics.json").write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return report_path


def compare_fields(baseline: dict[str, object], candidate: dict[str, object]) -> dict[str, object]:
    # Core fields that must match for strictly_comparable.
    core_fields = [
        ("model.name", ("model", "name")),
        ("backend.name", ("backend", "name")),
        ("backend.image", ("backend", "image")),
        ("backend.container_command", ("backend", "container_command")),
        ("workload.input_tokens", ("workload", "input_tokens")),
        ("workload.output_tokens", ("workload", "output_tokens")),
        ("workload.concurrency", ("workload", "concurrency")),
        ("workload.stream", ("workload", "stream")),
        ("workload.total_requests", ("workload", "total_requests")),
        ("workload.prompt_jsonl", ("workload", "prompt_jsonl")),
        ("workload.prompt_dir", ("workload", "prompt_dir")),
    ]
    # Environment fields: differences here make it partially_comparable, not not_comparable.
    env_fields = [
        ("hardware.gpu_model", ("hardware", "gpu_model")),
        ("hardware.gpu_count", ("hardware", "gpu_count")),
    ]
    diffs = []
    for label, path in core_fields + env_fields:
        left = get_nested(baseline, path)
        right = get_nested(candidate, path)
        if left != right:
            diffs.append({"field": label, "baseline": left, "candidate": right})
    if not diffs:
        level = "strictly_comparable"
    else:
        env_field_names = {f[0] for f in env_fields}
        core_diffs = [d for d in diffs if d["field"] not in env_field_names]
        # Only environment/version fields differ → still partially comparable.
        level = "partially_comparable" if not core_diffs else "not_comparable"
    return {"level": level, "diffs": diffs}


def find_baseline_for_run(run_dir: Path, baselines_dir: Path) -> Path:
    manifest = load_manifest(run_dir)
    model = slug((manifest.get("model") or {}).get("name") or (manifest.get("model") or {}).get("path") or "unknown-model")
    hardware = slug((manifest.get("hardware") or {}).get("gpu_model") or "unknown-hardware")
    backend = slug((manifest.get("backend") or {}).get("name") or "unknown-backend")
    baseline_index = baselines_dir / model / hardware / f"{backend}.json"
    if not baseline_index.exists():
        raise FileNotFoundError(f"No baseline found: {baseline_index}")
    data = json.loads(baseline_index.read_text(encoding="utf-8"))
    baseline_run_dir = Path(str(data["run_dir"]))
    if not baseline_run_dir.exists():
        raise FileNotFoundError(f"Baseline run directory does not exist: {baseline_run_dir}")
    return baseline_run_dir


def load_manifest(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "run_manifest.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def get_nested(data: dict[str, object], path: tuple[str, ...]) -> object:
    current: object = data
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _workload_deltas(baseline: dict[str, Any], candidate: dict[str, Any]) -> list[dict[str, Any]]:
    base_map = _workload_map(baseline)
    cand_map = _workload_map(candidate)
    rows = []
    for key in sorted(set(base_map) | set(cand_map)):
        base_metrics = base_map.get(key, {})
        cand_metrics = cand_map.get(key, {})
        rows.append(
            {
                "workload": {
                    "input_tokens": key[0],
                    "output_tokens": key[1],
                    "concurrency": key[2],
                },
                "metrics": [
                    _metric_delta(metric, base_metrics.get(metric), cand_metrics.get(metric)) for metric in WORKLOAD_METRICS
                ],
            }
        )
    return rows


def _workload_map(manifest: dict[str, Any]) -> dict[tuple[int, int, int], dict[str, Any]]:
    results = (manifest.get("summary") or {}).get("backend_results") or []
    mapped = {}
    for item in results:
        workload = item.get("workload") or {}
        key = (
            int(workload.get("input_tokens") or 0),
            int(workload.get("output_tokens") or 0),
            int(workload.get("concurrency") or 0),
        )
        mapped[key] = item.get("metrics") or {}
    return mapped


def _metric_delta(metric: str, baseline_value: Any, candidate_value: Any) -> dict[str, Any]:
    pct = None
    if baseline_value not in (None, 0) and candidate_value is not None:
        try:
            pct = (float(candidate_value) - float(baseline_value)) / float(baseline_value) * 100.0
        except (TypeError, ValueError):
            pct = None
    return {
        "metric": metric,
        "baseline": baseline_value,
        "candidate": candidate_value,
        "delta": None if baseline_value is None or candidate_value is None else candidate_value - baseline_value,
        "delta_pct": None if pct is None else round(pct, 3),
    }


def _render_markdown(comparison: dict[str, Any], chart: Path | None) -> str:
    comparability = comparison["comparability"]
    diffs = comparability.get("diffs") or []
    diff_lines = "\n".join(
        f"- {d['field']}: `{d['baseline']}` -> `{d['candidate']}`" for d in diffs
    ) or "- core fields match"
    summary_rows = "\n".join(_delta_row(row) for row in comparison["summary_deltas"])
    workload_rows = []
    for item in comparison["workload_deltas"]:
        workload = item["workload"]
        label = f"i{workload['input_tokens']}/o{workload['output_tokens']}/c{workload['concurrency']}"
        for metric in item["metrics"]:
            workload_rows.append(_delta_row(metric, label))
    chart_md = f"![compare_metrics](images/{chart.name})\n" if chart else ""

    # Separate throughput, latency, and error deltas.
    throughput_metric_names = {"qps", "output_tokens_per_sec", "total_tokens_per_sec"}
    latency_metric_names = {"ttft_p99_ms", "tpot_p99_ms", "e2e_p99_ms"}
    error_metric_names = {"success_requests", "failed_requests"}

    throughput_wl_rows = []
    latency_wl_rows = []
    error_wl_rows = []
    for item in comparison["workload_deltas"]:
        workload = item["workload"]
        label = f"i{workload['input_tokens']}/o{workload['output_tokens']}/c{workload['concurrency']}"
        for metric in item["metrics"]:
            row = _delta_row(metric, label)
            name = metric.get("metric", "")
            if name in throughput_metric_names:
                throughput_wl_rows.append(row)
            elif name in latency_metric_names:
                latency_wl_rows.append(row)
            elif name in error_metric_names:
                error_wl_rows.append(row)
            else:
                throughput_wl_rows.append(row)

    return f"""# 推理结果对比报告

## 对比摘要

- baseline: `{comparison["baseline_run_id"]}`
- candidate: `{comparison["candidate_run_id"]}`
- comparability: `{comparability["level"]}`

{chart_md}

## 可比性判断

可比等级: `{comparability["level"]}`

{diff_lines}

## 配置差异

| field | baseline | candidate |
|---|---|---|
{chr(10).join(f"| {d['field']} | `{d['baseline']}` | `{d['candidate']}` |" for d in diffs) or "| - | - | - |"}

## 吞吐变化

| workload | metric | baseline | candidate | delta | delta % |
|---|---|---:|---:|---:|---:|
{chr(10).join(throughput_wl_rows) or "| - | - | - | - | - | - |"}

## 延迟变化

| workload | metric | baseline | candidate | delta | delta % |
|---|---|---:|---:|---:|---:|
{chr(10).join(latency_wl_rows) or "| - | - | - | - | - | - |"}

## 错误率变化

| workload | metric | baseline | candidate | delta | delta % |
|---|---|---:|---:|---:|---:|
{chr(10).join(error_wl_rows) or "| - | - | - | - | - | - |"}

## 核心指标总览

| metric | baseline | candidate | delta | delta % |
|---|---:|---:|---:|---:|
{summary_rows}

## 结论

- 可比性等级: `{comparability["level"]}`
- 配置差异数: `{len(diffs)}`
- 详细 workload 级指标变化请参见上方各章节。
"""


def _delta_row(row: dict[str, Any], prefix: str | None = None) -> str:
    values = [
        row["metric"],
        _fmt(row["baseline"]),
        _fmt(row["candidate"]),
        _fmt(row["delta"]),
        _fmt_pct(row["delta_pct"]),
    ]
    if prefix is not None:
        values = [prefix, *values]
    return "| " + " | ".join(values) + " |"


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def _fmt_pct(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value):+.1f}%"


