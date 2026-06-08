from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from llm_bench.compare import compare_manifests


@dataclass
class RegressionThresholds:
    max_output_tps_drop_pct: float | None = 5.0
    max_total_tps_drop_pct: float | None = None
    max_qps_drop_pct: float | None = None
    max_e2e_p99_increase_pct: float | None = 20.0
    max_ttft_p99_increase_pct: float | None = 20.0
    max_tpot_p99_increase_pct: float | None = None
    max_failed_requests: int | None = 0
    require_comparable: bool = True


def evaluate_regression(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    thresholds: RegressionThresholds,
) -> dict[str, Any]:
    comparison = compare_manifests(baseline, candidate)
    violations: list[dict[str, Any]] = []
    comparability = (comparison.get("comparability") or {}).get("level")
    if thresholds.require_comparable and comparability != "strictly_comparable":
        violations.append(
            {
                "scope": "comparability",
                "metric": "level",
                "threshold": "strictly_comparable",
                "actual": comparability,
                "message": f"Run is not strictly comparable: {comparability}",
            }
        )

    _check_failure_budget(comparison, thresholds, violations)
    _check_summary_latency(comparison, thresholds, violations)
    _check_workloads(comparison, thresholds, violations)
    return {
        "status": "fail" if violations else "pass",
        "thresholds": asdict(thresholds),
        "violations": violations,
        "comparison": comparison,
    }


def write_gate_result(path: Path, result: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def _check_failure_budget(
    comparison: dict[str, Any],
    thresholds: RegressionThresholds,
    violations: list[dict[str, Any]],
) -> None:
    if thresholds.max_failed_requests is None:
        return
    for row in comparison.get("summary_deltas") or []:
        if row.get("metric") != "failed_requests":
            continue
        failed = row.get("candidate")
        if failed is not None and int(failed) > thresholds.max_failed_requests:
            violations.append(
                {
                    "scope": "summary",
                    "metric": "failed_requests",
                    "threshold": thresholds.max_failed_requests,
                    "actual": failed,
                    "message": f"Candidate failed_requests {failed} exceeds {thresholds.max_failed_requests}",
                }
            )


def _check_summary_latency(
    comparison: dict[str, Any],
    thresholds: RegressionThresholds,
    violations: list[dict[str, Any]],
) -> None:
    latency_limits = {
        "e2e_p99_ms": thresholds.max_e2e_p99_increase_pct,
        "ttft_p99_ms": thresholds.max_ttft_p99_increase_pct,
        "tpot_p99_ms": thresholds.max_tpot_p99_increase_pct,
    }
    for row in comparison.get("summary_deltas") or []:
        limit = latency_limits.get(row.get("metric"))
        if limit is None:
            continue
        delta_pct = row.get("delta_pct")
        if delta_pct is not None and float(delta_pct) > limit:
            violations.append(_violation("summary", row, limit, "increase_pct"))


def _check_workloads(
    comparison: dict[str, Any],
    thresholds: RegressionThresholds,
    violations: list[dict[str, Any]],
) -> None:
    throughput_limits = {
        "output_tokens_per_sec": thresholds.max_output_tps_drop_pct,
        "total_tokens_per_sec": thresholds.max_total_tps_drop_pct,
        "qps": thresholds.max_qps_drop_pct,
    }
    latency_limits = {
        "e2e_p99_ms": thresholds.max_e2e_p99_increase_pct,
        "ttft_p99_ms": thresholds.max_ttft_p99_increase_pct,
        "tpot_p99_ms": thresholds.max_tpot_p99_increase_pct,
    }
    for item in comparison.get("workload_deltas") or []:
        workload = item.get("workload") or {}
        scope = f"i{workload.get('input_tokens')}/o{workload.get('output_tokens')}/c{workload.get('concurrency')}"
        for row in item.get("metrics") or []:
            metric = row.get("metric")
            delta_pct = row.get("delta_pct")
            if delta_pct is None:
                continue
            drop_limit = throughput_limits.get(metric)
            if drop_limit is not None and float(delta_pct) < -float(drop_limit):
                violations.append(_violation(scope, row, drop_limit, "drop_pct"))
            increase_limit = latency_limits.get(metric)
            if increase_limit is not None and float(delta_pct) > float(increase_limit):
                violations.append(_violation(scope, row, increase_limit, "increase_pct"))


def _violation(scope: str, row: dict[str, Any], threshold: float, kind: str) -> dict[str, Any]:
    metric = row.get("metric")
    delta_pct = row.get("delta_pct")
    direction = "drop" if kind == "drop_pct" else "increase"
    return {
        "scope": scope,
        "metric": metric,
        "threshold": threshold,
        "actual_delta_pct": delta_pct,
        "baseline": row.get("baseline"),
        "candidate": row.get("candidate"),
        "message": f"{scope} {metric} {direction} {delta_pct:+.1f}% exceeds {threshold:.1f}%",
    }
