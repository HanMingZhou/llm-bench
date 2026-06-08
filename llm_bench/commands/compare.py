from __future__ import annotations

import argparse
from pathlib import Path

from llm_bench.compare import compare_fields, find_baseline_for_run, load_manifest, write_compare_report
from llm_bench.regression import RegressionThresholds, evaluate_regression, write_gate_result


def register_compare(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    compare = sub.add_parser("compare", help="Compare two historical runs.")
    compare.add_argument("--baseline", type=Path)
    compare.add_argument("--candidate", type=Path, required=True)
    compare.add_argument("--to-baseline", action="store_true")
    compare.add_argument("--baselines-dir", type=Path, default=Path("baselines"))
    compare.add_argument("--output-dir", type=Path)
    compare.set_defaults(func=cmd_compare)

    gate = sub.add_parser("gate", help="Fail CI when a candidate regresses past thresholds.")
    gate.add_argument("--baseline", type=Path)
    gate.add_argument("--candidate", type=Path, required=True)
    gate.add_argument("--to-baseline", action="store_true")
    gate.add_argument("--baselines-dir", type=Path, default=Path("baselines"))
    gate.add_argument("--output", type=Path)
    gate.add_argument("--max-output-tps-drop-pct", type=float, default=5.0)
    gate.add_argument("--max-total-tps-drop-pct", type=float)
    gate.add_argument("--max-qps-drop-pct", type=float)
    gate.add_argument("--max-e2e-p99-increase-pct", type=float, default=20.0)
    gate.add_argument("--max-ttft-p99-increase-pct", type=float, default=20.0)
    gate.add_argument("--max-tpot-p99-increase-pct", type=float)
    gate.add_argument("--max-failed-requests", type=int, default=0)
    gate.add_argument("--require-comparable", action=argparse.BooleanOptionalAction, default=True)
    gate.set_defaults(func=cmd_gate)


def cmd_compare(args: argparse.Namespace) -> None:
    baseline_dir = _resolve_baseline(args)
    baseline = load_manifest(baseline_dir)
    candidate = load_manifest(args.candidate)
    comparable = compare_fields(baseline, candidate)
    print(f"comparability: {comparable['level']}")
    for diff in comparable.get("diffs") or []:
        print(f"- {diff['field']}: {diff['baseline']} -> {diff['candidate']}")
    for metric in ("output_tokens_per_sec", "e2e_p99_ms", "ttft_p99_ms", "tpot_p99_ms", "qps"):
        _print_metric_delta(baseline, candidate, metric)
    report = write_compare_report(baseline_dir, args.candidate, args.output_dir)
    print(f"compare_report: {report}")


def cmd_gate(args: argparse.Namespace) -> None:
    baseline_dir = _resolve_baseline(args)
    thresholds = RegressionThresholds(
        max_output_tps_drop_pct=args.max_output_tps_drop_pct,
        max_total_tps_drop_pct=args.max_total_tps_drop_pct,
        max_qps_drop_pct=args.max_qps_drop_pct,
        max_e2e_p99_increase_pct=args.max_e2e_p99_increase_pct,
        max_ttft_p99_increase_pct=args.max_ttft_p99_increase_pct,
        max_tpot_p99_increase_pct=args.max_tpot_p99_increase_pct,
        max_failed_requests=args.max_failed_requests,
        require_comparable=args.require_comparable,
    )
    result = evaluate_regression(load_manifest(baseline_dir), load_manifest(args.candidate), thresholds)
    output = args.output or args.candidate / "reports" / "gate_result.json"
    write_gate_result(output, result)
    print(f"gate_status: {result['status']}")
    print(f"gate_result: {output}")
    for violation in result["violations"]:
        print(f"- {violation['message']}")
    if result["status"] != "pass":
        raise SystemExit(1)


def _resolve_baseline(args: argparse.Namespace) -> Path:
    baseline_dir = args.baseline
    if args.to_baseline:
        baseline_dir = find_baseline_for_run(args.candidate, args.baselines_dir)
    if baseline_dir is None:
        raise ValueError("compare/gate requires --baseline or --to-baseline")
    return baseline_dir


def _print_metric_delta(baseline: dict[str, object], candidate: dict[str, object], metric: str) -> None:
    left = (baseline.get("summary") or {}).get(metric)
    right = (candidate.get("summary") or {}).get(metric)
    if left in (None, 0) or right is None:
        return
    pct = (float(right) - float(left)) / float(left) * 100.0
    print(f"{metric}: {left} -> {right} ({pct:+.1f}%)")
