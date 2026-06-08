from __future__ import annotations

import argparse
from pathlib import Path

from llm_bench.cleanup import build_cleanup_plan, execute_cleanup
from llm_bench.commands.common import first_not_none
from llm_bench.config import BenchConfig, load_config


def register_cleanup(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    cleanup = sub.add_parser("cleanup", help="Clean old heavy run artifacts.")
    cleanup.add_argument("--config", type=Path)
    cleanup.add_argument("--runs-dir", type=Path, default=Path("benchmark_output/runs"))
    cleanup.add_argument("--request-metrics-days", type=int, default=None)
    cleanup.add_argument("--gpu-metrics-days", type=int, default=None)
    cleanup.add_argument("--logs-days", type=int, default=None)
    cleanup.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=True)
    cleanup.set_defaults(func=cmd_cleanup)


def cmd_cleanup(args: argparse.Namespace) -> None:
    config = load_config(args.config) if args.config else BenchConfig()
    plan = build_cleanup_plan(
        args.runs_dir,
        request_metrics_days=first_not_none(args.request_metrics_days, config.retention.request_metrics_days),
        gpu_metrics_days=first_not_none(args.gpu_metrics_days, config.retention.gpu_metrics_days),
        logs_days=first_not_none(args.logs_days, config.retention.logs_days),
    )
    for path in plan.delete_files:
        print(f"delete_file: {path}")
    for path in plan.delete_dirs:
        print(f"delete_dir: {path}")
    execute_cleanup(plan, dry_run=args.dry_run)
    print(f"dry_run: {args.dry_run}")
