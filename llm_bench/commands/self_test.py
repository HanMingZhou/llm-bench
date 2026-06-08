from __future__ import annotations

import argparse

from llm_bench.archive import create_run_dir, write_run_archive
from llm_bench.backends.dry_run import DryRunBackend
from llm_bench.config import BenchConfig


def register_self_test(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = sub.add_parser(
        "self-test",
        help="Validate CLI archive/report generation without real inference.",
        description=(
            "Run a simulated dry-run benchmark and write a full archive (no Docker,\n"
            "no GPU, no model). Useful for verifying the CLI plumbing, workload\n"
            "loading, report generation, and cleanup."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--output-dir", default="benchmark_output/self_test_runs")
    parser.add_argument("--run-name", default="")
    parser.add_argument("--prompt-dir", default="")
    parser.add_argument("--prompt-jsonl", default="")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--total-requests", type=int, default=3)
    parser.add_argument("--include-samples", action="store_true")
    parser.set_defaults(func=cmd_self_test)


def cmd_self_test(args: argparse.Namespace) -> None:
    config = BenchConfig()
    config.backend.name = "dry-run"
    config.backend.model_name = "self-test"
    config.report.output_dir = args.output_dir
    config.report.run_name = args.run_name
    config.report.include_samples = args.include_samples
    config.workload.concurrency = [args.concurrency]
    config.workload.total_requests = args.total_requests
    config.workload.warmup_requests = 0
    config.workload.prompt_dir = args.prompt_dir
    config.workload.prompt_jsonl = args.prompt_jsonl
    if args.prompt_jsonl:
        config.workload.mode = "jsonl"
    elif args.prompt_dir:
        config.workload.mode = "prompt-dir"
    requested = {
        "command": "self-test",
        "workload": {
            "concurrency": config.workload.concurrency,
            "total_requests": config.workload.total_requests,
            "prompt_dir": config.workload.prompt_dir,
            "prompt_jsonl": config.workload.prompt_jsonl,
        },
        "report": {
            "output_dir": config.report.output_dir,
            "run_name": config.report.run_name,
            "include_samples": config.report.include_samples,
        },
    }
    run_dir = create_run_dir(config)
    result = DryRunBackend().run(config)
    manifest = write_run_archive(run_dir, config, requested, result, runtime={})
    print(f"self_test_run_id: {manifest['run_id']}")
    print(f"self_test_run_dir: {run_dir}")
    print(f"report: {run_dir / 'reports' / 'inference_report.md'}")
