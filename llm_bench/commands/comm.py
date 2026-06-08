from __future__ import annotations

import argparse

from llm_bench.comm import NcclConfig, run_nccl_all_reduce
from llm_bench.environment import discover_docker_images
from llm_bench.interactive import run_comm_wizard


DEFAULT_NCCL_IMAGE = "nccl-tests:latest"


def register_comm(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    comm = sub.add_parser(
        "comm",
        help="Run communication benchmarks.",
        epilog="Use 'llm-bench wizard' for interactive setup.",
    )
    comm_sub = comm.add_subparsers(dest="comm_command", metavar="command")
    comm.set_defaults(func=lambda _args: comm.print_help())
    all_reduce = comm_sub.add_parser(
        "all-reduce",
        help="Run NCCL all_reduce_perf in Docker.",
        description=(
            "Run nccl-tests `all_reduce_perf` inside a container. The command itself\n"
            "is whatever you put after `--`. Flags starting with `--` must use the\n"
            "`--docker-arg=...` form so argparse does not swallow them, for example:\n"
            "  llm-bench comm all-reduce --image nccl-tests:latest \\\n"
            "    --docker-arg=--gpus=all --docker-arg=--shm-size=16g --docker-arg=--ipc=host -- \\\n"
            "    /opt/nccl-tests/build/all_reduce_perf -b 8 -e 1G -f 2 -g 8 -n 100 -w 20"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    all_reduce.add_argument("-i", "--interactive", action="store_true")
    all_reduce.add_argument("--image", default="")
    all_reduce.add_argument("--output-dir", default="benchmark_output/comm_runs")
    all_reduce.add_argument("--run-name", default="")
    all_reduce.add_argument("--timeout", type=int, default=1800)
    all_reduce.add_argument(
        "--docker-arg",
        action="append",
        default=[],
        help="Extra docker run arg, repeatable. Use --docker-arg=--shm-size=16g for flags starting with --.",
    )
    all_reduce.add_argument("--dry-run", action="store_true")
    all_reduce.set_defaults(func=cmd_comm_all_reduce)


def cmd_comm_all_reduce(args: argparse.Namespace) -> None:
    if args.interactive:
        args = run_comm_wizard(args)
    image = _resolve_image(args.image)
    command = list(getattr(args, "passthrough", None) or [])
    if not command:
        raise RuntimeError(
            "Missing NCCL command. Append it after `--`, for example:\n"
            "  llm-bench comm all-reduce --image nccl-tests:latest --docker-arg=--gpus=all -- "
            "/opt/nccl-tests/build/all_reduce_perf -b 8 -e 1G -f 2 -g 8 -n 100 -w 20"
        )
    config = NcclConfig(
        image=image,
        command=command,
        output_dir=args.output_dir,
        run_name=args.run_name,
        docker_args=list(args.docker_arg or []),
        timeout_seconds=args.timeout,
        dry_run=args.dry_run,
    )
    run_dir = run_nccl_all_reduce(config)
    print(f"comm_run_dir: {run_dir}")
    print(f"report: {run_dir / 'reports' / 'nccl_report.md'}")


def _resolve_image(image: str) -> str:
    if image:
        return image
    discovered = discover_docker_images("nccl")
    if discovered:
        return discovered[0]["name"]
    return DEFAULT_NCCL_IMAGE
