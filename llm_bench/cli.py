from __future__ import annotations

import argparse
import sys

from llm_bench.commands.cleanup import register_cleanup
from llm_bench.commands.comm import register_comm
from llm_bench.commands.compare import register_compare
from llm_bench.commands.config import register_config
from llm_bench.commands.history import register_history
from llm_bench.commands.infer import register_check, register_infer, register_report, register_wizard
from llm_bench.commands.self_test import register_self_test


# Subcommands that consume a passthrough argv (everything after `--`).
PASSTHROUGH_COMMANDS = {"infer", "comm"}


def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]
    pre, post = _split_passthrough(argv)
    parser = build_parser()
    args = parser.parse_args(pre)
    args.passthrough = post
    if not hasattr(args, "func"):
        parser.print_help()
        return
    try:
        args.func(args)
    except KeyboardInterrupt:
        print("interrupted by user", file=sys.stderr)
        raise SystemExit(130) from None
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm-bench",
        description=(
            "Local benchmark CLI for LLM inference (vLLM / SGLang / transformers) and NCCL all-reduce.\n\n"
            "For vLLM / SGLang / NCCL it is a thin wrapper: the real serving / NCCL\n"
            "command is whatever you pass after `--`. The tool only starts the container,\n"
            "forwards the port, runs the workload client, and writes a Markdown report.\n"
            "For transformers it calls from_pretrained / generate in-process using the\n"
            "library's own parameter names."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", metavar="command")
    register_infer(sub)
    register_wizard(sub)
    register_check(sub)
    register_report(sub)
    register_history(sub)
    register_compare(sub)
    register_config(sub)
    register_comm(sub)
    register_cleanup(sub)
    register_self_test(sub)
    return parser


def _split_passthrough(argv: list[str]) -> tuple[list[str], list[str]]:
    """Split argv at the first standalone `--` token.

    Anything before `--` belongs to argparse (tool-level flags).
    Anything after `--` is the literal command the user wants to run inside
    the container (or invoked locally for comm tests).
    """
    if "--" not in argv:
        return list(argv), []
    idx = argv.index("--")
    return list(argv[:idx]), list(argv[idx + 1:])
