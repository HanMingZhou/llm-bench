from __future__ import annotations

import argparse
from pathlib import Path

from llm_bench.config import BenchConfig
from llm_bench.interactive import run_infer_wizard
from llm_bench.yaml_io import dump_yaml


def register_config(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    config = sub.add_parser("config", help="Manage benchmark config files.")
    config_sub = config.add_subparsers(dest="config_command", metavar="command")
    init = config_sub.add_parser("init", help="Write a default YAML config (or interactively).")
    init.add_argument("--output", type=Path, default=Path("configs/inference.yaml"))
    init.add_argument("-i", "--interactive", action="store_true")
    init.set_defaults(func=cmd_config_init)


def cmd_config_init(args: argparse.Namespace) -> None:
    config = BenchConfig()
    if args.interactive:
        config, _, _ = run_infer_wizard(config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dump_yaml(args.output, config.to_dict())
    print(args.output)
