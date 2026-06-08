from __future__ import annotations

import argparse
import json
from pathlib import Path

from llm_bench.commands.common import slug
from llm_bench.compare import load_manifest


def register_history(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    list_cmd = sub.add_parser("list", help="List historical runs.")
    list_cmd.add_argument("--runs-dir", type=Path, default=Path("benchmark_output/runs"))
    list_cmd.set_defaults(func=cmd_list)

    show = sub.add_parser("show", help="Show one run manifest.")
    show.add_argument("run_dir", type=Path)
    show.set_defaults(func=cmd_show)

    baseline = sub.add_parser("baseline", help="Manage baseline run indexes.")
    baseline_sub = baseline.add_subparsers(dest="baseline_command", metavar="command")
    baseline.set_defaults(func=lambda _args: baseline.print_help())
    baseline_set = baseline_sub.add_parser("set", help="Set one run as a baseline.")
    baseline_set.add_argument("run_dir", type=Path)
    baseline_set.add_argument("--baselines-dir", type=Path, default=Path("baselines"))
    baseline_set.set_defaults(func=cmd_baseline_set)
    baseline_list = baseline_sub.add_parser("list", help="List baselines.")
    baseline_list.add_argument("--baselines-dir", type=Path, default=Path("baselines"))
    baseline_list.set_defaults(func=cmd_baseline_list)


def cmd_list(args: argparse.Namespace) -> None:
    for run in _load_runs(args.runs_dir):
        backend = (run.get("backend") or {}).get("name")
        model = (run.get("model") or {}).get("name") or (run.get("model") or {}).get("path")
        summary = run.get("summary") or {}
        print(f"{run.get('run_id')}\t{model}\t{backend}\tP99={summary.get('e2e_p99_ms')}")


def cmd_show(args: argparse.Namespace) -> None:
    manifest = args.run_dir / "run_manifest.json"
    if not manifest.exists():
        raise FileNotFoundError(manifest)
    print(manifest.read_text(encoding="utf-8"))


def cmd_baseline_set(args: argparse.Namespace) -> None:
    manifest = load_manifest(args.run_dir)
    model = slug((manifest.get("model") or {}).get("name") or (manifest.get("model") or {}).get("path") or "unknown-model")
    hardware = slug((manifest.get("hardware") or {}).get("gpu_model") or "unknown-hardware")
    backend = slug((manifest.get("backend") or {}).get("name") or "unknown-backend")
    target = args.baselines_dir / model / hardware / f"{backend}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_dir": str(args.run_dir),
        "run_id": manifest.get("run_id"),
        "model": manifest.get("model"),
        "hardware": manifest.get("hardware"),
        "backend": manifest.get("backend"),
        "summary": manifest.get("summary"),
        "tags": manifest.get("tags"),
    }
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(target)


def cmd_baseline_list(args: argparse.Namespace) -> None:
    for path in sorted(args.baselines_dir.glob("**/*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        print(f"{path}\t{data.get('run_id')}\t{data.get('run_dir')}")


def _load_runs(runs_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if not runs_dir.exists():
        return rows
    for manifest_path in sorted(runs_dir.glob("*/run_manifest.json")):
        try:
            rows.append(json.loads(manifest_path.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            continue
    return rows
