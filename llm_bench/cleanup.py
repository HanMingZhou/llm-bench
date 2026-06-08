from __future__ import annotations

import shutil
import time
from dataclasses import dataclass
from pathlib import Path


# Files that are always preserved regardless of retention settings.
PROTECTED_FILES = {
    "run_manifest.json",
    "config.requested.yaml",
    "config.resolved.yaml",
    "environment.json",
    "metrics.summary.json",
}

PROTECTED_DIRS = {
    "reports",
}


@dataclass
class CleanupPlan:
    delete_files: list[Path]
    delete_dirs: list[Path]


def build_cleanup_plan(
    runs_dir: Path,
    request_metrics_days: int | None = None,
    gpu_metrics_days: int | None = None,
    logs_days: int | None = None,
    keep_summary_forever: bool = True,
    keep_output_samples: bool = True,
) -> CleanupPlan:
    now = time.time()
    files: list[Path] = []
    dirs: list[Path] = []
    for run_dir in sorted(runs_dir.glob("*")):
        if not run_dir.is_dir():
            continue
        if request_metrics_days is not None:
            path = run_dir / "metrics.requests.jsonl"
            if _older_than(path, request_metrics_days, now):
                files.append(path)
        if gpu_metrics_days is not None:
            path = run_dir / "metrics.gpu.jsonl"
            if _older_than(path, gpu_metrics_days, now):
                files.append(path)
        if logs_days is not None:
            logs = run_dir / "logs"
            if logs.exists() and _older_than(logs, logs_days, now):
                dirs.append(logs)
        # Clean up output samples if not keeping them.
        if not keep_output_samples:
            samples_path = run_dir / "samples.jsonl"
            if samples_path.exists():
                files.append(samples_path)
    # Filter out protected files/dirs if keep_summary_forever is True.
    if keep_summary_forever:
        files = [f for f in files if f.name not in PROTECTED_FILES]
        dirs = [d for d in dirs if d.name not in PROTECTED_DIRS]
    return CleanupPlan(files, dirs)


def execute_cleanup(plan: CleanupPlan, dry_run: bool = True) -> None:
    if dry_run:
        return
    for path in plan.delete_files:
        if path.exists():
            path.unlink()
    for path in plan.delete_dirs:
        if path.exists():
            shutil.rmtree(path)


def _older_than(path: Path, days: int, now: float) -> bool:
    if not path.exists():
        return False
    if days < 0:
        return False
    age_seconds = now - path.stat().st_mtime
    return age_seconds >= days * 86400
