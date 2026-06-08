from __future__ import annotations

from pathlib import Path

from llm_bench.textutil import slug  # re-exported for backwards compatibility

__all__ = ["split_csv", "first_not_none", "slug", "existing_report_path"]


def split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def first_not_none(*values: object) -> object:
    for value in values:
        if value is not None:
            return value
    return None


def existing_report_path(run_dir: Path) -> Path:
    report_path = run_dir / "reports" / "inference_report.md"
    if not report_path.exists():
        raise FileNotFoundError(report_path)
    return report_path
