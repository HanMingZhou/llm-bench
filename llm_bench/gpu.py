from __future__ import annotations

import csv
import shutil
import subprocess
import threading
import time
from functools import lru_cache
from io import StringIO
from typing import Any


class GpuSampler:
    def __init__(self, interval_seconds: float = 1.0) -> None:
        self.interval_seconds = interval_seconds
        self.rows: list[dict[str, object]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if shutil.which("nvidia-smi") is None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> list[dict[str, object]]:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        return self.rows

    def _run(self) -> None:
        while not self._stop.is_set():
            self.rows.extend(query_gpu_metrics())
            self._stop.wait(self.interval_seconds)


def query_gpu_metrics() -> list[dict[str, object]]:
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return []
    fields = [
        "timestamp",
        "index",
        "name",
        "utilization.gpu",
        "memory.used",
        "memory.total",
        "temperature.gpu",
        "power.draw",
    ]
    cmd = [
        nvidia_smi,
        f"--query-gpu={','.join(fields)}",
        "--format=csv,noheader,nounits",
    ]
    try:
        proc = subprocess.run(cmd, text=True, capture_output=True, timeout=10)
    except (subprocess.TimeoutExpired, OSError):
        return []
    if proc.returncode != 0:
        return []
    rows = []
    reader = csv.reader(StringIO(proc.stdout))
    # Alias nvidia-smi's dotted query names to friendly snake_case keys so
    # downstream code (metrics aggregation, report renderer) can read them
    # without ambiguity. We keep the dotted keys too for backward compat with
    # already-archived metrics.gpu.jsonl files.
    aliases = {
        "utilization.gpu": "utilization",
        "memory.used": "memory_used_mb",
        "memory.total": "memory_total_mb",
        "temperature.gpu": "temperature",
        "power.draw": "power_w",
    }
    for row in reader:
        if len(row) != len(fields):
            continue
        data = {field: _coerce(value.strip()) for field, value in zip(fields, row)}
        for src, dst in aliases.items():
            if src in data:
                data[dst] = data[src]
        data["sample_time_unix"] = time.time()
        rows.append(data)
    return rows


@lru_cache(maxsize=1)
def query_gpu_topology() -> dict[str, object]:
    # Topology does not change within a process; cache it so a single run does
    # not invoke `nvidia-smi topo -m` repeatedly (archive + environment both ask).
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return {"available": False, "error": "nvidia-smi command not found", "raw": ""}
    try:
        proc = subprocess.run([nvidia_smi, "topo", "-m"], text=True, capture_output=True, timeout=10)
    except (subprocess.TimeoutExpired, OSError) as exc:
        return {"available": False, "error": str(exc), "raw": ""}
    if proc.returncode != 0:
        return {"available": False, "error": proc.stderr.strip() or proc.stdout.strip(), "raw": proc.stdout}
    raw = proc.stdout
    links = []
    for token in ("NV", "PIX", "PXB", "PHB", "SYS"):
        if token in raw:
            links.append(token)
    return {"available": True, "links": links, "raw": raw}


def _coerce(value: str) -> Any:
    try:
        if "." in value:
            return float(value)
        return int(value)
    except ValueError:
        return value
