from __future__ import annotations

import json
import re
import shlex
import shutil
import socket
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from llm_bench.charts import render_nccl_charts
from llm_bench.environment import inspect_gpu
from llm_bench.gpu import query_gpu_topology


@dataclass
class NcclConfig:
    """Thin NCCL test wrapper.

    The command actually executed inside the container is whatever the user
    passes via `command` (i.e. the argv after `--` on the CLI). The tool does
    not translate or rename NCCL parameters.
    """

    image: str = "nccl-tests:latest"
    command: list[str] = field(default_factory=list)
    output_dir: str = "benchmark_output/comm_runs"
    run_name: str = ""
    docker_args: list[str] = field(default_factory=list)
    timeout_seconds: int = 1800
    dry_run: bool = False


def run_nccl_all_reduce(config: NcclConfig) -> Path:
    if not config.command:
        raise ValueError(
            "missing NCCL command. Append it after `--`, for example:\n"
            "  llm-bench comm all-reduce --image nccl-tests:latest --docker-arg=--gpus=all -- "
            "/opt/nccl-tests/build/all_reduce_perf -b 8 -e 1G -f 2 -g 8 -n 100 -w 20"
        )
    run_dir = _create_run_dir(config)
    logs_dir = run_dir / "logs"
    reports_dir = run_dir / "reports"
    logs_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    started = datetime.now().astimezone()
    container_name = f"llm-bench-nccl-{int(started.timestamp())}"
    cmd = _docker_cmd(config, container_name)
    plan_script = _write_launch_plan(run_dir, cmd)
    print("docker command:", flush=True)
    print(shlex.join(cmd), flush=True)
    if config.dry_run:
        proc: Any = _DryProc()
    else:
        try:
            proc = subprocess.run(cmd, text=True, capture_output=True, timeout=config.timeout_seconds)
        except subprocess.TimeoutExpired as exc:
            # The docker client was killed, but the named container may still be
            # running; force-remove it so it doesn't leak, then record a failure.
            _force_remove_container(container_name)
            proc = _TimeoutProc(exc, config.timeout_seconds)
    ended = datetime.now().astimezone()
    stdout = proc.stdout
    stderr = proc.stderr
    (logs_dir / "nccl.stdout.log").write_text(stdout, encoding="utf-8")
    (logs_dir / "nccl.stderr.log").write_text(stderr, encoding="utf-8")
    rows = parse_all_reduce_output(stdout)
    summary = summarize_nccl(rows, proc.returncode)
    image_paths = render_nccl_charts(rows, reports_dir / "images")
    # P1: snapshot the GPU topology so the report can quote the theoretical
    # bandwidth ceiling (NVLink / PCIe class) alongside the measured busbw.
    gpu_topology = query_gpu_topology() or {}
    gpu_info = inspect_gpu()
    host_info = _query_host_info()
    gpu_count = _extract_g_flag(config.command)
    manifest = {
        "run_id": run_dir.name,
        "created_at": started.isoformat(timespec="seconds"),
        "ended_at": ended.isoformat(timespec="seconds"),
        "kind": "nccl-all-reduce",
        "docker_command": cmd,
        "container_command": list(config.command),
        "plan_script": str(plan_script),
        "config": asdict(config),
        "returncode": proc.returncode,
        "summary": summary,
        "gpu_count": gpu_count,
        "gpu_topology": gpu_topology,
        "gpu_info": gpu_info,
        "host_info": host_info,
        "stdout_log": str(logs_dir / "nccl.stdout.log"),
        "stderr_log": str(logs_dir / "nccl.stderr.log"),
    }
    _write_json(run_dir / "run_manifest.json", manifest)
    _write_json(run_dir / "nccl.summary.json", summary)
    _write_jsonl(run_dir / "nccl.results.jsonl", rows)
    (reports_dir / "nccl_report.md").write_text(
        render_nccl_report(manifest, rows, image_paths),
        encoding="utf-8",
    )
    return run_dir


def _query_host_info() -> dict[str, Any]:
    """Collect hostname, OS, CPU, memory, NVIDIA driver & CUDA version."""
    info: dict[str, Any] = {}
    try:
        info["hostname"] = socket.gethostname()
    except Exception:
        pass
    # OS info
    try:
        proc = subprocess.run(
            ["uname", "-sr"], text=True, capture_output=True, timeout=5,
        )
        if proc.returncode == 0:
            info["os"] = proc.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    # CPU model
    try:
        proc = subprocess.run(
            ["lscpu"], text=True, capture_output=True, timeout=5,
        )
        if proc.returncode == 0:
            for line in proc.stdout.splitlines():
                if line.startswith("Model name:") or line.startswith("型号名称："):
                    info["cpu_model"] = line.split(":", 1)[-1].strip()
                elif line.startswith("CPU(s):"):
                    info["cpu_cores"] = line.split(":", 1)[-1].strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    # Total memory
    try:
        proc = subprocess.run(
            ["free", "-g"], text=True, capture_output=True, timeout=5,
        )
        if proc.returncode == 0:
            for line in proc.stdout.splitlines():
                if line.startswith("Mem:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        info["memory_total_gb"] = parts[1]
                    break
    except (OSError, subprocess.TimeoutExpired):
        pass
    # NVIDIA driver & CUDA version
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        try:
            proc = subprocess.run(
                [nvidia_smi, "--query-gpu=driver_version", "--format=csv,noheader,nounits"],
                text=True, capture_output=True, timeout=10,
            )
            if proc.returncode == 0:
                info["driver_version"] = proc.stdout.strip().splitlines()[0].strip()
        except (OSError, subprocess.TimeoutExpired):
            pass
        try:
            proc = subprocess.run(
                [nvidia_smi], text=True, capture_output=True, timeout=10,
            )
            if proc.returncode == 0:
                for line in proc.stdout.splitlines():
                    if "CUDA Version:" in line:
                        match = re.search(r"CUDA Version:\s*([\d.]+)", line)
                        if match:
                            info["cuda_version"] = match.group(1)
                        break
        except (OSError, subprocess.TimeoutExpired):
            pass
    return info


def _extract_g_flag(argv: list[str]) -> int:
    """all_reduce_perf -g N tells nccl-tests how many GPUs per process to use."""
    for i, tok in enumerate(argv):
        if tok == "-g" and i + 1 < len(argv):
            try:
                return int(argv[i + 1])
            except (TypeError, ValueError):
                return 0
        if tok.startswith("-g") and len(tok) > 2 and tok[2:].isdigit():
            return int(tok[2:])
    return 0


def _docker_cmd(config: NcclConfig, container_name: str) -> list[str]:
    cmd: list[str] = ["docker", "run", "--rm", "--name", container_name]
    cmd.extend(config.docker_args)
    cmd.append(config.image)
    cmd.extend(config.command)
    return cmd


def _force_remove_container(container_name: str) -> None:
    try:
        subprocess.run(["docker", "rm", "-f", container_name], text=True, capture_output=True, timeout=20)
    except (subprocess.TimeoutExpired, OSError):
        pass


def parse_all_reduce_output(output: str) -> list[dict[str, Any]]:
    rows = []
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) < 12:
            continue
        if not re.match(r"^\d+$", parts[0]):
            continue
        try:
            rows.append(
                {
                    "size_bytes": int(parts[0]),
                    "count": int(parts[1]),
                    "type": parts[2],
                    "redop": parts[3],
                    "root": int(parts[4]),
                    "time_us": float(parts[5]),
                    "algbw_gbps": float(parts[6]),
                    "busbw_gbps": float(parts[7]),
                    "error": float(parts[8]),
                    "time_us_out": float(parts[9]),
                    "algbw_gbps_out": float(parts[10]),
                    "busbw_gbps_out": float(parts[11]),
                    "error_out": float(parts[12]) if len(parts) > 12 and _is_number(parts[12]) else None,
                }
            )
        except (ValueError, IndexError):
            continue
    return rows


def summarize_nccl(rows: list[dict[str, Any]], returncode: int) -> dict[str, Any]:
    if not rows:
        return {
            "success": returncode == 0,
            "rows": 0,
            "max_busbw_gbps": 0.0,
            "max_busbw_inplace_gbps": 0.0,
            "max_algbw_gbps": 0.0,
            "max_algbw_inplace_gbps": 0.0,
            "min_time_us": 0.0,
            "max_error": 0.0,
            "max_error_inplace": 0.0,
            "non_zero_error_rows": 0,
        }
    # Errors: nccl-tests reports per-row numerical correctness drift; non-zero
    # values flag the implementation deviated from the reference.
    errors_oop = [float(row.get("error") or 0) for row in rows]
    errors_ip = [float(row.get("error_out") or 0) for row in rows]
    non_zero_errors = sum(1 for e in errors_oop + errors_ip if e > 0)
    return {
        "success": returncode == 0,
        "rows": len(rows),
        "max_busbw_gbps": max(float(row["busbw_gbps"]) for row in rows),
        "max_busbw_inplace_gbps": max(float(row.get("busbw_gbps_out") or 0) for row in rows),
        "max_algbw_gbps": max(float(row["algbw_gbps"]) for row in rows),
        "max_algbw_inplace_gbps": max(float(row.get("algbw_gbps_out") or 0) for row in rows),
        "min_time_us": min(float(row["time_us"]) for row in rows),
        "largest_size_bytes": max(int(row["size_bytes"]) for row in rows),
        "largest_size_busbw_gbps": _largest_size_metric(rows, "busbw_gbps"),
        "largest_size_busbw_inplace_gbps": _largest_size_metric(rows, "busbw_gbps_out"),
        "max_error": max(errors_oop) if errors_oop else 0.0,
        "max_error_inplace": max(errors_ip) if errors_ip else 0.0,
        "non_zero_error_rows": non_zero_errors,
    }


def render_nccl_report(manifest: dict[str, Any], rows: list[dict[str, Any]], image_paths: list[Path] | None = None) -> str:
    summary = manifest["summary"]
    # P0: full result table now exposes in-place columns + the numerical
    # error column highlighted when non-zero (correctness regression flag).
    table_rows = "\n".join(_format_nccl_row(row) for row in rows[:40]) or (
        "| - | - | - | - | - | - | - | - | - |"
    )
    images = "\n\n".join(f"![{path.stem}](images/{path.name})" for path in (image_paths or []))
    diagnostics = _render_diagnostics(manifest)
    container_cmd = shlex.join(manifest.get("container_command") or [])
    # P1: GPU topology + theoretical bandwidth ceiling. Lets the user judge
    # whether the measured busbw is "good" relative to physical limits.
    topology_section = _render_topology_section(manifest)
    # P0: human-readable largest tested size (256MB instead of 268435456).
    largest_size_human = _format_bytes(summary.get("largest_size_bytes") or 0)
    err_warning = ""
    if int(summary.get("non_zero_error_rows") or 0) > 0:
        err_warning = (
            f"\n> ⚠ {summary['non_zero_error_rows']} 行存在非零数值误差 "
            f"(max={summary.get('max_error')} / max_inplace={summary.get('max_error_inplace')})，"
            "可能是 NCCL 实现 bug 或硬件问题。\n"
        )
    return f"""# NCCL All-Reduce 测试报告

> run_id: `{manifest["run_id"]}` · image: `{manifest["config"]["image"]}` · returncode: `{manifest.get("returncode")}`

## 关键指标

| metric | value | unit |
|---|---:|---|
| **max busbw (out-of-place)** | `{summary["max_busbw_gbps"]}` | GB/s |
| **max busbw (in-place)** | `{summary.get("max_busbw_inplace_gbps", 0)}` | GB/s |
| max algbw (out-of-place) | `{summary["max_algbw_gbps"]}` | GB/s |
| max algbw (in-place) | `{summary.get("max_algbw_inplace_gbps", 0)}` | GB/s |
| min time | `{summary["min_time_us"]}` | µs |
| largest tested size | `{largest_size_human}` (`{summary.get("largest_size_bytes")}` bytes) | - |
| busbw @ largest size (out-of-place) | `{summary.get("largest_size_busbw_gbps")}` | GB/s |
| busbw @ largest size (in-place) | `{summary.get("largest_size_busbw_inplace_gbps", 0)}` | GB/s |
| 非零误差行数 | `{summary.get("non_zero_error_rows", 0)}` | rows |
{err_warning}
## 启动命令

```bash
{container_cmd}
```

{topology_section}

## 指标含义

- **algbw (algorithm bandwidth)** = 传输字节 / 实测时间。反映**每张卡**看到的数据量。
- **busbw (bus bandwidth)** = `algbw × 2(N−1)/N`（all-reduce ring）。反映 NCCL **实际触达的硬件带宽**——这是判断 NCCL 实现好不好的核心指标。
- **out-of-place / in-place**: nccl-tests 同时测两种模式。in-place 复用输入 buffer 通常更快；二者差距过大说明实现有抖动。
- **延迟带宽拐点**: 小消息时延为主、大消息带宽为主。看 busbw 在哪个 size 接近其 max 值（一般 ≥ 64MB），那里就是带宽起作用的拐点。
- **error**: nccl-tests 对每个 size 做正确性校验，非零代表数值漂移。

{images}

{diagnostics}

## 结果表 (前 40 行)

> 单位：size = bytes（人类可读列见 size_h）；time = µs；algbw / busbw = GB/s；err = 数值误差（0 = 正确）

| size_h | size | time(oop) | algbw(oop) | busbw(oop) | err(oop) | time(ip) | algbw(ip) | busbw(ip) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{table_rows}

完整输出见 `logs/nccl.stdout.log` 和 `nccl.results.jsonl`。执行计划见 `{Path(manifest["plan_script"]).name}`。
"""


def _format_nccl_row(row: dict[str, Any]) -> str:
    """Render one result row with size in human units + in-place columns + error."""
    size_bytes = int(row.get("size_bytes") or 0)
    err_oop = float(row.get("error") or 0)
    err_marker = f"**{err_oop}**" if err_oop > 0 else f"{err_oop}"
    return (
        f"| `{_format_bytes(size_bytes)}` | {size_bytes} "
        f"| {row.get('time_us')} | {row.get('algbw_gbps')} | {row.get('busbw_gbps')} "
        f"| {err_marker} "
        f"| {row.get('time_us_out', '-')} | {row.get('algbw_gbps_out', '-')} | {row.get('busbw_gbps_out', '-')} |"
    )


def _format_bytes(num: int) -> str:
    """1024 -> '1KB', 268435456 -> '256MB', 1073741824 -> '1GB'."""
    if num <= 0:
        return "0B"
    units = ("B", "KB", "MB", "GB", "TB")
    n = float(num)
    idx = 0
    while n >= 1024 and idx < len(units) - 1:
        n /= 1024
        idx += 1
    if n == int(n):
        return f"{int(n)}{units[idx]}"
    return f"{n:.1f}{units[idx]}"


def _render_topology_section(manifest: dict[str, Any]) -> str:
    """P1: GPU count + topology + theoretical bandwidth ceiling.

    The topology is cached by query_gpu_topology so this is cheap. Theoretical
    ceilings come from the strongest link in the topology (NVLink > PCIe).
    """
    topo = manifest.get("gpu_topology") or {}
    gpu_info = manifest.get("gpu_info") or {}
    host_info = manifest.get("host_info") or {}
    gpu_count = manifest.get("gpu_count") or 0

    # GPU details table
    gpus = gpu_info.get("gpus") or []
    gpu_detail_rows = ""
    if gpus:
        for g in gpus:
            gpu_detail_rows += (
                f"| GPU [{g.get('index', '?')}] | `{g.get('name', 'unknown')}` "
                f"| {g.get('memory_total_mb', '?')} MiB |\n"
            )
    else:
        gpu_detail_rows = "| GPU | 未检测到 | - |\n"

    # Host info rows
    host_rows = ""
    if host_info.get("hostname"):
        host_rows += f"| 主机名 | `{host_info['hostname']}` | - |\n"
    if host_info.get("os"):
        host_rows += f"| 操作系统 | `{host_info['os']}` | - |\n"
    if host_info.get("cpu_model"):
        host_rows += f"| CPU | `{host_info['cpu_model']}` | {host_info.get('cpu_cores', '?')} cores |\n"
    if host_info.get("memory_total_gb"):
        host_rows += f"| 内存 | `{host_info['memory_total_gb']}` GB | - |\n"
    if host_info.get("driver_version"):
        host_rows += f"| NVIDIA 驱动 | `{host_info['driver_version']}` | - |\n"
    if host_info.get("cuda_version"):
        host_rows += f"| CUDA 版本 | `{host_info['cuda_version']}` | - |\n"

    # Topology rows
    topo_rows = ""
    if topo.get("available"):
        links = topo.get("links") or []
        interconnect = ", ".join(str(link) for link in links) if links else "unknown"
        topo_rows += (
            f"| GPU 间最强链路 | `{interconnect}` "
            f"| NV* = NVLink；PIX/PXB = 同 root 的 PCIe；PHB = 跨 root 的 PCIe；SYS = 跨 NUMA |\n"
        )
        ceiling = _theoretical_bandwidth(links)
        if ceiling:
            topo_rows += (
                f"| 理论 busbw 上限 (参考) | `{ceiling['gbps']}` GB/s ({ceiling['source']}) "
                f"| 实测 max 应接近此值；远低于说明软件 / 拓扑 / 配置有瓶颈 |\n"
            )
        else:
            topo_rows += (
                "| 理论 busbw 上限 (参考) | 无法估算 "
                "| 拓扑包含 SYS (跨 NUMA / PCIe root)，带宽受系统总线限制 |\n"
            )

    return (
        "## 拓扑 / 环境\n\n"
        "| field | value | 说明 |\n|---|---|---|\n"
        f"| GPU 数 (测试 -g) | `{gpu_count}` | 容器内可见 |\n"
        f"{gpu_detail_rows}"
        f"{host_rows}"
        f"{topo_rows}"
    )


# Bandwidth ceilings indexed by NCCL topology link type (nvidia-smi topo -m
# vocabulary). Numbers are uni-directional theoretical maxes; busbw should
# get within 70-90% of these in healthy runs.
_BANDWIDTH_REFERENCE = [
    # (token, gbps, source)
    ("NV12", 600.0, "NVLink 4 (H100 x 18 lanes)"),
    ("NV8", 300.0, "NVLink 3 (A100 SXM)"),
    ("NV6", 200.0, "NVLink 3 partial"),
    ("NV4", 200.0, "NVLink 2 (V100)"),
    ("NV2", 100.0, "NVLink partial"),
    ("NV1", 50.0, "NVLink single lane"),
    ("PIX", 32.0, "PCIe 4 x16 (sibling under same root)"),
    ("PXB", 32.0, "PCIe 4 x16 (multi-switch)"),
    ("PHB", 16.0, "PCIe across host bridges"),
]


def _theoretical_bandwidth(links: list) -> dict[str, Any] | None:
    """Pick the strongest link in the topology and return its rated bandwidth."""
    token_set = {str(link).upper() for link in links}
    for token, gbps, source in _BANDWIDTH_REFERENCE:
        if token in token_set:
            return {"token": token, "gbps": gbps, "source": source}
    return None


def _render_diagnostics(manifest: dict[str, Any]) -> str:
    stderr_log = Path(manifest.get("stderr_log") or "")
    returncode = manifest.get("returncode")
    stderr_tail = _tail_file(stderr_log)
    if returncode == 0 and not stderr_tail:
        return ""
    lines = ["## 错误与诊断", ""]
    if returncode != 0:
        lines.append(f"- returncode: `{returncode}`")
    if stderr_tail:
        lines.extend(["", "### stderr 摘要", "", "```text", stderr_tail, "```"])
    return "\n".join(lines)


def _create_run_dir(config: NcclConfig) -> Path:
    created = datetime.now().astimezone()
    run_name = config.run_name or f"{created:%Y%m%d_%H%M%S}_nccl_all_reduce"
    parent = Path(config.output_dir)
    candidate = parent / run_name
    suffix = 1
    while True:
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            return candidate
        except FileExistsError:
            candidate = parent / f"{run_name}_{suffix}"
            suffix += 1


def _write_launch_plan(run_dir: Path, cmd: list[str]) -> Path:
    path = run_dir / "launch_plan.sh"
    path.write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + shlex.join(cmd) + "\n", encoding="utf-8")
    path.chmod(0o755)
    return path


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _largest_size_metric(rows: list[dict[str, Any]], metric: str) -> float:
    largest = max(int(row["size_bytes"]) for row in rows)
    candidates = [row for row in rows if int(row["size_bytes"]) == largest]
    return float(candidates[-1][metric]) if candidates else 0.0


def _is_number(value: str) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False


def _tail_file(path: Path, max_chars: int = 2000) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")[-max_chars:].strip()


class _DryProc:
    stdout = ""
    stderr = "dry-run: NCCL command was not executed.\n"
    returncode = 0


class _TimeoutProc:
    """Stand-in for a CompletedProcess when the NCCL run times out.

    Keeps partial output (if any) and reports the conventional 124 exit code so
    the run is archived as a failure instead of crashing the CLI.
    """

    def __init__(self, exc: subprocess.TimeoutExpired, timeout_seconds: int) -> None:
        self.stdout = _as_text(exc.stdout)
        self.stderr = _as_text(exc.stderr) + f"\nnccl command timed out after {timeout_seconds}s.\n"
        self.returncode = 124


def _as_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)
