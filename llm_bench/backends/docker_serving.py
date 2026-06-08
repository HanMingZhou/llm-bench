from __future__ import annotations

import subprocess
import time
from pathlib import Path

from llm_bench.backends.base import BackendResult
from llm_bench.config import BenchConfig
from llm_bench.errors import classify_error
from llm_bench.gpu import GpuSampler
from llm_bench.http_client import HttpBenchTarget, run_openai_http_benchmark, smoke_ping_server, wait_for_openai_server


class DockerServingBackend:
    """A thin wrapper around `docker run <image> <user-provided argv>`.

    The user passes the real serving framework command (e.g. `vllm serve ...`
    or `python -m sglang.launch_server ...`) via CLI `--` or via the YAML
    `command:` field. This backend does not translate, validate, or reorder
    those arguments. It only injects the docker-level concerns the tool owns:
    container name, --rm, port forwarding, HF cache mount, HF_TOKEN, and any
    extra docker arguments the user supplied through --docker-arg.
    """

    def __init__(self, name: str) -> None:
        if name not in {"vllm", "sglang"}:
            raise ValueError(f"Unsupported docker serving backend: {name}")
        self.name = name

    def preview_command(self, config: BenchConfig) -> list[str]:
        return self._docker_run_command(config, f"llm-bench-{self.name}-preview")

    def run(self, config: BenchConfig) -> BackendResult:
        if not config.backend.command:
            raise ValueError(
                "missing container command. Pass it after `--` on the CLI, "
                "or set `backend.command:` in your YAML."
            )
        if not config.backend.model_name:
            raise ValueError("missing --model-name. It is required as the OpenAI API `model` field.")

        stamp = int(time.time())
        output_dir = Path(config.report.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        # Merge stdout + stderr into a single log file. vllm / sglang / uvicorn
        # write INFO via stderr, so a separate "stderr.log" misleads users into
        # thinking the run errored when it's just normal startup output.
        log_path = output_dir / f".active_backend_{self.name}_{stamp}.log"
        config.backend.stdout_log = str(log_path)
        config.backend.stderr_log = ""
        log_file = log_path.open("w", encoding="utf-8")
        container_name = f"llm-bench-{self.name}-{stamp}"
        cmd = self._docker_run_command(config, container_name)
        config.backend.launch_command = cmd
        started = time.perf_counter()
        try:
            proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, text=True)
            print(f"[setup] container started | name={container_name}", flush=True)
            print(f"[setup] container log | path={log_path}", flush=True)
        except OSError as exc:
            log_file.close()
            return BackendResult(
                backend=self.name,
                request_metrics=[],
                startup_seconds=round(time.perf_counter() - started, 3),
                errors=[f"{classify_error(exc)}: {exc}"],
                gpu_metrics=[],
            )

        sampler = GpuSampler()
        sampler.start()
        errors: list[str] = []
        gpu_metrics: list[dict[str, object]] = []
        wait_state = _WaitState()
        try:
            base_url = f"http://127.0.0.1:{config.backend.port}"
            if not wait_for_openai_server(
                base_url,
                config.backend.startup_timeout_seconds,
                on_wait=lambda elapsed: _print_wait_status(elapsed, log_path, proc, wait_state),
            ):
                errors.append(f"server health check timed out after {config.backend.startup_timeout_seconds}s: {base_url}")
                log_tail = _tail_file(log_path)
                if log_tail:
                    errors.append(f"{classify_error(log_tail)}: {log_tail}")
                gpu_metrics = sampler.stop()
                return BackendResult(
                    backend=self.name,
                    request_metrics=[],
                    startup_seconds=round(time.perf_counter() - started, 3),
                    errors=errors,
                    gpu_metrics=gpu_metrics,
                )
            startup_seconds = round(time.perf_counter() - started, 3)
            target = HttpBenchTarget(
                url=base_url,
                model=config.backend.model_name,
                backend=self.name,
            )
            smoke_error = _smoke_ping_with_retry(target, config.workload.api)
            if smoke_error:
                errors.append(smoke_error)
                gpu_metrics = sampler.stop()
                return BackendResult(
                    backend=self.name,
                    request_metrics=[],
                    startup_seconds=startup_seconds,
                    errors=errors,
                    gpu_metrics=gpu_metrics,
                )
            request_metrics = run_openai_http_benchmark(config, target)
            gpu_metrics = sampler.stop()
            return BackendResult(
                backend=self.name,
                request_metrics=request_metrics,
                startup_seconds=startup_seconds,
                errors=errors,
                gpu_metrics=gpu_metrics,
            )
        finally:
            if not gpu_metrics:
                sampler.stop()
            exit_code = proc.poll()
            if exit_code is not None and exit_code != 0:
                errors.append(f"container exited with code {exit_code}")
            if not config.backend.keep_container:
                _stop_container(container_name)
            if proc.poll() is None and not config.backend.keep_container:
                proc.terminate()
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    proc.kill()
            log_file.close()

    def _docker_run_command(self, config: BenchConfig, container_name: str) -> list[str]:
        cmd: list[str] = ["docker", "run"]
        if not config.backend.keep_container:
            cmd.append("--rm")
        cmd.extend(["--name", container_name])
        cmd.extend(["-p", f"{config.backend.port}:{config.backend.port}"])
        if config.backend.hf_cache:
            cmd.extend(["-v", f"{config.backend.hf_cache}:/root/.cache/huggingface"])
            cmd.extend(["-e", "HF_HOME=/root/.cache/huggingface"])
        if config.backend.hf_token:
            cmd.extend(["-e", f"HF_TOKEN={config.backend.hf_token}"])
            cmd.extend(["-e", f"HUGGING_FACE_HUB_TOKEN={config.backend.hf_token}"])
        for mount in config.backend.extra_mounts:
            if mount:
                cmd.extend(["-v", mount])
        cmd.extend(config.backend.docker_args)
        cmd.append(config.backend.image)
        cmd.extend(config.backend.command)
        return cmd


def _stop_container(container_name: str) -> None:
    subprocess.run(["docker", "rm", "-f", container_name], text=True, capture_output=True, timeout=20)


def _tail_file(path: Path, max_chars: int = 2000) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    return text[-max_chars:].strip()


def _smoke_ping_with_retry(target: HttpBenchTarget, api: str, attempts: int = 3, sleep_seconds: int = 5) -> str | None:
    """Run the smoke ping a few times: vllm's first forward pass can be slow
    even after /v1/models returns 200 (CUDA graph capture). Tolerate that."""
    print(f"[setup] smoke ping start | api={api} | max_tokens=1 | attempts={attempts}", flush=True)
    last_err: str | None = None
    for attempt in range(1, attempts + 1):
        err = smoke_ping_server(target, api=api, timeout_seconds=120)
        if err is None:
            print(f"[setup] smoke ping ok | attempt={attempt}/{attempts}", flush=True)
            return None
        last_err = err
        print(f"[setup] smoke ping fail | attempt={attempt}/{attempts} | err={err}", flush=True)
        if attempt < attempts:
            time.sleep(sleep_seconds)
    return f"server failed smoke ping after {attempts} attempts: {last_err}"


class _WaitState:
    """Track previously printed log offset so we only show new output."""

    def __init__(self) -> None:
        self.offset = 0


_STARTUP_HINTS = (
    ("no cuda runtime is found", "未检测到 CUDA runtime；docker run 需要 `--gpus all`，且宿主机要装 nvidia-container-toolkit。"),
    ("failed to infer device type", "vllm 找不到设备；通常是 docker 缺 `--gpus all` 或没装 nvidia-container-toolkit。"),
    ("could not select device driver", "docker 没有 GPU driver；安装并启用 nvidia-container-toolkit 后重试。"),
    ("permission denied", "docker socket 权限不足；当前用户加入 docker 组或用 sudo 跑。"),
    ("address already in use", "端口已被占用；换一个 --port 或停掉占用进程。"),
    ("out of memory", "GPU OOM；降低 --gpu-memory-utilization 或 --max-model-len，或加卡 (--tensor-parallel-size)。"),
    ("no such file or directory", "容器内找不到该路径；检查模型挂载 (-v) 和容器内的模型路径是否一致。"),
)


def _print_wait_status(
    elapsed: int,
    log_path: Path,
    proc,
    state: "_WaitState",
) -> bool:
    exited = proc.poll() is not None
    if elapsed != 0 and elapsed % 10 != 0 and not exited:
        return True
    status = "running" if proc.poll() is None else f"exited={proc.poll()}"
    print(f"[setup] waiting for server | elapsed={elapsed}s | container={status}", flush=True)
    new_log, state.offset = _read_new(log_path, state.offset)
    if new_log.strip():
        # Indent so it's visually distinct from the [setup] prefix lines.
        for line in new_log.rstrip().splitlines():
            print(f"    {line}", flush=True)
    if exited:
        _print_diagnostics(log_path, proc.poll())
    return not exited


def _read_new(path: Path, offset: int) -> tuple[str, int]:
    """Return (text_since_offset, new_offset). Caps payload to 2 KB / call."""
    if not path.exists():
        return "", offset
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            fh.seek(offset)
            data = fh.read(2048)
            new_offset = fh.tell()
    except OSError:
        return "", offset
    return data, new_offset


def _print_diagnostics(log_path: Path, exit_code: int | None) -> None:
    """When the container exits unexpectedly, surface a friendly hint."""
    tail = _tail_file(log_path, max_chars=4000).lower()
    hits: list[str] = []
    for needle, hint in _STARTUP_HINTS:
        if needle in tail and hint not in hits:
            hits.append(hint)
    if not hits:
        return
    print(f"[setup] container exited | code={exit_code} | possible causes:", flush=True)
    for hint in hits:
        print(f"    - {hint}", flush=True)
