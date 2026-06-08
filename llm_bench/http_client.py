from __future__ import annotations

import contextlib
import json
import queue
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from itertools import cycle

from llm_bench.backends.base import RequestMetric
from llm_bench.config import BenchConfig
from llm_bench.errors import classify_error
from llm_bench.tokenizer import build_token_counter
from llm_bench.workload import WorkloadRequest, build_workload_requests


@dataclass
class HttpBenchTarget:
    url: str
    model: str
    backend: str


def smoke_ping_server(
    target: HttpBenchTarget,
    api: str = "completions",
    timeout_seconds: int = 60,
) -> str | None:
    """End-to-end smoke check before the real benchmark starts.

    `wait_for_openai_server` only verifies the HTTP server answers `/v1/models`
    or `/health`; vllm/sglang can flip those to 200 before the first actual
    forward pass works (CUDA graph capture, kernel autotune, ...). We send one
    tiny request with `max_tokens=1` so the server walks the full request path
    once. Returns None on success, an error string otherwise.
    """
    if api == "chat":
        url = f"{target.url.rstrip('/')}/v1/chat/completions"
        payload = {
            "model": target.model,
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 1,
            "temperature": 0.0,
            "stream": False,
        }
    else:
        url = f"{target.url.rstrip('/')}/v1/completions"
        payload = {
            "model": target.model,
            "prompt": "ping",
            "max_tokens": 1,
            "temperature": 0.0,
            "stream": False,
        }
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
            body = resp.read(8192)
            if resp.status >= 300:
                return f"smoke ping HTTP {resp.status}: {body[:200]!r}"
            if b'"choices"' not in body:
                return f"smoke ping unexpected body: {body[:200]!r}"
            return None
    except urllib.error.HTTPError as exc:
        try:
            err_body = exc.read(500).decode("utf-8", errors="replace")
        except Exception:
            err_body = ""
        return f"smoke ping HTTP {exc.code}: {err_body or exc.reason}"
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        return f"smoke ping failed: {exc}"


def wait_for_openai_server(base_url: str, timeout_seconds: int, on_wait=None) -> bool:
    deadline = time.monotonic() + timeout_seconds
    started = time.monotonic()
    last_elapsed = -1
    urls = [
        f"{base_url.rstrip('/')}/v1/models",
        f"{base_url.rstrip('/')}/health",
    ]
    while time.monotonic() < deadline:
        for url in urls:
            try:
                with urllib.request.urlopen(url, timeout=3) as resp:
                    if 200 <= resp.status < 300:
                        return True
            except Exception:
                pass
        elapsed = int(time.monotonic() - started)
        if on_wait and elapsed != last_elapsed:
            if on_wait(elapsed) is False:
                return False
            last_elapsed = elapsed
        time.sleep(1)
    return False


def run_openai_http_benchmark(config: BenchConfig, target: HttpBenchTarget) -> list[RequestMetric]:
    rows: list[RequestMetric] = []
    token_counter = build_token_counter(config)
    workload_requests = build_workload_requests(config, token_counter=token_counter)
    request_index = 1

    total_segments = len(config.workload.concurrency)
    for seg_idx, concurrency in enumerate(config.workload.concurrency, start=1):
        warmup_cycle = cycle(workload_requests)
        warmup_n = config.workload.warmup_requests
        if warmup_n:
            print(f"[seg {seg_idx}/{total_segments}] warmup | requests={warmup_n}", flush=True)
        for warmup_idx in range(warmup_n):
            _send_one(
                request_id=f"warmup_{warmup_idx:06d}",
                config=config,
                target=target,
                workload=next(warmup_cycle),
                concurrency=concurrency,
            )

        work = queue.Queue()
        request_cycle = cycle(workload_requests)
        request_count = config.workload.total_requests
        if config.workload.duration_seconds:
            request_count = max(concurrency, 1)
        for _ in range(request_count):
            work.put((request_index, next(request_cycle)))
            request_index += 1

        print(
            f"[seg {seg_idx}/{total_segments}] start | concurrency={concurrency} | total={request_count}",
            flush=True,
        )
        seg_started = time.monotonic()
        progress = _BenchmarkProgress(total=request_count, label=f"seg {seg_idx}/{total_segments}")
        seg_rows: list[RequestMetric] = []
        lock = threading.Lock()
        workers = []
        deadline = time.monotonic() + config.workload.duration_seconds if config.workload.duration_seconds else None
        next_request_lock = threading.Lock()
        for _ in range(max(concurrency, 1)):
            thread = threading.Thread(
                target=_worker,
                args=(
                    work, lock, seg_rows, config, target, concurrency,
                    deadline, request_cycle, next_request_lock, progress,
                ),
                daemon=True,
            )
            thread.start()
            workers.append(thread)
        for thread in workers:
            thread.join()
        progress.finalize()
        rows.extend(seg_rows)
        ok = sum(1 for r in seg_rows if r.success)
        seg_elapsed = max(time.monotonic() - seg_started, 1e-6)
        print(
            f"[seg {seg_idx}/{total_segments}] done | elapsed={seg_elapsed:.1f}s | "
            f"ok={ok} | fail={len(seg_rows) - ok} | qps={ok / seg_elapsed:.2f}",
            flush=True,
        )
    return rows


class _BenchmarkProgress:
    """Periodically prints '[seg X/Y] progress N/M (P%) at Q req/s' from worker threads."""

    def __init__(self, total: int, label: str, every_pct: float = 10.0) -> None:
        self._total = max(total, 1)
        self._label = label
        self._step = max(int(self._total * every_pct / 100.0), 1)
        self._next = self._step
        self._done = 0
        self._lock = threading.Lock()
        self._started = time.monotonic()

    def tick(self) -> None:
        with self._lock:
            self._done += 1
            if self._done < self._next and self._done < self._total:
                return
            self._next += self._step
            done = self._done
        elapsed = max(time.monotonic() - self._started, 1e-6)
        pct = done * 100.0 / self._total
        rate = done / elapsed
        print(f"[{self._label}] progress | {done}/{self._total} ({pct:.0f}%) | rate={rate:.2f} req/s", flush=True)

    def finalize(self) -> None:
        with self._lock:
            done = self._done
        if 0 < done < self._total:
            print(f"[{self._label}] progress | {done}/{self._total} | status=early_stop", flush=True)


def _worker(
    work: queue.Queue,
    lock: threading.Lock,
    rows: list[RequestMetric],
    config: BenchConfig,
    target: HttpBenchTarget,
    concurrency: int,
    deadline: float | None,
    request_cycle,
    next_request_lock: threading.Lock,
    progress: "_BenchmarkProgress | None" = None,
) -> None:
    while True:
        from_queue = False
        try:
            request_number, workload = work.get_nowait()
            from_queue = True
        except queue.Empty:
            if deadline is None or time.monotonic() >= deadline:
                return
            with next_request_lock:
                request_number = int(time.time_ns())
                workload = next(request_cycle)
        metric = _send_one(
            request_id=f"req_{request_number:06d}",
            config=config,
            target=target,
            workload=workload,
            concurrency=concurrency,
        )
        with lock:
            rows.append(metric)
        if progress is not None:
            progress.tick()
        if from_queue:
            with contextlib.suppress(ValueError):
                work.task_done()


def _send_one(
    request_id: str,
    config: BenchConfig,
    target: HttpBenchTarget,
    workload: WorkloadRequest,
    concurrency: int,
) -> RequestMetric:
    payload = _payload(config, target, workload)
    endpoint = "chat/completions" if config.workload.api == "chat" else "completions"
    url = f"{target.url.rstrip('/')}/v1/{endpoint}"
    started = time.perf_counter()
    start_unix = time.time()
    first_byte_at: float | None = None
    actual_output_tokens = 0
    output_text = ""
    try:
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=config.workload.request_timeout_seconds) as resp:
            if config.workload.stream:
                chunks = []
                while True:
                    line = resp.readline()
                    if not line:
                        break
                    if first_byte_at is None and line.startswith(b"data: "):
                        sse_payload = line[len(b"data: "):].strip()
                        if sse_payload and sse_payload != b"[DONE]":
                            try:
                                sse_data = json.loads(sse_payload)
                                sse_choices = sse_data.get("choices") or []
                                if sse_choices:
                                    sse_text = (
                                        sse_choices[0].get("delta", {}).get("content")
                                        or sse_choices[0].get("text")
                                        or ""
                                    )
                                    if sse_text:
                                        first_byte_at = time.perf_counter()
                            except json.JSONDecodeError:
                                pass
                    chunks.append(line)
                body = b"".join(chunks).decode("utf-8", errors="replace")
                actual_output_tokens = _estimate_stream_tokens(body, workload.output_tokens)
                output_text = _extract_stream_text(body)
            else:
                body_bytes = resp.read()
                first_byte_at = time.perf_counter()
                body = body_bytes.decode("utf-8", errors="replace")
                actual_output_tokens = _extract_completion_tokens(body, workload.output_tokens)
                output_text = _extract_completion_text(body)
        ended = time.perf_counter()
        end_unix = time.time()
        e2e_ms = (ended - started) * 1000.0
        if config.workload.stream:
            # Streaming gives a real first-token timestamp, so TTFT and the
            # decode-phase TPOT are both meaningful.
            ttft_ms = ((first_byte_at or ended) - started) * 1000.0
            tpot_ms = max((e2e_ms - ttft_ms) / max(actual_output_tokens - 1, 1), 0.0)
        else:
            # Non-streaming cannot isolate the first token: the whole response
            # arrives at once. Report TTFT == E2E and an amortized per-token
            # cost rather than a misleading near-zero TPOT.
            ttft_ms = e2e_ms
            tpot_ms = e2e_ms / max(actual_output_tokens, 1)
        return RequestMetric(
            request_id=request_id,
            backend=target.backend,
            concurrency=concurrency,
            input_tokens=workload.input_tokens,
            output_tokens=actual_output_tokens,
            requested_output_tokens=workload.output_tokens,
            ttft_ms=round(ttft_ms, 3),
            tpot_ms=round(tpot_ms, 3),
            e2e_latency_ms=round(e2e_ms, 3),
            metadata=workload.metadata,
            prompt_sample=workload.prompt[:500],
            output_sample=output_text[:1000],
            output_valid=actual_output_tokens > 0,
            validation_error=None if actual_output_tokens > 0 else "empty output",
            start_unix=start_unix,
            end_unix=end_unix,
        )
    except urllib.error.HTTPError as exc:
        ended = time.perf_counter()
        error = _http_error_message(exc)
        return RequestMetric(
            request_id=request_id,
            backend=target.backend,
            concurrency=concurrency,
            input_tokens=workload.input_tokens,
            output_tokens=0,
            requested_output_tokens=workload.output_tokens,
            ttft_ms=0.0,
            tpot_ms=0.0,
            e2e_latency_ms=round((ended - started) * 1000.0, 3),
            success=False,
            error=error,
            error_category=classify_error(error),
            metadata=workload.metadata,
            start_unix=start_unix,
            end_unix=time.time(),
        )
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
        ended = time.perf_counter()
        return RequestMetric(
            request_id=request_id,
            backend=target.backend,
            concurrency=concurrency,
            input_tokens=workload.input_tokens,
            output_tokens=0,
            requested_output_tokens=workload.output_tokens,
            ttft_ms=0.0,
            tpot_ms=0.0,
            e2e_latency_ms=round((ended - started) * 1000.0, 3),
            success=False,
            error=str(exc),
            error_category=classify_error(exc),
            metadata=workload.metadata,
            start_unix=start_unix,
            end_unix=time.time(),
        )


def _extract_completion_tokens(body: str, fallback: int) -> int:
    data = json.loads(body)
    usage = data.get("usage") or {}
    completion_tokens = usage.get("completion_tokens")
    if isinstance(completion_tokens, int):
        return completion_tokens
    choices = data.get("choices") or []
    if choices:
        text = choices[0].get("text") or choices[0].get("message", {}).get("content") or ""
        return max(len(str(text).split()), 1)
    return fallback


def _extract_completion_text(body: str) -> str:
    data = json.loads(body)
    choices = data.get("choices") or []
    if not choices:
        return ""
    choice = choices[0]
    return str(choice.get("text") or choice.get("message", {}).get("content") or "")


def _payload(config: BenchConfig, target: HttpBenchTarget, workload: WorkloadRequest) -> dict[str, object]:
    # When streaming, ask the server to include the final `usage` block in the
    # last SSE chunk so we can count tokens with the model's real tokenizer
    # instead of approximating "1 chunk = 1 token" (vllm/sglang batch tokens
    # per chunk under load, undercounting Output TPS otherwise).
    stream_options: dict[str, bool] | None = (
        {"include_usage": True} if config.workload.stream else None
    )
    if config.workload.api == "chat":
        messages = workload.messages or [{"role": "user", "content": workload.prompt}]
        payload: dict[str, object] = {
            "model": target.model,
            "messages": messages,
            "max_tokens": workload.output_tokens,
            "temperature": config.workload.temperature,
            "top_p": config.workload.top_p,
            "stream": config.workload.stream,
        }
        if stream_options is not None:
            payload["stream_options"] = stream_options
        return payload
    payload = {
        "model": target.model,
        "prompt": workload.prompt,
        "max_tokens": workload.output_tokens,
        "temperature": config.workload.temperature,
        "top_p": config.workload.top_p,
        "stream": config.workload.stream,
    }
    if stream_options is not None:
        payload["stream_options"] = stream_options
    return payload


def _estimate_stream_tokens(body: str, fallback: int) -> int:
    """Count output tokens from an SSE stream body.

    Priority (most accurate first):
      1. The server-reported `usage.completion_tokens` in any SSE chunk. OpenAI
         spec since 2024 returns it in the final delta when
         `stream_options.include_usage=true`; vllm and sglang both honour
         this. This is the real tokenizer count.
      2. Falls back to chunk-counting (1 chunk = 1 token), which is a coarse
         approximation: vllm/sglang batch multiple tokens per SSE chunk under
         load, so this UNDERCOUNTS — leading to a depressed Output TPS.
         Better than `fallback` (the requested max_tokens) when usage is absent.
    """
    chunk_count = 0
    usage_total = 0
    for line in body.splitlines():
        if not line.startswith("data: "):
            continue
        payload = line[len("data: ") :].strip()
        if payload == "[DONE]":
            continue
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            continue
        # Prefer the server's exact tokenizer count when present.
        usage = data.get("usage") or {}
        completion_tokens = usage.get("completion_tokens")
        if isinstance(completion_tokens, int) and completion_tokens > 0:
            usage_total = completion_tokens  # OpenAI emits this in the final chunk
        choices = data.get("choices") or []
        if not choices:
            continue
        text = choices[0].get("text") or choices[0].get("delta", {}).get("content") or ""
        if text:
            chunk_count += 1
    if usage_total > 0:
        return usage_total
    return chunk_count or fallback


def _extract_stream_text(body: str) -> str:
    parts = []
    for line in body.splitlines():
        if not line.startswith("data: "):
            continue
        payload = line[len("data: ") :].strip()
        if payload == "[DONE]":
            continue
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            continue
        choices = data.get("choices") or []
        if not choices:
            continue
        text = choices[0].get("text") or choices[0].get("delta", {}).get("content") or ""
        if text:
            parts.append(str(text))
    return "".join(parts)


def _http_error_message(exc: urllib.error.HTTPError) -> str:
    try:
        body = exc.read(1000).decode("utf-8", errors="replace")
    except Exception:
        body = ""
    body_suffix = f": {body}" if body else ""
    return f"HTTP error {exc.code}{body_suffix}"
