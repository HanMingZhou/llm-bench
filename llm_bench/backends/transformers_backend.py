"""Native HuggingFace transformers backend.

This backend does not use Docker and does not start an HTTP server. It calls
`AutoModelForCausalLM.from_pretrained` and `model.generate` directly in the
same process. All parameter names match the transformers library exactly:
`torch_dtype`, `device_map`, `trust_remote_code`, `revision`, `do_sample`,
`temperature`, `top_p`, `top_k`, `repetition_penalty`, `num_beams`, etc.

Concurrency is simulated through generate's `batch_size`: the workload's
concurrency list is interpreted as desired batch sizes, since the local
transformers backend cannot truly parallelize requests on a single GPU.
"""
from __future__ import annotations

import logging
import time

from llm_bench.backends.base import BackendResult, RequestMetric
from llm_bench.config import BenchConfig, TransformersConfig
from llm_bench.errors import classify_error
from llm_bench.gpu import GpuSampler
from llm_bench.workload import build_workload_requests

logger = logging.getLogger(__name__)


class TransformersBackend:
    name = "transformers"

    def run(self, config: BenchConfig) -> BackendResult:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "transformers backend requires `torch` and `transformers`. "
                "Install them with `pip install torch transformers`."
            ) from exc

        tx = config.transformers
        if not tx.model_path:
            raise ValueError("transformers backend requires --model-path (or transformers.model_path in YAML).")

        from_pretrained_kwargs = _from_pretrained_kwargs(tx, torch)
        # `temperature` and `top_p` are read from the workload config so the
        # same CLI flags work across vllm/sglang/transformers.
        generate_defaults = _generate_defaults(tx, config.workload.temperature, config.workload.top_p)

        started = time.perf_counter()
        # Loading a 9B model in fp/bf16 across 2 GPUs takes 30-90s on consumer
        # cards. Without these prints the CLI looks frozen between the
        # Execution plan block and the first warmup tick.
        print(f"[setup] loading tokenizer | path={tx.tokenizer_path or tx.model_path}", flush=True)
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                tx.tokenizer_path or tx.model_path,
                trust_remote_code=tx.trust_remote_code,
                revision=tx.revision,
            )
        except Exception as exc:
            raise RuntimeError(f"tokenizer load failed: {exc}") from exc
        print("[setup] tokenizer ready", flush=True)

        print(
            f"[setup] loading model | path={tx.model_path} | dtype={tx.torch_dtype} | "
            f"device_map={tx.device_map} | quantization={tx.quantization or 'none'}",
            flush=True,
        )
        load_started = time.perf_counter()
        try:
            model = AutoModelForCausalLM.from_pretrained(tx.model_path, **from_pretrained_kwargs)
        except Exception as exc:
            raise RuntimeError(f"model load failed: {exc}") from exc
        model.eval()
        startup_seconds = round(time.perf_counter() - started, 3)
        print(f"[setup] model ready | elapsed={time.perf_counter() - load_started:.1f}s", flush=True)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        peak_memory_mb: float | None = None
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        sampler = GpuSampler()
        sampler.start()
        rows: list[RequestMetric] = []
        oom_detected = False
        try:
            def token_counter(text: str) -> int:
                return len(tokenizer.encode(text, add_special_tokens=False))

            workload_requests = build_workload_requests(config, token_counter)

            warmup = workload_requests[: min(config.workload.warmup_requests, len(workload_requests))]
            if warmup:
                print(f"[setup] warmup | requests={len(warmup)}", flush=True)
            for req in warmup:
                _run_one(model, tokenizer, req.prompt, req.output_tokens, generate_defaults, torch)

            total = config.workload.total_requests
            expanded = (workload_requests * max(total // max(len(workload_requests), 1), 1))[:total]

            total_segments = len(config.workload.concurrency)
            for seg_idx, concurrency in enumerate(config.workload.concurrency, start=1):
                batch_size = max(tx.batch_size, concurrency, 1)
                print(
                    f"[seg {seg_idx}/{total_segments}] start | "
                    f"concurrency={concurrency} | batch_size={batch_size} | total={len(expanded)}",
                    flush=True,
                )
                seg_started = time.perf_counter()
                seg_done_before = len(rows)
                next_progress = max(int(len(expanded) * 0.10), 1)
                idx = 0
                while idx < len(expanded):
                    batch = expanded[idx:idx + batch_size]
                    metrics = _run_batch(
                        start_index=len(rows) + 1,
                        model=model,
                        tokenizer=tokenizer,
                        prompts=[r.prompt for r in batch],
                        input_tokens_list=[r.input_tokens for r in batch],
                        output_tokens=batch[0].output_tokens,
                        metadata_list=[r.metadata for r in batch],
                        concurrency=concurrency,
                        generate_defaults=generate_defaults,
                        backend=config.backend.name,
                        torch=torch,
                    )
                    for metric in metrics:
                        if metric.error_category == "oom":
                            oom_detected = True
                        rows.append(metric)
                    idx += batch_size
                    # Progress every ~10% so the user sees something tick.
                    seg_done = len(rows) - seg_done_before
                    if seg_done >= next_progress or idx >= len(expanded):
                        elapsed = max(time.perf_counter() - seg_started, 1e-6)
                        pct = seg_done * 100.0 / max(len(expanded), 1)
                        rate = seg_done / elapsed
                        print(
                            f"[seg {seg_idx}/{total_segments}] progress {seg_done}/{len(expanded)} "
                            f"({pct:.0f}%) | {rate:.2f} req/s",
                            flush=True,
                        )
                        next_progress = seg_done + max(int(len(expanded) * 0.10), 1)
                seg_elapsed = max(time.perf_counter() - seg_started, 1e-6)
                seg_ok = sum(1 for r in rows[seg_done_before:] if r.success)
                seg_fail = len(rows) - seg_done_before - seg_ok
                print(
                    f"[seg {seg_idx}/{total_segments}] done | elapsed={seg_elapsed:.1f}s | "
                    f"ok={seg_ok} | fail={seg_fail} | rate={seg_ok / seg_elapsed:.2f} req/s",
                    flush=True,
                )
        finally:
            gpu_metrics = sampler.stop()

        if torch.cuda.is_available():
            peak_memory_mb = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2)

        errors: list[str] = []
        if oom_detected:
            errors.append("OOM detected during inference")

        return BackendResult(
            backend=config.backend.name,
            request_metrics=rows,
            startup_seconds=startup_seconds,
            errors=errors,
            gpu_metrics=gpu_metrics,
            peak_memory_mb=peak_memory_mb,
        )


def _from_pretrained_kwargs(tx: TransformersConfig, torch) -> dict[str, object]:
    """Translate TransformersConfig field names into `from_pretrained` kwargs.

    Note: "translate" here only means dataclass-attribute -> kwarg, not
    renaming. Every kwarg name below is the literal name from the transformers
    library. `torch_dtype` is mapped to the actual `torch.dtype` object the
    library expects.

    transformers >= 4.49 renamed `torch_dtype` to `dtype` and emits a
    DeprecationWarning when the old name is used. We pass the new name when
    the installed transformers supports it, falling back to `torch_dtype`
    otherwise so older installs keep working.
    """
    dtype_obj = _torch_dtype(torch, tx.torch_dtype)
    kwargs: dict[str, object] = {
        _dtype_kwarg_name(): dtype_obj,
        "device_map": tx.device_map,
        "trust_remote_code": tx.trust_remote_code,
        "revision": tx.revision,
        "low_cpu_mem_usage": tx.low_cpu_mem_usage,
    }
    quantization = (tx.quantization or "").lower().strip()
    if quantization in {"4bit", "int4", "nf4"}:
        kwargs["load_in_4bit"] = True
    elif quantization in {"8bit", "int8"}:
        kwargs["load_in_8bit"] = True
    return kwargs


def _dtype_kwarg_name() -> str:
    """Detect whether the installed transformers wants `dtype` or `torch_dtype`."""
    try:
        from transformers import AutoModelForCausalLM
        import inspect
        sig = inspect.signature(AutoModelForCausalLM.from_pretrained)
        if "dtype" in sig.parameters:
            return "dtype"
    except Exception:
        pass
    return "torch_dtype"


def _generate_defaults(tx: TransformersConfig, temperature: float, top_p: float) -> dict[str, object]:
    return {
        "do_sample": tx.do_sample,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": tx.top_k,
        "repetition_penalty": tx.repetition_penalty,
        "num_beams": tx.num_beams,
    }


def _run_one(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    generate_defaults: dict[str, object],
    torch,
) -> None:
    encoded = tokenizer(prompt, return_tensors="pt")
    encoded = {key: value.to(model.device) for key, value in encoded.items()}
    try:
        with torch.no_grad():
            model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
                **generate_defaults,
            )
    except torch.cuda.OutOfMemoryError:
        logger.warning("OOM during warmup, skipping remaining warmup")
    except Exception:
        pass


def _run_batch(
    start_index: int,
    model,
    tokenizer,
    prompts: list[str],
    input_tokens_list: list[int],
    output_tokens: int,
    metadata_list: list[dict[str, object]],
    concurrency: int,
    generate_defaults: dict[str, object],
    backend: str,
    torch,
) -> list[RequestMetric]:
    if len(prompts) == 1:
        return [_run_single(start_index, model, tokenizer, prompts[0], input_tokens_list[0], output_tokens, metadata_list[0], concurrency, generate_defaults, backend, torch)]
    encoded = tokenizer(prompts, return_tensors="pt", padding=True, truncation=False)
    encoded = {key: value.to(model.device) for key, value in encoded.items()}
    started = time.perf_counter()
    start_unix = time.time()
    try:
        with torch.no_grad():
            outputs = model.generate(
                **encoded,
                max_new_tokens=output_tokens,
                pad_token_id=tokenizer.eos_token_id,
                **generate_defaults,
            )
        ended = time.perf_counter()
        end_unix = time.time()
        e2e_ms = (ended - started) * 1000.0
        per_request_ms = e2e_ms / len(prompts)
        input_len = int(encoded["input_ids"].shape[-1])
        metrics: list[RequestMetric] = []
        for i, prompt in enumerate(prompts):
            actual_output = max(int(outputs[i].shape[-1]) - input_len, 1)
            output_text = tokenizer.decode(outputs[i][input_len:], skip_special_tokens=True)
            metrics.append(
                RequestMetric(
                    request_id=f"req_{start_index + i:06d}",
                    backend=backend,
                    concurrency=concurrency,
                    input_tokens=input_tokens_list[i],
                    output_tokens=actual_output,
                    requested_output_tokens=output_tokens,
                    ttft_ms=round(per_request_ms, 3),
                    tpot_ms=round(per_request_ms / max(actual_output, 1), 3),
                    e2e_latency_ms=round(per_request_ms, 3),
                    metadata=metadata_list[i],
                    prompt_sample=prompt[:500],
                    output_sample=output_text[:1000],
                    output_valid=actual_output > 0,
                    # Real wall-clock timestamps let metrics.py compute Output
                    # TPS off the actual batch span instead of the (E2E summed
                    # then divided by concurrency) fallback, which over-divides
                    # for transformers batched inference and undercounts TPS.
                    start_unix=start_unix,
                    end_unix=end_unix,
                    validation_error=None if actual_output > 0 else "empty output",
                )
            )
        return metrics
    except torch.cuda.OutOfMemoryError as exc:
        ended = time.perf_counter()
        return [
            _error_metric(start_index + i, prompts[i], input_tokens_list[i], output_tokens, metadata_list[i], concurrency, backend, "oom", str(exc), ended - started, len(prompts))
            for i in range(len(prompts))
        ]
    except Exception as exc:
        ended = time.perf_counter()
        return [
            _error_metric(start_index + i, prompts[i], input_tokens_list[i], output_tokens, metadata_list[i], concurrency, backend, classify_error(exc), str(exc), ended - started, len(prompts))
            for i in range(len(prompts))
        ]


def _run_single(
    start_index: int,
    model,
    tokenizer,
    prompt: str,
    input_tokens: int,
    output_tokens: int,
    metadata: dict[str, object],
    concurrency: int,
    generate_defaults: dict[str, object],
    backend: str,
    torch,
) -> RequestMetric:
    encoded = tokenizer(prompt, return_tensors="pt")
    encoded = {key: value.to(model.device) for key, value in encoded.items()}
    started = time.perf_counter()
    start_unix = time.time()
    try:
        with torch.no_grad():
            output = model.generate(
                **encoded,
                max_new_tokens=output_tokens,
                pad_token_id=tokenizer.eos_token_id,
                **generate_defaults,
            )
        ended = time.perf_counter()
        end_unix = time.time()
        prompt_len = int(encoded["input_ids"].shape[-1])
        actual_output = max(int(output.shape[-1]) - prompt_len, 1)
        output_text = tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)
        e2e_ms = (ended - started) * 1000.0
        return RequestMetric(
            request_id=f"req_{start_index:06d}",
            backend=backend,
            concurrency=concurrency,
            input_tokens=input_tokens,
            output_tokens=actual_output,
            requested_output_tokens=output_tokens,
            ttft_ms=round(e2e_ms, 3),
            tpot_ms=round(e2e_ms / max(actual_output, 1), 3),
            e2e_latency_ms=round(e2e_ms, 3),
            metadata=metadata,
            prompt_sample=prompt[:500],
            output_sample=output_text[:1000],
            output_valid=actual_output > 0,
            validation_error=None if actual_output > 0 else "empty output",
            start_unix=start_unix,
            end_unix=end_unix,
        )
    except torch.cuda.OutOfMemoryError as exc:
        ended = time.perf_counter()
        return _error_metric(start_index, prompt, input_tokens, output_tokens, metadata, concurrency, backend, "oom", str(exc), ended - started, 1)
    except Exception as exc:
        ended = time.perf_counter()
        return _error_metric(start_index, prompt, input_tokens, output_tokens, metadata, concurrency, backend, classify_error(exc), str(exc), ended - started, 1)


def _error_metric(
    index: int,
    prompt: str,
    input_tokens: int,
    output_tokens: int,
    metadata: dict[str, object],
    concurrency: int,
    backend: str,
    category: str,
    message: str,
    elapsed: float,
    batch_size: int,
) -> RequestMetric:
    return RequestMetric(
        request_id=f"req_{index:06d}",
        backend=backend,
        concurrency=concurrency,
        input_tokens=input_tokens,
        output_tokens=0,
        requested_output_tokens=output_tokens,
        ttft_ms=0.0,
        tpot_ms=0.0,
        e2e_latency_ms=round(elapsed * 1000.0 / max(batch_size, 1), 3),
        success=False,
        error=message,
        error_category=category,
        metadata=metadata,
    )


def _torch_dtype(torch, dtype: str):
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }.get((dtype or "").lower(), torch.bfloat16)
