from __future__ import annotations

import shlex
from pathlib import Path

from llm_bench.backends.base import BackendResult
from llm_bench.config import BenchConfig


def render_markdown_report(
    manifest: dict[str, object],
    summary: dict[str, object],
    config: BenchConfig,
    result: BackendResult,
    runtime: dict[str, object],
    chart_paths: list[Path],
) -> str:
    backend_block = manifest.get("backend") or {}
    container_command = backend_block.get("container_command") or []
    docker_args = backend_block.get("docker_args") or []
    launch_command = backend_block.get("launch_command") or []

    # -- charts: 按用途分桶（决定每个章节插哪几张图）
    chart_buckets = _bucket_charts(chart_paths)

    # -- model 名 + 后端配置块
    if config.backend.name == "transformers":
        backend_section = _transformers_backend_section(config, result)
        model_display = config.transformers.model_path or "unknown"
    else:
        container_cmd_md = shlex.join(container_command) if container_command else "(none)"
        launch_cmd_md = shlex.join(launch_command) if launch_command else "(none)"
        docker_args_md = " ".join(docker_args) if docker_args else "(none)"
        backend_section = (
            "## 后端配置 (容器)\n\n"
            "| field | value |\n|---|---|\n"
            f"| image | `{config.backend.image}` |\n"
            f"| port | `{config.backend.port}` |\n"
            f"| model_name (API) | `{config.backend.model_name}` |\n"
            f"| docker_args | `{docker_args_md}` |\n"
            f"| startup_seconds | `{result.startup_seconds}` |\n\n"
            "容器内命令：\n\n"
            f"```bash\n{container_cmd_md}\n```\n\n"
            "实际 docker run 命令：\n\n"
            f"```bash\n{launch_cmd_md}\n```\n"
        )
        model_display = config.backend.model_name or "unknown"

    return f"""# 推理压测报告 · {model_display}

> run_id: `{manifest["run_id"]}` · backend: `{config.backend.name}` · profile: `{config.workload.profile}`

# 一、配置

{_render_environment(runtime, config)}

{backend_section}

{_render_workload_config(config)}

# 二、性能指标

{_render_tldr(summary, config)}

{_render_performance_summary(summary, config)}

{_render_per_workload_tables(summary, config, chart_buckets)}

{_render_gpu_block(summary, result.gpu_metrics or [], chart_buckets)}

# 三、错误与建议

{_render_errors_outliers(summary, result)}

{_render_recommendations(summary, result.gpu_metrics or [])}

# 四、名词解释

{_render_glossary_body()}
"""


# ---------------------------------------------------------------------------
# 顶部 TL;DR：5 个最关键的数字，扫一眼能下判断
# ---------------------------------------------------------------------------


def _render_tldr(summary: dict[str, object], config: BenchConfig) -> str:
    if config.backend.name == "dry-run":
        return (
            "## TL;DR\n\n"
            "> ⚠ 自检 run (dry-run)，仅验证报告链路，不代表真实性能。"
        )
    output_tps = summary.get("output_tokens_per_sec", 0)
    input_tps = summary.get("input_tokens_per_sec", 0)
    decode_tps_p50 = summary.get("decode_tps_p50", 0)
    prefill_tps_mean = summary.get("prefill_tps_per_req_mean", 0)
    ttft_p99 = summary.get("ttft_p99_ms", 0)
    ok = summary.get("success_requests", 0)
    fail = summary.get("failed_requests", 0)
    qps = summary.get("qps", 0)

    transformers_note = ""
    if config.backend.name == "transformers":
        transformers_note = (
            "\n\n> ℹ transformers 后端把 concurrency 当作 batch_size，"
            "延迟为「批耗时 ÷ 批大小」摊还值；与 vLLM/SGLang 口径不同，**不建议横向对比**。"
        )

    return (
        "## TL;DR\n\n"
        "| 关键指标 | 值 | 含义 |\n|---|---:|---|\n"
        f"| **Output TPS (system)** | **`{output_tps}` tok/s** | 整个系统每秒输出 token 数 (主指标) |\n"
        f"| Decode TPS (per req, p50) | `{decode_tps_p50}` tok/s | 单请求 decode 速度 (= 1000/TPOT) |\n"
        f"| Prefill TPS (per req, mean) | `{prefill_tps_mean}` tok/s | 单请求 prefill 速度 (= input_tokens/TTFT) |\n"
        f"| Input TPS (system) | `{input_tps}` tok/s | 整个系统每秒输入 token 数 |\n"
        f"| TTFT p99 | `{ttft_p99}` ms | 首 token 时延 (尾部) |\n"
        f"| 请求统计 | `{ok}` ok / `{fail}` fail | QPS=`{qps}` |"
        + transformers_note
    )


# ---------------------------------------------------------------------------
# 性能摘要：四个独立的小段，每段给一两个核心数字 + 简短解读
# ---------------------------------------------------------------------------


def _render_performance_summary(summary: dict[str, object], config: BenchConfig) -> str:
    if config.backend.name == "dry-run":
        return ""
    return (
        "## 性能摘要 (全局聚合)\n\n"
        + _section_throughput(summary)
        + "\n\n"
        + _section_decode(summary)
        + "\n\n"
        + _section_prefill(summary)
        + "\n\n"
        + _section_latency(summary)
    )


def _section_throughput(summary: dict[str, object]) -> str:
    return (
        "### 吞吐 (system throughput, 跨并发聚合)\n\n"
        "| metric | value | unit |\n|---|---:|---|\n"
        f"| Output TPS | `{summary.get('output_tokens_per_sec', 0)}` | tokens/s |\n"
        f"| Input TPS | `{summary.get('input_tokens_per_sec', 0)}` | tokens/s |\n"
        f"| Total TPS (input+output) | `{summary.get('total_tokens_per_sec', 0)}` | tokens/s |\n"
        f"| QPS | `{summary.get('qps', 0)}` | req/s |"
    )


def _section_decode(summary: dict[str, object]) -> str:
    p50 = summary.get("decode_tps_p50", 0)
    p99 = summary.get("decode_tps_p99", 0)
    tpot_p50 = summary.get("tpot_p50_ms", 0)
    tpot_p99 = summary.get("tpot_p99_ms", 0)
    return (
        "### Decode 速度 (per-request, 用户感受)\n\n"
        "Decode TPS = `1000 / TPOT(ms)`，表示**单个请求**每秒能吐多少 token。\n\n"
        "- 「**Decode TPS @ TPOT p50**」: 一半用户感受快于此值\n"
        "- 「**Decode TPS @ TPOT p99**」: 99% 用户的最差体验（= 最慢请求的速度）\n\n"
        "| metric | @ TPOT p50 | @ TPOT p99 |\n|---|---:|---:|\n"
        f"| Decode TPS | `{p50}` tok/s | `{p99}` tok/s |\n"
        f"| TPOT | `{tpot_p50}` ms | `{tpot_p99}` ms |"
    )


def _section_prefill(summary: dict[str, object]) -> str:
    return (
        "### Prefill 速度 (per-request, 长上下文/RAG/Agent 关键)\n\n"
        "Prefill TPS = `input_tokens / TTFT(s)`，表示**单个请求** prefill 阶段的吞吐。\n\n"
        "| metric | mean | p50 | p99 |\n|---|---:|---:|---:|\n"
        f"| Prefill TPS | `{summary.get('prefill_tps_per_req_mean', 0)}` tok/s | "
        f"`{summary.get('prefill_tps_per_req_p50', 0)}` tok/s | "
        f"`{summary.get('prefill_tps_per_req_p99', 0)}` tok/s |"
    )


def _section_latency(summary: dict[str, object]) -> str:
    return (
        "### 时延分位 (per-request, ms)\n\n"
        "| metric | p50 | p90 | p99 |\n|---|---:|---:|---:|\n"
        f"| TTFT | `{summary.get('ttft_p50_ms', 0)}` | `{summary.get('ttft_p90_ms', 0)}` | `{summary.get('ttft_p99_ms', 0)}` |\n"
        f"| TPOT | `{summary.get('tpot_p50_ms', 0)}` | `{summary.get('tpot_p90_ms', 0)}` | `{summary.get('tpot_p99_ms', 0)}` |\n"
        f"| E2E | `{summary.get('e2e_p50_ms', 0)}` | `{summary.get('e2e_p90_ms', 0)}` | `{summary.get('e2e_p99_ms', 0)}` |"
    )


# ---------------------------------------------------------------------------
# 分 workload (input/output/concurrency) 明细表 + 相关图
# ---------------------------------------------------------------------------


def _render_per_workload_tables(
    summary: dict[str, object],
    config: BenchConfig,
    chart_buckets: dict[str, list[str]],
) -> str:
    backend_results = summary.get("backend_results") or []
    if not backend_results:
        return ""

    throughput_rows = []
    decode_rows = []
    latency_rows = []
    for item in backend_results:
        w = item["workload"]
        m = item["metrics"]
        key = f"i{w['input_tokens']}/o{w['output_tokens']}/c{w['concurrency']}"
        throughput_rows.append(
            f"| `{key}` | `{m.get('output_tokens_per_sec', 0)}` | "
            f"`{m.get('input_tokens_per_sec', 0)}` | `{m.get('qps', 0)}` |"
        )
        decode_rows.append(
            f"| `{key}` | `{m.get('decode_tps_p50', 0)}` | `{m.get('decode_tps_p99', 0)}` | "
            f"`{m.get('prefill_tps_per_req_mean', 0)}` | `{m.get('prefill_tps_per_req_p99', 0)}` |"
        )
        latency_rows.append(
            f"| `{key}` | `{m.get('ttft_p50_ms', 0)}` | `{m.get('ttft_p90_ms', 0)}` | `{m.get('ttft_p99_ms', 0)}` "
            f"| `{m.get('tpot_p50_ms', 0)}` | `{m.get('tpot_p90_ms', 0)}` | `{m.get('tpot_p99_ms', 0)}` "
            f"| `{m.get('e2e_p50_ms', 0)}` | `{m.get('e2e_p90_ms', 0)}` | `{m.get('e2e_p99_ms', 0)}` |"
        )

    throughput_charts = "\n\n".join(chart_buckets.get("throughput", []))
    latency_charts = "\n\n".join(chart_buckets.get("latency", []))
    # Avoid emitting a "\n\n\n\n" hole where the image would otherwise sit
    # when matplotlib isn't installed and no PNGs were generated.
    throughput_block = f"\n{throughput_charts}\n" if throughput_charts else ""
    latency_block = f"\n{latency_charts}\n" if latency_charts else ""

    return f"""## 分 workload 明细

> workload key = `i<input_tokens>/o<output_tokens>/c<concurrency>`

### 吞吐 (按 workload)
{throughput_block}
| workload | Output TPS | Input TPS | QPS |
|---|---:|---:|---:|
{chr(10).join(throughput_rows)}

### Decode / Prefill 速度 (按 workload)

| workload | Decode p50 | Decode p99 | Prefill mean | Prefill p99 |
|---|---:|---:|---:|---:|
{chr(10).join(decode_rows)}

### 时延 (按 workload)
{latency_block}
| workload | TTFT p50 | TTFT p90 | TTFT p99 | TPOT p50 | TPOT p90 | TPOT p99 | E2E p50 | E2E p90 | E2E p99 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(latency_rows)}"""


# ---------------------------------------------------------------------------
# GPU：数字汇总表 + 图
# ---------------------------------------------------------------------------


def _render_gpu_block(
    summary: dict[str, object],
    gpu_rows: list[dict[str, object]],
    chart_buckets: dict[str, list[str]],
) -> str:
    if not gpu_rows:
        return (
            "## GPU 与显存\n\n"
            "未采集到 GPU 指标。常见原因：自检 run、未安装 `nvidia-smi`、或非 NVIDIA 环境。"
        )
    gpu_charts = "\n\n".join(chart_buckets.get("gpu", []))
    return (
        "## GPU 与显存\n\n"
        f"采集到 `{len(gpu_rows)}` 条 GPU 采样，原始数据见 `metrics.gpu.jsonl`。\n\n"
        "| metric | value | unit |\n|---|---:|---|\n"
        f"| Util Avg | `{summary.get('gpu_avg_utilization', 0)}` | % |\n"
        f"| Util Max | `{summary.get('gpu_max_utilization', 0)}` | % |\n"
        f"| Mem Avg | `{summary.get('gpu_avg_memory_used_mb', 0)}` | MiB |\n"
        f"| Mem Peak | `{summary.get('gpu_peak_memory_used_mb', 0)}` | MiB |\n"
        f"| Temp Avg | `{summary.get('gpu_avg_temperature', 0)}` | °C |\n"
        f"| Temp Max | `{summary.get('gpu_max_temperature', 0)}` | °C |\n"
        f"| Power Avg | `{summary.get('gpu_avg_power_w', 0)}` | W |\n"
        f"| Power Max | `{summary.get('gpu_max_power_w', 0)}` | W |\n\n"
        f"{gpu_charts}"
    )


# ---------------------------------------------------------------------------
# 环境 / Workload 配置 / 错误 / 离群点 / 建议 / 名词解释
# ---------------------------------------------------------------------------


def _render_environment(runtime: dict[str, object], config: BenchConfig) -> str:
    gpu = (runtime or {}).get("gpu") or {}
    gpus = gpu.get("gpus") or []
    models = sorted({str(item.get("name")) for item in gpus if item.get("name")})
    gpu_line = (
        f"`{', '.join(models)}` ×{gpu.get('gpu_count', 0)}"
        if models
        else "未检测到"
    )
    lines = ["## 环境", "", "| field | value |", "|---|---|"]
    lines.append(f"| GPU | {gpu_line} |")
    docker = (runtime or {}).get("docker") or {}
    port = (runtime or {}).get("port") or {}
    disk = (runtime or {}).get("disk") or {}
    if docker:
        lines.append(f"| docker | installed=`{docker.get('installed')}` daemon_ok=`{docker.get('daemon_ok')}` image=`{docker.get('image') or '-'}` |")
    if port:
        lines.append(f"| port available | `{port.get('available')}` |")
    if disk:
        lines.append(f"| disk free | `{disk.get('free_gb')}` GB |")
    return "\n".join(lines)


def _render_workload_config(config: BenchConfig) -> str:
    return (
        "## Workload 配置\n\n"
        "| field | value |\n|---|---|\n"
        f"| profile | `{config.workload.profile}` |\n"
        f"| mode | `{config.workload.mode}` |\n"
        f"| api | `{config.workload.api}` |\n"
        f"| stream | `{config.workload.stream}` |\n"
        f"| input_tokens | `{config.workload.input_tokens}` |\n"
        f"| output_tokens | `{config.workload.output_tokens}` |\n"
        f"| concurrency | `{config.workload.concurrency}` |\n"
        f"| total_requests | `{config.workload.total_requests}` |\n"
        f"| prompt_jsonl | `{config.workload.prompt_jsonl or '-'}` |\n"
        f"| prompt_dir | `{config.workload.prompt_dir or '-'}` |"
    )


def _render_errors_outliers(summary: dict[str, object], result: BackendResult) -> str:
    errors = result.errors or []
    err_text = "\n".join(f"- {e}" for e in errors) if errors else "- 无"
    timeout = int(summary.get("timeout_requests") or 0)
    oom = int(summary.get("oom_count") or 0)
    failed = int(summary.get("failed_requests") or 0)
    cats = summary.get("error_categories") or {}
    cat_lines = "\n".join(f"  - `{k}` × `{v}`" for k, v in cats.items()) if cats else "  - (无)"
    return (
        "## 错误与离群点\n\n"
        "### 错误概况\n\n"
        f"- failed_requests: `{failed}`\n"
        f"- timeout_requests: `{timeout}`\n"
        f"- oom_count: `{oom}`\n"
        "- error_categories:\n"
        f"{cat_lines}\n\n"
        "### 启动 / 全局错误\n\n"
        f"{err_text}\n\n"
        "### 请求级离群点 (E2E 最慢的 5 条)\n\n"
        f"{_render_outliers(result)}"
    )


def _render_outliers(result: BackendResult) -> str:
    rows = [m for m in result.request_metrics if m.success]
    if len(rows) < 3:
        return "请求数量较少，跳过离群点分析。"
    ordered = sorted(rows, key=lambda m: m.e2e_latency_ms, reverse=True)
    lines = ["| request_id | e2e ms | ttft ms | tpot ms | input | output | concurrency |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for m in ordered[:5]:
        # Round ms values for display - raw floats look like debug dumps in a
        # report (12677.156029000746 ms is 4 fractional digits more than anyone
        # could use). Token counts and concurrency are already int-like.
        lines.append(
            f"| `{m.request_id}` | `{m.e2e_latency_ms:.1f}` | `{m.ttft_ms:.1f}` | `{m.tpot_ms:.2f}` "
            f"| `{m.input_tokens}` | `{m.output_tokens}` | `{m.concurrency}` |"
        )
    return "\n".join(lines)


def _render_recommendations(
    summary: dict[str, object],
    gpu_rows: list[dict[str, object]],
) -> str:
    recs: list[str] = []
    if summary.get("failed_requests", 0):
        recs.append("存在失败请求，检查 `metrics.requests.jsonl` 的 `error_category` 与容器日志。")
    for item in summary.get("backend_results") or []:
        metrics = item.get("metrics") or {}
        workload = item.get("workload") or {}
        ttft = float(metrics.get("ttft_p99_ms") or 0)
        tpot = float(metrics.get("tpot_p99_ms") or 0)
        if ttft > 2000:
            recs.append(
                f"workload `i{workload.get('input_tokens')}/o{workload.get('output_tokens')}/c{workload.get('concurrency')}` "
                f"TTFT p99=`{ttft}`ms 偏高 → 检查 prefill / KV cache / 排队。"
            )
        if tpot > 100:
            recs.append(
                f"workload `i{workload.get('input_tokens')}/o{workload.get('output_tokens')}/c{workload.get('concurrency')}` "
                f"TPOT p99=`{tpot}`ms 偏高 → 检查 decode / GPU 利用率 / TP 通信。"
            )
    if gpu_rows:
        utils = [_to_float(row.get("utilization", row.get("utilization.gpu"))) for row in gpu_rows]
        if utils:
            avg = sum(utils) / len(utils)
            if avg < 40:
                recs.append(f"GPU 平均利用率 `{avg:.1f}%` 偏低 → 检查 batch / concurrency / 上游瓶颈。")
    if not recs:
        recs.append("未发现明显异常。")
    return "## 优化建议\n\n" + "\n".join(f"- {item}" for item in recs)


def _render_glossary_body() -> str:
    return (
        "- **Output TPS (system)**: 整个推理系统每秒输出的 token 数，大模型领域主指标 (vLLM / SGLang / TRT-LLM 一致)。\n"
        "- **Decode TPS (per req)**: 单个请求每秒能 decode 多少 token，= `1000 / TPOT(ms)`。用户感受到的「打字速度」。\n"
        "- **Prefill TPS (per req)**: 单个请求 prefill 阶段的吞吐，= `input_tokens / TTFT(s)`。长上下文 / RAG / Agent 性能的关键。\n"
        "- **Input TPS**: 整个系统每秒输入的 token 数。\n"
        "- **TTFT (Time To First Token)**: 发出请求到收到第一个 token 的时延 (prefill + 排队)。\n"
        "- **TPOT (Time Per Output Token)**: 第一个 token 之后，平均每个输出 token 的时间。\n"
        "- **E2E**: 单请求端到端时延。\n"
        "- **QPS**: 每秒成功请求数。在 LLM 场景下不如 Output TPS 直观 (单请求 token 数差异巨大)。\n"
        "- 说明: `stream=true` 时 TTFT / TPOT 是真实测量；`stream=false` 时 TTFT≈E2E、TPOT 为摊还值。"
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _transformers_backend_section(config: BenchConfig, result: BackendResult) -> str:
    tx = config.transformers
    return (
        "## 后端配置 (transformers)\n\n"
        "| field | value |\n|---|---|\n"
        f"| model_path | `{tx.model_path}` |\n"
        f"| tokenizer_path | `{tx.tokenizer_path or '(= model_path)'}` |\n"
        f"| torch_dtype | `{tx.torch_dtype}` |\n"
        f"| device_map | `{tx.device_map}` |\n"
        f"| quantization | `{tx.quantization or '(none)'}` |\n"
        f"| trust_remote_code | `{tx.trust_remote_code}` |\n"
        f"| revision | `{tx.revision}` |\n"
        f"| batch_size | `{tx.batch_size}` |\n"
        f"| sampling | do_sample=`{tx.do_sample}` top_k=`{tx.top_k}` "
        f"top_p=`{config.workload.top_p}` temperature=`{config.workload.temperature}` |\n"
        f"| startup_seconds | `{result.startup_seconds}` |\n"
        f"| peak_memory_mb | `{result.peak_memory_mb}` |"
    )


_PER_INPUT_THROUGHPUT_KEYS = ("output_tps", "decode_tps")
_PER_INPUT_LATENCY_KEYS = ("ttft", "e2e", "tpot")


def _bucket_charts(chart_paths: list[Path]) -> dict[str, list[str]]:
    """按 chart 文件名分桶，让各章节插对应的图。"""
    buckets: dict[str, list[str]] = {"throughput": [], "latency": [], "gpu": [], "other": []}
    for path in chart_paths:
        ref = f"![{path.stem}](images/{path.name})"
        name = path.stem.lower()
        # The faceted views (per_input_<I>_<metric>.png, per_output_<O>_<metric>.png)
        # route to throughput/latency based on the metric suffix, not the
        # facet prefix.
        if name.startswith("per_input_") or name.startswith("per_output_"):
            if any(k in name for k in _PER_INPUT_LATENCY_KEYS):
                buckets["latency"].append(ref)
            else:
                buckets["throughput"].append(ref)
        elif "throughput" in name or "qps" in name or "concurrency" in name:
            buckets["throughput"].append(ref)
        elif "latency" in name or "tpot" in name or "ttft" in name or "percentile" in name:
            buckets["latency"].append(ref)
        elif "gpu" in name:
            buckets["gpu"].append(ref)
        else:
            buckets["other"].append(ref)
    return buckets


def _to_float(value: object) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0
