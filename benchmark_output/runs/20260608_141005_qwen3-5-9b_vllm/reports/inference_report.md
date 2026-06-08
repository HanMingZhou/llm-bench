# 推理压测报告 · Qwen3.5-9B

> run_id: `20260608_141005_qwen3-5-9b_vllm` · backend: `vllm` · profile: `quick`

# 一、配置

## 环境

| field | value |
|---|---|
| GPU | `NVIDIA GeForce RTX 3090` ×2 |
| docker | installed=`True` daemon_ok=`True` image=`vllm-openai:latest` |
| port available | `True` |
| disk free | `684.167` GB |

## 后端配置 (容器)

| field | value |
|---|---|
| image | `vllm-openai:latest` |
| port | `8000` |
| model_name (API) | `Qwen3.5-9B` |
| docker_args | `--gpus all --shm-size 16g --ipc=host` |
| startup_seconds | `337.972` |

容器内命令：

```bash
/root/.cache/modelscope/hub/models/Qwen/Qwen3.5-9B --host 0.0.0.0 --port 8000 --served-model-name Qwen3.5-9B --tensor-parallel-size 2 --gpu-memory-utilization 0.9 --max-model-len 4096
```

实际 docker run 命令：

```bash
docker run --rm --name llm-bench-vllm-1780927805 -p 8000:8000 -v /root/.cache/huggingface:/root/.cache/huggingface -e HF_HOME=/root/.cache/huggingface -v /root/.cache/modelscope:/root/.cache/modelscope --gpus all --shm-size 16g --ipc=host vllm-openai:latest /root/.cache/modelscope/hub/models/Qwen/Qwen3.5-9B --host 0.0.0.0 --port 8000 --served-model-name Qwen3.5-9B --tensor-parallel-size 2 --gpu-memory-utilization 0.9 --max-model-len 4096
```


## Workload 配置

| field | value |
|---|---|
| profile | `quick` |
| mode | `fixed` |
| api | `completions` |
| stream | `True` |
| input_tokens | `[512]` |
| output_tokens | `[128]` |
| concurrency | `[1, 4]` |
| total_requests | `32` |
| prompt_jsonl | `-` |
| prompt_dir | `-` |

# 二、性能指标

## TL;DR

| 关键指标 | 值 | 含义 |
|---|---:|---|
| **Output TPS (system)** | **`51.13` tok/s** | 整个系统每秒输出 token 数 (主指标) |
| Decode TPS (per req, p50) | `33.518` tok/s | 单请求 decode 速度 (= 1000/TPOT) |
| Prefill TPS (per req, mean) | `1437.898` tok/s | 单请求 prefill 速度 (= input_tokens/TTFT) |
| Input TPS (system) | `204.519` tok/s | 整个系统每秒输入 token 数 |
| TTFT p99 | `1758.743` ms | 首 token 时延 (尾部) |
| 请求统计 | `64` ok / `0` fail | QPS=`0.399` |

## 性能摘要 (全局聚合)

### 吞吐 (system throughput, 跨并发聚合)

| metric | value | unit |
|---|---:|---|
| Output TPS | `51.13` | tokens/s |
| Input TPS | `204.519` | tokens/s |
| Total TPS (input+output) | `255.648` | tokens/s |
| QPS | `0.399` | req/s |

### Decode 速度 (per-request, 用户感受)

Decode TPS = `1000 / TPOT(ms)`，表示**单个请求**每秒能吐多少 token。

- 「**Decode TPS @ TPOT p50**」: 一半用户感受快于此值
- 「**Decode TPS @ TPOT p99**」: 99% 用户的最差体验（= 最慢请求的速度）

| metric | @ TPOT p50 | @ TPOT p99 |
|---|---:|---:|
| Decode TPS | `33.518` tok/s | `25.729` tok/s |
| TPOT | `29.835` ms | `38.866` ms |

### Prefill 速度 (per-request, 长上下文/RAG/Agent 关键)

Prefill TPS = `input_tokens / TTFT(s)`，表示**单个请求** prefill 阶段的吞吐。

| metric | mean | p50 | p99 |
|---|---:|---:|---:|
| Prefill TPS | `1437.898` tok/s | `1554.819` tok/s | `2710.191` tok/s |

### 时延分位 (per-request, ms)

| metric | p50 | p90 | p99 |
|---|---:|---:|---:|
| TTFT | `329.322` | `1200.776` | `1758.743` |
| TPOT | `29.835` | `36.503` | `38.866` |
| E2E | `4675.677` | `5227.437` | `5407.192` |

## 分 workload 明细

> workload key = `i<input_tokens>/o<output_tokens>/c<concurrency>`

### 吞吐 (按 workload)

![throughput_output_tokens_per_sec](images/throughput_output_tokens_per_sec.png)

![qps](images/qps.png)

![per_input_512_output_tps](images/per_input_512_output_tps.png)

![per_input_512_decode_tps_p50](images/per_input_512_decode_tps_p50.png)

![concurrency_trend](images/concurrency_trend.png)

![concurrency_dual](images/concurrency_dual.png)

| workload | Output TPS | Input TPS | QPS |
|---|---:|---:|---:|
| `i512/o128/c1` | `40.058` | `160.233` | `0.313` |
| `i512/o128/c4` | `100.499` | `401.996` | `0.785` |

### Decode / Prefill 速度 (按 workload)

| workload | Decode p50 | Decode p99 | Prefill mean | Prefill p99 |
|---|---:|---:|---:|---:|
| `i512/o128/c1` | `39.844` | `31.175` | `2112.006` | `2743.404` |
| `i512/o128/c4` | `30.519` | `25.416` | `763.79` | `1641.021` |

### 时延 (按 workload)

![latency_p99_ms](images/latency_p99_ms.png)

![tpot_p99_ms](images/tpot_p99_ms.png)

![latency_percentiles_ms](images/latency_percentiles_ms.png)

![per_input_512_ttft_p99](images/per_input_512_ttft_p99.png)

![per_input_512_e2e_p99](images/per_input_512_e2e_p99.png)

| workload | TTFT p50 | TTFT p90 | TTFT p99 | TPOT p50 | TPOT p90 | TPOT p99 | E2E p50 | E2E p90 | E2E p99 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `i512/o128/c1` | `245.679` | `299.514` | `330.386` | `25.098` | `31.24` | `32.077` | `3430.464` | `4264.596` | `4393.523` |
| `i512/o128/c4` | `915.987` | `1205.335` | `1759.116` | `32.766` | `37.55` | `39.346` | `5017.452` | `5336.964` | `5408.519` |

## GPU 与显存

采集到 `952` 条 GPU 采样，原始数据见 `metrics.gpu.jsonl`。

| metric | value | unit |
|---|---:|---|
| Util Avg | `38.416` | % |
| Util Max | `100` | % |
| Mem Avg | `14233.372` | MiB |
| Mem Peak | `23072` | MiB |
| Temp Avg | `68.524` | °C |
| Temp Max | `94` | °C |
| Power Avg | `164.399` | W |
| Power Max | `331.91` | W |

![gpu_utilization](images/gpu_utilization.png)

![gpu_memory](images/gpu_memory.png)

# 三、错误与建议

## 错误与离群点

### 错误概况

- failed_requests: `0`
- timeout_requests: `0`
- oom_count: `0`
- error_categories:
  - (无)

### 启动 / 全局错误

- 无

### 请求级离群点 (E2E 最慢的 5 条)

| request_id | e2e ms | ttft ms | tpot ms | input | output | concurrency |
|---|---:|---:|---:|---:|---:|---:|
| `req_000036` | `5409.8` | `1758.3` | `28.75` | `512` | `128` | `4` |
| `req_000035` | `5405.7` | `1759.5` | `28.71` | `512` | `128` | `4` |
| `req_000034` | `5378.4` | `565.3` | `37.90` | `512` | `128` | `4` |
| `req_000033` | `5348.6` | `292.7` | `39.81` | `512` | `128` | `4` |
| `req_000061` | `5232.2` | `366.6` | `38.31` | `512` | `128` | `4` |

## 优化建议

- GPU 平均利用率 `38.4%` 偏低 → 检查 batch / concurrency / 上游瓶颈。

# 四、名词解释

- **Output TPS (system)**: 整个推理系统每秒输出的 token 数，大模型领域主指标 (vLLM / SGLang / TRT-LLM 一致)。
- **Decode TPS (per req)**: 单个请求每秒能 decode 多少 token，= `1000 / TPOT(ms)`。用户感受到的「打字速度」。
- **Prefill TPS (per req)**: 单个请求 prefill 阶段的吞吐，= `input_tokens / TTFT(s)`。长上下文 / RAG / Agent 性能的关键。
- **Input TPS**: 整个系统每秒输入的 token 数。
- **TTFT (Time To First Token)**: 发出请求到收到第一个 token 的时延 (prefill + 排队)。
- **TPOT (Time Per Output Token)**: 第一个 token 之后，平均每个输出 token 的时间。
- **E2E**: 单请求端到端时延。
- **QPS**: 每秒成功请求数。在 LLM 场景下不如 Output TPS 直观 (单请求 token 数差异巨大)。
- 说明: `stream=true` 时 TTFT / TPOT 是真实测量；`stream=false` 时 TTFT≈E2E、TPOT 为摊还值。
