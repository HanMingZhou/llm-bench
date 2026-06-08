# llm-bench

<img src="logo.png" alt="llm-bench logo" width="180">

本地大模型推理与 NCCL 通信 benchmark CLI
----------------------------------------

- vLLM / SGLang：容器里跑什么命令，**完全由你在 `--` 后面写**
- transformers：参数名严格对齐 `AutoModelForCausalLM.from_pretrained` / `model.generate`，例如 `--torch-dtype` / `--device-map` / `--quantization` / `--trust-remote-code` / `--do-sample`

工具只负责：

- 起容器、转发端口、挂载 HF cache、注入 `HF_TOKEN`
- 跑 OpenAI 协议的压测客户端
- 采集 GPU 指标
- 生成 Markdown 报告、PNG 图表、JSONL 明细
- 历史结果对比 + CI 回归阈值检查

> 习惯了 `vllm serve --tensor-parallel-size 2 --gpu-memory-utilization 0.9 ...` 的写法？
> 这里参数直接照搬，所有 vLLM / SGLang 的参数都是它们自己的原始参数名，工具不重命名、不翻译。注意 `vllm/vllm-openai` 镜像的 entrypoint 已经是 `vllm serve`，所以 `--` 后面只贴模型路径和参数、**不要再写 `vllm serve`**（sglang 镜像没有这个 entrypoint，需带上 `python -m sglang.launch_server`）。

## QuickStart

最小示例（推理）：

```bash
llm-bench infer \
  --backend vllm \
  --image vllm/vllm-openai:latest \
  --model-name Qwen/Qwen2.5-7B-Instruct \
  --port 8000 \
  --workload-profile quick \
  --docker-arg=--gpus=all \
  --docker-arg=--shm-size=16g \
  --docker-arg=--ipc=host \
  -- \
  /root/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/<hash> \
    --host 0.0.0.0 --port 8000 \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 4096
```

最小示例（SGLang，镜像没有 server entrypoint，需带上完整启动器）：

```bash
llm-bench infer \
  --backend sglang \
  --image lmsysorg/sglang:latest \
  --model-name Qwen/Qwen2.5-7B-Instruct \
  --port 30000 \
  --workload-profile quick \
  --docker-arg=--gpus=all \
  --docker-arg=--shm-size=32g \
  --docker-arg=--ipc=host \
  -- \
  python3 -m sglang.launch_server \
    --model-path /root/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/<hash> \
    --host 0.0.0.0 --port 30000 \
    --tp 2
```

最小示例（transformers 本地后端，不走 docker）：

```bash
llm-bench infer \
  --backend transformers \
  --model-path /mnt/models/Qwen2.5-7B-Instruct \
  --torch-dtype bfloat16 \
  --device-map cuda:0 \
  --trust-remote-code \
  --workload-profile quick
```

参数名直接对齐 `AutoModelForCausalLM.from_pretrained` 和 `model.generate`，例如：

```bash
llm-bench infer \
  --backend transformers \
  --model-path /mnt/models/Qwen2.5-7B-Instruct \
  --torch-dtype bfloat16 \
  --device-map cuda:0 \
  --quantization 4bit \
  --trust-remote-code \
  --revision main \
  --do-sample \
  --temperature 0.7 --top-p 0.9 --top-k 50 \
  --repetition-penalty 1.1 \
  --num-beams 1 \
  --batch-size 2 \
  --total-requests 16 \
  --input-tokens 512 --output-tokens 128
```

transformers 后端在同进程内 `from_pretrained` + `generate`，不起 docker、不起 HTTP server，因此**不需要也不接受** `-- ...` 容器命令。

最小示例（NCCL all-reduce）：

```bash
llm-bench comm all-reduce \
  --image nccl-tests:latest \
  --docker-arg=--gpus=all \
  --docker-arg=--shm-size=16g \
  --docker-arg=--ipc=host \
  -- \
  /opt/nccl-tests/build/all_reduce_perf -b 8 -e 1G -f 2 -g 8 -n 100 -w 20
```

两个共同特点：

- `--` 前面是**工具自己的参数**（要起哪个镜像、压测客户端怎么发请求、报告写哪里）。
- `--` 后面是**真实的容器内命令**，和你直接在 host 上跑 `docker run ... <image> <这串命令>` 完全一样。

如果你以前在某个机器上手动用过 `all_reduce_perf` 这类命令，把它原样贴到 `--` 后面就行。vllm 要稍微注意：官方镜像 entrypoint 已经是 `vllm serve`，所以 `--` 后面只贴模型路径和参数，不要再写 `vllm serve`（否则会重复成 `vllm serve vllm serve ...`）。

## 安装

```bash
pip install .
llm-bench --help
```

或直接源码运行：

```bash
python -m llm_bench --help
```

打包成单文件二进制（PyInstaller）：

```bash
bash scripts/build_binary.sh
./dist/llm-bench --help
```

## 工具会做哪些 docker run 注入

下面是上面 vLLM 示例最终拼出来的实际 `docker run`：

```bash
docker run --rm \
  --name llm-bench-vllm-1717760000 \
  -p 8000:8000 \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e HF_HOME=/root/.cache/huggingface \
  --gpus all \
  --shm-size 16g \
  --ipc=host \
  vllm/vllm-openai:latest \
  /root/.cache/huggingface/hub/...snapshots/<hash> \
    --host 0.0.0.0 --port 8000 \
    --tensor-parallel-size 2 \
    --gpu-memory-utilization 0.9 \
    --max-model-len 4096
```

> 镜像名后面紧跟的就是你 `--` 后贴的内容。由于该镜像 entrypoint 是 `vllm serve`，容器内最终执行的是 `vllm serve /root/.cache/.../snapshots/<hash> --host 0.0.0.0 --port 8000 ...`——`vllm serve` 由镜像补上，所以你不能自己再写一遍。

注入项一览：


| 自动注入                                                    | 触发条件                                           |
| ----------------------------------------------------------- | -------------------------------------------------- |
| `--rm`                                                      | 默认开启，`--keep-container` 可关                  |
| `--name llm-bench-<backend>-<ts>`                           | 总是                                               |
| `-p <port>:<port>`                                          | `--port` 指定                                      |
| `-v <hf_cache>:/root/.cache/huggingface` + `-e HF_HOME=...` | `--hf-cache` 非空（默认是 `~/.cache/huggingface`） |
| `-e HF_TOKEN=...` + `-e HUGGING_FACE_HUB_TOKEN=...`         | `--hf-token` 或环境变量 `HF_TOKEN`                 |
| `--gpus` / `--shm-size` / `--ipc=host` / 其他               | 完全由`--docker-arg=...` 控制，工具不预设          |

容器名后面、镜像名后面那一长串，**就是你写在 `--` 后面的内容，原样、不修改**。

## 工具层 CLI 参数（推理）

```text
llm-bench infer
  --backend {vllm,sglang,transformers,dry-run}

  # 仅 vllm / sglang ---------------------------------------------------------
  --image IMAGE                     要启动的 docker 镜像
  --port PORT                       宿主机端口（也通过 -p 转发到容器同一端口）
  --model-name NAME                 OpenAI API 请求体里 model 字段，必填
  --docker-arg ARG                  额外 docker 参数，可重复；带 -- 前缀时写成 --docker-arg=...
  --startup-timeout SECONDS         等待容器内 /v1/models 健康，默认 900
  --keep-container                  不加 --rm，跑完不清理容器

  # 仅 transformers（参数名严格 = transformers 库参数名）---------------------
  --model-path PATH                 本地路径或 HF repo id，传给 from_pretrained
  --tokenizer-path PATH             默认 = --model-path
  --torch-dtype {float16,bfloat16,float32}
  --device-map STR                  auto / cuda:0 / cpu / ...
  --trust-remote-code / --no-...
  --revision REV
  --quantization Q                  4bit / int4 / nf4 / 8bit / int8 / awq / gptq
  --low-cpu-mem-usage / --no-...
  --do-sample / --no-do-sample
  --top-k N
  --repetition-penalty F
  --num-beams N
  --batch-size N                    transformers 内部 batch

  # 共用（HF cache、压测客户端、报告）---------------------------------------
  --hf-cache PATH                   挂载到容器 /root/.cache/huggingface（默认 ~/.cache/huggingface）
  --hf-token TOKEN                  作为 HF_TOKEN / HUGGING_FACE_HUB_TOKEN 注入
  --workload-profile {quick,standard,long-context,custom}
  --concurrency C1,C2,...
  --input-tokens N1,N2,...
  --output-tokens N1,N2,...
  --total-requests N
  --warmup-requests N
  --request-timeout SECONDS         单请求超时
  --api {completions,chat}          仅 vllm/sglang 生效
  --stream / --no-stream            仅 vllm/sglang 生效
  --temperature, --top-p, --seed
  --prompt-jsonl FILE               或者 --prompt-dir DIR
  --output-dir DIR                  报告输出目录
  --run-name NAME, --tag TAG
  --skip-env-check                  跳过预检（debug 用）

  -i, --interactive                 启动 wizard 流程
  --config CONFIG.yaml              从 YAML 加载默认值；CLI 覆盖 YAML

  -- ...                            真实容器内命令（仅 vllm / sglang 用）
```

注意：当你想在 `--docker-arg` 里传一个以 `--` 开头的值时，用 `=`：

```bash
--docker-arg=--shm-size=16g
--docker-arg=--ipc=host
--docker-arg=--gpus=all
```

`--docker-arg --shm-size --docker-arg 16g` 这种形式会被 argparse 误解。

## 工具层 CLI 参数（NCCL）

```text
llm-bench comm all-reduce
  --image IMAGE                     docker 镜像（默认 nccl-tests:latest，或自动从本机扫到的 nccl 镜像）
  --output-dir DIR
  --run-name NAME
  --timeout SECONDS                 整体超时，默认 1800
  --docker-arg ARG                  额外 docker 参数，可重复
  --dry-run                         只生成命令，不真的跑
  -i, --interactive                 启动 NCCL wizard

  -- ...                            真实容器内命令（all_reduce_perf -b ... -e ... -g ... ...）
```

## YAML 配置文件

CLI 参数也可以放进 YAML 复用：

```yaml
backend:
  name: vllm
  image: vllm/vllm-openai:latest
  port: 8000
  model_name: Qwen/Qwen2.5-7B-Instruct
  hf_cache: ~/.cache/huggingface
  docker_args:
    - --gpus
    - all
    - --shm-size
    - 16g
    - --ipc=host
  command:
    # 镜像 entrypoint 已是 vllm serve，这里只列模型路径和参数
    - /root/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/<hash>
    - --host
    - 0.0.0.0
    - --port
    - "8000"
    - --tensor-parallel-size
    - "2"
    - --gpu-memory-utilization
    - "0.9"
    - --max-model-len
    - "4096"

workload:
  profile: quick
  api: completions
  stream: false
  prompt_dir: examples/prompts

report:
  output_dir: benchmark_output/runs
```

跑：

```bash
llm-bench infer --config configs/inference.yaml
```

`backend.command` 字段就是真实 argv，不再有任何 `{model}` / `{tp}` 占位符。

## 交互式 wizard

不熟悉参数怎么写时，用 wizard：

```bash
llm-bench wizard
```

8 步：

1. 任务类型（推理 / NCCL）
2. 后端（vllm / sglang）
3. 镜像（从 `docker images` 自动列出）
4. **编辑容器内启动命令**（默认模板用真实参数名，可在编辑器里直接改）
5. `model-name` + `port`
6. HF cache + HF token
7. workload profile（quick / standard / long-context / custom）
8. summary（预览完整 docker run，确认开始）

所有步骤支持 `b` / `←` / Backspace 回退。

## 模型路径放哪里？

工具只负责把 HF cache 挂进去（`-v ~/.cache/huggingface:/root/.cache/huggingface`）。`--` 后面的模型路径就用容器内能看到的路径，比如：

```bash
-- /root/.cache/huggingface/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots/<hash>
```

或者，让 vllm 自己去 HF 解析仓库名（要保证 token 和 cache 已经挂进去）：

```bash
-- Qwen/Qwen2.5-7B-Instruct
```

任意你在直接用 `vllm serve` 时会写在它后面的路径形式，贴到 `--` 后面都成立。

## 报告产物

每次 `infer` 后会产生：

```text
benchmark_output/runs/<run_id>/
├── config.requested.yaml         你这次实际指定的字段
├── config.resolved.yaml          合并后生效的完整配置
├── run_manifest.json             run 元信息 + 汇总指标
├── environment.json              机器 / Docker / GPU 信息
├── metrics.summary.json          全局聚合 + 按 workload 分组
├── metrics.requests.jsonl        每个请求的明细
├── metrics.gpu.jsonl             GPU 采样
├── launch_plan.sh                实际跑的 docker 命令
├── logs/
│   └── backend.log               容器内 stdout+stderr 合并日志
└── reports/
    ├── inference_report.md
    └── images/*.png
```

## 历史 / 对比 / CI 阈值

```bash
llm-bench list                                          # 列出历史 run
llm-bench show <run_dir>                                # 看一次 run 的 manifest
llm-bench compare --baseline A --candidate B            # 写对比报告
llm-bench gate --baseline A --candidate B \             # CI 阈值检查
  --max-output-tps-drop-pct 5 \
  --max-e2e-p99-increase-pct 20
```

`gate` 在指标退化超过阈值时返回非零退出码，适合 CI。

`baseline set <run_dir>` 把某次 run 登记为该 (model, hardware, backend) 的基线，之后用 `--to-baseline` 自动匹配。

## 自检

如果你只想验证工具本身（不起 docker、不要 GPU、不需要模型），用 self-test：

```bash
llm-bench self-test --prompt-dir examples/prompts --concurrency 1 --total-requests 3
```

这会跑一个 dry-run 后端，产物结构和真实 run 一致，但所有指标是合成的。

## 日志保留与清理

```bash
llm-bench cleanup \
  --runs-dir benchmark_output/runs \
  --request-metrics-days 30 \
  --gpu-metrics-days 30 \
  --logs-days 14 \
  --no-dry-run
```

默认 `--dry-run` 只打印将要删除的文件，加 `--no-dry-run` 才真删。`run_manifest.json` / `config.*.yaml` / `metrics.summary.json` 等小文件**永远保留**，只清理大体积的 jsonl 和日志。

## 如何看报告里的指标

报告顶部有 **TL;DR** + **性能摘要** 两块，先看下面这几个最关键的：


| 指标                      | 怎么算                        | 看什么                                                                                                                                      |
| ------------------------- | ----------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| **Output TPS (system)**   | `Σ output_tokens / 真实墙钟` | LLM 压测**主指标**（vLLM/SGLang/TRT-LLM 一致）。多并发下比 QPS 直观得多——QPS=0.8 可能是 32 个 token，也可能是 2048 个 token，吞吐差几十倍 |
| **Decode TPS (per req)**  | `1000 / TPOT(ms)`             | 单个请求每秒能 decode 多少 token——用户感受到的「打字速度」。聊天框场景必看                                                                |
| **Prefill TPS (per req)** | `input_tokens / TTFT(s)`      | 单个请求 prefill 速度。长上下文 / RAG / Agent 场景的关键                                                                                    |
| **Input TPS (system)**    | `Σ input_tokens / 真实墙钟`  | 系统每秒能消化多少输入 token。RAG 这类 input ≫ output 的场景必看                                                                           |
| **TTFT p99**              | 首 token 到达时间的 99 分位   | 首字延迟的尾部体验                                                                                                                          |
| **TPOT p99**              | 单 token 间隔的 99 分位       | decode 阶段的尾部体验                                                                                                                       |
| **E2E p99**               | 单请求端到端时延 99 分位      | 完整响应时间的尾部                                                                                                                          |
| QPS                       | `请求数 / 真实墙钟`           | 仅对"按请求计费"或纯路由能力对比有意义；LLM 场景看 Output TPS 更准                                                                          |
| **busbw / algbw (NCCL)**  | nccl-tests 输出               | bus bandwidth =`algbw × 2(N-1)/N`（all-reduce ring）。看大消息（≥ 64MB）下稳定值是否接近物理带宽                                          |

**记忆要点**：

- **system**（全局）vs **per-request**（单请求）是两套口径，不要混着比
- 高并发下：system Output TPS **↑**，per-request Decode TPS **↓**（每个人等的久但总产出多）
- 报告里 GPU 利用率 / 显存 / 温度 / 功耗都有**数字汇总表**，不用看图估算
- 报告底部的"名词解释"段每次都会自动渲染

## 设计原则

1. **不翻译参数**：你写什么命令，容器里就跑什么命令。
2. **工具参数只描述工具自己关心的事**：image、port、HF cache、压测客户端、报告。
3. **不预设 docker flag**：`--shm-size` / `--ipc=host` / `--gpus all` 等都通过 `--docker-arg=...` 由你显式给。
4. **环境预检在前**：docker / image / port / GPU / 磁盘四项不通过就不启容器，避免无效等待。
5. **失败也归档**：预检失败也写 run 目录、报告，方便事后定位。

## 不支持的能力（明确说明）

- 不支持多机 NCCL 的 `mpirun` 自动编排。多机请在外部用 `mpirun` 调度，每个节点跑自己的 `llm-bench comm all-reduce`，再用 `compare` 汇总。
- 不会自动下载模型 / 镜像。`docker pull` / `huggingface-cli download` 提前在机器上准备好。
- transformers 后端需要本地安装 `torch + transformers`（`pip install torch transformers`），工具不会自动装。

## 开发

```bash
pip install -e '.[dev]'
pytest
```
