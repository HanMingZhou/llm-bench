"""Interactive wizard for `llm-bench wizard` and `llm-bench infer -i`.

Design goals:
- Few steps. Inference wizard is 8 steps total.
- No parameter translation. The container command is edited as raw argv,
  using the framework's own option names (e.g. `vllm serve --tensor-parallel-size`).
- Back navigation works everywhere. Each step can be undone with `b` / Left / Backspace.
- Tool-level concerns (image, port, model-name, hf cache, workload) are the
  only things the wizard collects.
"""
from __future__ import annotations

import argparse
import copy
import os
import re
import shlex
import shutil
import sys
import termios
import tty
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable

try:
    # Importing readline globally is enough to give input() arrow-key cursor
    # navigation, backspace at arbitrary positions, and Home/End editing. The
    # module is part of the Python stdlib on every Unix; on Windows it does not
    # exist, but this wizard already requires termios/tty (Unix only).
    import readline  # noqa: F401
except ImportError:
    pass

from llm_bench.backends.docker_serving import DockerServingBackend
from llm_bench.config import BenchConfig, PROFILES, apply_profile, default_backend_image, default_hf_cache
from llm_bench.environment import discover_docker_images, discover_model_paths


KEY_UP = "up"
KEY_DOWN = "down"
KEY_ENTER = "enter"
KEY_SPACE = "space"
KEY_BACK = "back"

TASK_INFER = "推理压测 (vLLM / SGLang)"
TASK_COMM = "NCCL all-reduce 通信压测"


@dataclass
class ParamSpec:
    """One CLI flag from the upstream framework (vllm / sglang).

    The `name` field is exactly the framework's own flag name (e.g.
    `--tensor-parallel-size`). The wizard never renames it.
    """

    name: str
    description: str
    default: str = ""
    kind: str = "str"  # str | int | float | bool | choice
    choices: list[str] = field(default_factory=list)
    category: str = "core"


# Sourced from `vllm serve --help=all` (and Red Hat AI Inference Server docs).
# Names are upstream-exact; the wizard does not translate them.
VLLM_PARAMS: list[ParamSpec] = [
    # core
    ParamSpec("--host", "服务监听地址（容器内）", "0.0.0.0", "str", category="core"),
    ParamSpec("--port", "服务端口（容器内）", "8000", "int", category="core"),
    ParamSpec("--served-model-name", "对外 API 暴露的模型名（不传时取 model-path 末段）", "", "str", category="core"),
    ParamSpec("--tokenizer", "tokenizer 路径或 HF id（不传则与模型一致）", "", "str", category="core"),
    # parallel
    ParamSpec("--tensor-parallel-size", "GPU 张量并行数（-tp）", "1", "int", category="parallel"),
    ParamSpec("--pipeline-parallel-size", "流水线并行（-pp）", "1", "int", category="parallel"),
    ParamSpec("--data-parallel-size", "数据并行副本数（-dp）", "1", "int", category="parallel"),
    ParamSpec(
        "--distributed-executor-backend",
        "分布式后端",
        "mp",
        "choice",
        choices=["mp", "ray"],
        category="parallel",
    ),
    ParamSpec("--enable-expert-parallel", "MoE 模型启用 expert parallel", "", "bool", category="parallel"),
    # memory
    ParamSpec("--gpu-memory-utilization", "GPU 显存利用率 0~1", "0.9", "float", category="memory"),
    ParamSpec("--max-model-len", "最大上下文 token 数", "4096", "int", category="memory"),
    ParamSpec("--swap-space", "CPU swap 空间 (GiB)", "4", "int", category="memory"),
    ParamSpec("--cpu-offload-gb", "权重 CPU offload (GiB)", "0", "int", category="memory"),
    ParamSpec(
        "--kv-cache-dtype",
        "KV cache 数据类型",
        "auto",
        "choice",
        choices=["auto", "fp8", "fp8_e5m2", "fp8_e4m3"],
        category="memory",
    ),
    ParamSpec("--block-size", "KV cache block size", "16", "int", category="memory"),
    # performance / scheduling
    ParamSpec("--enable-prefix-caching", "启用 prefix caching", "", "bool", category="performance"),
    ParamSpec("--enable-chunked-prefill", "启用 chunked prefill", "", "bool", category="performance"),
    ParamSpec("--enforce-eager", "强制 eager 模式（关闭 CUDA Graph）", "", "bool", category="performance"),
    ParamSpec("--max-num-batched-tokens", "单 iteration 最大 batched tokens", "8192", "int", category="performance"),
    ParamSpec("--max-num-seqs", "单 iteration 最大 sequences", "256", "int", category="performance"),
    ParamSpec("--max-seq-len-to-capture", "CUDA Graph 捕获的最大序列长度", "8192", "int", category="performance"),
    ParamSpec("--num-scheduler-steps", "scheduler 步数（multi-step decoding）", "1", "int", category="performance"),
    # model
    ParamSpec(
        "--dtype",
        "权重 / 激活精度",
        "auto",
        "choice",
        choices=["auto", "half", "float16", "bfloat16", "float", "float32"],
        category="model",
    ),
    ParamSpec(
        "--quantization",
        "量化方式（-q）",
        "",
        "choice",
        choices=["", "awq", "gptq", "fp8", "marlin", "gptq_marlin", "awq_marlin", "bitsandbytes", "modelopt"],
        category="model",
    ),
    ParamSpec("--trust-remote-code", "信任 HF 仓库里的自定义代码", "", "bool", category="model"),
    ParamSpec("--revision", "模型 revision (branch/tag/commit)", "main", "str", category="model"),
    ParamSpec("--tokenizer-mode", "tokenizer 模式", "auto", "choice", choices=["auto", "slow", "mistral"], category="model"),
    ParamSpec("--seed", "随机种子", "", "int", category="model"),
    # logging
    ParamSpec("--disable-log-requests", "关闭请求日志（高 QPS 推荐）", "", "bool", category="logging"),
    ParamSpec("--disable-log-stats", "关闭 stats 日志", "", "bool", category="logging"),
    # advanced
    ParamSpec("--enable-lora", "启用 LoRA", "", "bool", category="advanced"),
    ParamSpec("--max-loras", "最大 LoRA 数", "1", "int", category="advanced"),
    ParamSpec("--max-lora-rank", "最大 LoRA rank", "16", "int", category="advanced"),
    ParamSpec("--speculative-model", "推测解码 draft 模型", "", "str", category="advanced"),
    ParamSpec("--num-speculative-tokens", "推测 token 数", "5", "int", category="advanced"),
]

# Sourced from sglang main branch docs/advanced_features/server_arguments.md.
SGLANG_PARAMS: list[ParamSpec] = [
    # core
    ParamSpec("--host", "服务监听地址（容器内）", "0.0.0.0", "str", category="core"),
    ParamSpec("--port", "服务端口（容器内）", "30000", "int", category="core"),
    ParamSpec("--served-model-name", "对外 API 暴露的模型名", "", "str", category="core"),
    ParamSpec("--tokenizer-path", "tokenizer 路径", "", "str", category="core"),
    # parallel
    ParamSpec("--tp", "tensor parallel (--tensor-parallel-size)", "1", "int", category="parallel"),
    ParamSpec("--dp", "data parallel (--data-parallel-size)", "1", "int", category="parallel"),
    ParamSpec("--pp-size", "pipeline parallel (--pipeline-parallel-size)", "1", "int", category="parallel"),
    ParamSpec("--moe-dp-size", "MoE 数据并行", "1", "int", category="parallel"),
    ParamSpec("--nnodes", "多机节点总数", "1", "int", category="parallel"),
    ParamSpec("--node-rank", "本节点 rank", "0", "int", category="parallel"),
    # memory
    ParamSpec("--mem-fraction-static", "静态显存分配比例 0~1", "0.9", "float", category="memory"),
    ParamSpec("--context-length", "最大上下文 token", "4096", "int", category="memory"),
    ParamSpec("--max-running-requests", "最大并发请求数", "256", "int", category="memory"),
    ParamSpec("--max-total-tokens", "memory pool 最大 token 数（留空自动计算）", "", "int", category="memory"),
    ParamSpec("--max-prefill-tokens", "单 prefill 最大 token", "16384", "int", category="memory"),
    ParamSpec("--chunked-prefill-size", "chunked prefill 大小（-1 关闭）", "8192", "int", category="memory"),
    ParamSpec(
        "--kv-cache-dtype",
        "KV cache 数据类型",
        "auto",
        "choice",
        choices=["auto", "fp8_e5m2", "fp8_e4m3", "bf16", "bfloat16", "fp4_e2m1"],
        category="memory",
    ),
    ParamSpec("--page-size", "KV cache page 大小", "1", "int", category="memory"),
    # performance / scheduling
    ParamSpec(
        "--attention-backend",
        "Attention backend",
        "flashinfer",
        "choice",
        choices=["flashinfer", "fa3", "triton", "torch_native"],
        category="performance",
    ),
    ParamSpec(
        "--schedule-policy",
        "调度策略",
        "fcfs",
        "choice",
        choices=["fcfs", "lpm", "random", "dfs-weight", "lof", "priority", "routing-key"],
        category="performance",
    ),
    ParamSpec("--schedule-conservativeness", "调度保守度", "1.0", "float", category="performance"),
    ParamSpec("--enable-dynamic-chunking", "动态 chunking", "", "bool", category="performance"),
    ParamSpec("--enable-p2p-check", "TP 时跨卡 P2P 检查", "", "bool", category="performance"),
    ParamSpec("--enable-deterministic-inference", "可复现模式（性能下降）", "", "bool", category="performance"),
    # model
    ParamSpec(
        "--dtype",
        "权重精度",
        "auto",
        "choice",
        choices=["auto", "half", "float16", "bfloat16", "float", "float32"],
        category="model",
    ),
    ParamSpec(
        "--quantization",
        "量化方式",
        "",
        "choice",
        choices=["", "awq", "fp8", "gptq", "marlin", "gptq_marlin", "awq_marlin", "bitsandbytes", "compressed-tensors", "w8a8_int8", "w8a8_fp8", "mxfp4", "mxfp8"],
        category="model",
    ),
    ParamSpec("--trust-remote-code", "信任 HF 仓库自定义代码", "", "bool", category="model"),
    ParamSpec("--revision", "模型 revision", "main", "str", category="model"),
    ParamSpec("--random-seed", "随机种子", "", "int", category="model"),
    # advanced
    ParamSpec("--watchdog-timeout", "watchdog 超时秒", "300", "float", category="advanced"),
    ParamSpec("--enable-expert-parallel", "MoE expert parallel", "", "bool", category="advanced"),
]



# ---------------------------------------------------------------------------
# Public entry points
# ---------------------------------------------------------------------------


def run_unified_wizard(
    config: BenchConfig,
) -> tuple[str, BenchConfig | argparse.Namespace, dict[str, Any], bool]:
    _require_tty()
    print("llm-bench wizard")
    print("使用 ↑/↓ 选择，Enter 确认，b / ← / Backspace 返回。")
    print()
    task = _select("选择任务类型", [TASK_INFER, TASK_COMM], TASK_INFER)
    if task == TASK_INFER:
        cfg, requested, should_start = run_infer_wizard(config)
        return "infer", cfg, requested, should_start
    args = run_comm_wizard()
    return "comm", args, {}, True


def run_infer_wizard(config: BenchConfig) -> tuple[BenchConfig, dict[str, Any], bool]:
    _require_tty()
    state = _InferWizard(config)
    return state.run()


def run_comm_wizard(args: argparse.Namespace | None = None) -> argparse.Namespace:
    _require_tty()
    if args is None:
        args = _default_comm_args()
    state = _CommWizard(args)
    return state.run()


# ---------------------------------------------------------------------------
# Inference wizard
# ---------------------------------------------------------------------------


class _InferWizard:
    def __init__(self, config: BenchConfig) -> None:
        self.config = config
        self.requested: dict[str, Any] = {}
        self.should_start = False

    def run(self) -> tuple[BenchConfig, dict[str, Any], bool]:
        steps = self._steps_for(self.config.backend.name)
        snapshots: list[tuple[BenchConfig, dict[str, Any]]] = []
        idx = 0
        while idx < len(steps):
            snapshots = snapshots[:idx]
            snapshots.append((copy.deepcopy(self.config), copy.deepcopy(self.requested)))
            try:
                steps[idx]()
                idx += 1
                steps = self._steps_for(self.config.backend.name)
            except _BackRequested:
                if idx == 0:
                    print("已经是第一步。")
                    continue
                self.config, self.requested = snapshots[idx - 1]
                steps = self._steps_for(self.config.backend.name)
                idx -= 1
                # Make it obvious where the "re-do" output starts so that the
                # screen does not look like a duplicate of the prior attempt.
                print()
                print("────────── ↩ 返回上一步，重新填写 ──────────")
                print()
        return self.config, self.requested, self.should_start

    def _steps_for(self, backend: str) -> list[Callable[[], None]]:
        if backend == "transformers":
            return [
                self._step_backend,
                self._step_transformers,
                self._step_profile_and_workload,
                self._step_prompt_and_api,
                self._step_summary,
            ]
        steps: list[Callable[[], None]] = [
            self._step_backend,
            self._step_image,
            # _step_command used to do everything (pick model + mount + sglang
            # launcher + select params + ask values + final edit) in one big
            # step, which meant pressing `b` from a later step always rewound
            # all of it. Split into three so back-button granularity matches
            # what users intuitively call "previous step".
            self._step_pick_model,
            self._step_pick_params,
            self._step_finalize_command,
            self._step_docker_args,
            self._step_model_and_port,
        ]
        # Only include the HF cache / token step when it is actually needed
        # (i.e. the user picked an HF id rather than a bind-mounted local /
        # ModelScope path). This avoids a self-skipping step that turns into a
        # back-button trap.
        if not self._uses_local_model():
            steps.append(self._step_hf)
        steps.extend([
            self._step_profile_and_workload,
            self._step_prompt_and_api,
            self._step_summary,
        ])
        return steps

    def _step_backend(self) -> None:
        choices = ["vllm", "sglang", "transformers"]
        default = self.config.backend.name if self.config.backend.name in choices else "vllm"
        backend = _select(
            _title("选择后端", "(vllm/sglang 走 docker + OpenAI HTTP；transformers 直接调 from_pretrained)"),
            choices,
            default,
        )
        self.config.backend.name = backend
        if backend in {"vllm", "sglang"} and not self.config.backend.image:
            self.config.backend.image = default_backend_image(backend)
        self._set_nested("backend", "name", backend)

    def _step_transformers(self) -> None:
        tx = self.config.transformers
        # Reuse the same model scanner the docker backends use, so transformers
        # users can also pick from HF / ModelScope cache instead of typing the
        # full path. _pick_model returns (path_or_id_for_command, hf_id).
        model_path, _hf_id = self._pick_model_for_transformers(tx.model_path)
        tx.model_path = model_path
        self._set_nested("transformers", "model_path", tx.model_path)
        tx.tokenizer_path = _text(
            "tokenizer_path（可留空，默认 = model_path）",
            tx.tokenizer_path,
        )
        if tx.tokenizer_path:
            self._set_nested("transformers", "tokenizer_path", tx.tokenizer_path)
        tx.torch_dtype = _select(
            "torch_dtype",
            ["bfloat16", "float16", "float32"],
            tx.torch_dtype,
        )
        self._set_nested("transformers", "torch_dtype", tx.torch_dtype)
        tx.device_map = _text("device_map (auto / cuda:0 / cpu)", tx.device_map or "auto")
        self._set_nested("transformers", "device_map", tx.device_map)
        if "cpu" in tx.device_map.lower():
            print("device_map 含 cpu，将跳过 GPU 预检；推理性能会显著低于 GPU。")
        quant_choices = ["(none)", "4bit", "8bit", "awq", "gptq"]
        quant_default = tx.quantization or "(none)"
        quant = _select("quantization", quant_choices, quant_default if quant_default in quant_choices else "(none)")
        tx.quantization = "" if quant == "(none)" else quant
        self._set_nested("transformers", "quantization", tx.quantization)
        tx.trust_remote_code = _select("trust_remote_code", ["No", "Yes"], "Yes" if tx.trust_remote_code else "No") == "Yes"
        self._set_nested("transformers", "trust_remote_code", tx.trust_remote_code)
        # Loop instead of raising _BackRequested on bad input: bouncing back
        # to _step_backend after a typo throws away everything chosen above.
        while True:
            raw = _text("batch_size (transformers 内部 batch)", str(tx.batch_size))
            try:
                tx.batch_size = int(raw)
                break
            except ValueError:
                print("batch_size 必须是整数，请重新输入（或 :back 回上一步）。")
        self._set_nested("transformers", "batch_size", tx.batch_size)

    def _pick_model_for_transformers(self, previous: str) -> tuple[str, str]:
        """Same flow as _pick_model but no docker mount side-effects.

        Returns (path_for_from_pretrained, hf_id_or_empty). For HF cache hits
        we use the HF id (from_pretrained resolves from the cache); for
        ModelScope / Local we use the absolute host path since transformers
        runs in-process and does not need a bind mount.
        """
        prompt_label = "model_path（from_pretrained 第一个位置参数）"
        try:
            discovered = discover_model_paths(limit=30)
        except Exception:
            discovered = []
        manual_label = "手动输入"
        if not discovered:
            while True:
                value = _text(prompt_label, previous)
                if value:
                    return value, ""
                print("model_path 不能为空，请重新输入（或输入 :back 返回上一步）。")
        labels = [f"{item['source']:<13} {item['name']}" for item in discovered]
        labels.append(manual_label)
        default = labels[0]
        if previous:
            for idx, item in enumerate(discovered):
                if item["name"].startswith(previous) or item["path"] == previous:
                    default = labels[idx]
                    break
        choice = _select(
            _title("选择模型", "(扫描自 HF cache / ModelScope cache / /mnt/models 等)"),
            labels,
            default,
        )
        if choice != manual_label:
            item = discovered[labels.index(choice)]
            hf_id = item["name"].split("@", 1)[0]
            # For HF entries we hand the HF id to from_pretrained; ModelScope
            # / Local give the absolute path since the file is already there.
            if item["source"] == "Hugging Face":
                return hf_id, hf_id
            return item["path"], hf_id if "/" in hf_id else ""
        while True:
            value = _text(prompt_label, previous)
            if value:
                return value, ""
            print("model_path 不能为空，请重新输入（或输入 :back 返回上一步）。")

    def _step_image(self) -> None:
        backend = self.config.backend.name
        all_images = discover_docker_images(backend)
        relevant = _filter_images_for_backend(all_images, backend)
        # If a backend-relevant image exists locally, only show those + manual.
        # Otherwise fall back to the full list so the user still has options.
        discovered = relevant or all_images
        manual_label = "手动输入镜像"

        if not discovered:
            print(f"未在本机扫到 {backend} 镜像，请输入要使用的镜像名。")
            while True:
                image = _text(_title("Docker 镜像"), "")
                if image:
                    break
                print("镜像名不能为空，请重新输入（或输入 :back 返回上一步）。")
            self.config.backend.image = image
            self._set_nested("backend", "image", self.config.backend.image)
            return

        labels = [_image_label(item) for item in discovered]
        labels.append(manual_label)
        default = labels[0]
        if self.config.backend.image:
            for idx, item in enumerate(discovered):
                if item["name"] == self.config.backend.image:
                    default = labels[idx]
                    break
        subtitle = None if relevant else f"(本机未识别到 {backend} 相关镜像，下面列出全部本地镜像供参考)"
        title = _title("选择 Docker 镜像", subtitle) if subtitle else "选择 Docker 镜像"
        choice = _select(title, labels, default)
        if choice == manual_label:
            self.config.backend.image = _text("Docker 镜像", self.config.backend.image)
        else:
            self.config.backend.image = discovered[labels.index(choice)]["name"]
        self._set_nested("backend", "image", self.config.backend.image)

    # _step_command is intentionally not on the step list any more (kept only
    # for backwards-compatible tests / monkeypatch); the real flow is the
    # three split steps below.
    def _step_command(self) -> None:  # pragma: no cover - back-compat shim
        self._step_pick_model()
        self._step_pick_params()
        self._step_finalize_command()

    def _step_pick_model(self) -> None:
        """Sub-step 1/3: pick a local/HF model + launcher choice."""
        backend = self.config.backend.name
        model_path, hf_id = self._pick_model(backend)
        if hf_id:
            self.config.backend.model_name = self.config.backend.model_name or hf_id
            self._set_nested("backend", "model_name", self.config.backend.model_name)
        self._set_nested("backend", "command_model_path", model_path)

        # Let the user pick the launcher subcommand explicitly.
        if backend == "vllm":
            previous = self._vllm_launcher_from_command()
            default_label = next(
                (label for label, argv in VLLM_LAUNCHERS.items() if argv == previous),
                VLLM_DEFAULT_LAUNCHER,
            )
            choice = _select(_title("vllm 启动器"), list(VLLM_LAUNCHERS.keys()), default_label)
            self._set_nested("backend", "vllm_launcher_label", choice)
        elif backend == "sglang":
            previous = self._sglang_launcher_from_command()
            default_label = next(
                (label for label, argv in SGLANG_LAUNCHERS.items() if argv == previous),
                SGLANG_DEFAULT_LAUNCHER,
            )
            choice = _select(_title("sglang 启动器"), list(SGLANG_LAUNCHERS.keys()), default_label)
            self._set_nested("backend", "sglang_launcher_label", choice)

    def _step_pick_params(self) -> None:
        """Sub-step 2/3: select startup params + ask values per chosen flag."""
        backend = self.config.backend.name
        catalog = VLLM_PARAMS if backend == "vllm" else SGLANG_PARAMS

        # Recover previously-chosen keys when the user comes back from a later
        # step, so the menu pre-checks them instead of forcing a blank restart.
        selected_keys: set[str] = set()
        for tok in self.config.backend.command:
            if tok.startswith("--") and any(p.name == tok for p in catalog):
                selected_keys.add(tok)

        # Recover previously-typed values so re-entering this step keeps them.
        previous_values: dict[str, str | None] = {}
        cmd = self.config.backend.command
        for i, tok in enumerate(cmd):
            if not tok.startswith("--"):
                continue
            if i + 1 < len(cmd) and not cmd[i + 1].startswith("--"):
                previous_values[tok] = cmd[i + 1]
            else:
                previous_values[tok] = "FLAG"

        chosen_params = _select_params(catalog, selected_keys)
        values = _ask_param_values(chosen_params, previous_values)
        # Stash the in-progress selection on the wizard instance so the
        # finalize step can pick it up. Using attributes (not setattr-on-cfg)
        # keeps the on-disk config clean.
        self._chosen_params = chosen_params  # type: ignore[attr-defined]
        self._param_values = values  # type: ignore[attr-defined]

    def _step_finalize_command(self) -> None:
        """Sub-step 3/3: show the assembled command, optionally edit it."""
        backend = self.config.backend.name
        chosen_params = getattr(self, "_chosen_params", [])
        values = getattr(self, "_param_values", {})

        vllm_launcher: list[str] | None = None
        sglang_launcher: list[str] | None = None
        if backend == "vllm":
            label = self.requested.get("backend", {}).get("vllm_launcher_label") or VLLM_DEFAULT_LAUNCHER
            vllm_launcher = VLLM_LAUNCHERS.get(label, VLLM_LAUNCHERS[VLLM_DEFAULT_LAUNCHER])
        elif backend == "sglang":
            label = self.requested.get("backend", {}).get("sglang_launcher_label") or SGLANG_DEFAULT_LAUNCHER
            sglang_launcher = SGLANG_LAUNCHERS.get(label, SGLANG_LAUNCHERS[SGLANG_DEFAULT_LAUNCHER])

        model_path = self.requested.get("backend", {}).get("command_model_path") or self._guess_model_path()
        while True:
            command = _build_command(backend, model_path, chosen_params, values, vllm_launcher=vllm_launcher, sglang_launcher=sglang_launcher)
            _print_param_table(chosen_params, values)
            print(f"容器启动命令: {shlex.join(command)}")
            action = _select(
                "下一步",
                ["使用此命令", "进入编辑器微调", "重新勾选参数"],
                "使用此命令",
            )
            if action == "重新勾选参数":
                # In-step retry rather than _BackRequested: keep model path /
                # sglang launcher selection from the previous sub-step intact.
                chosen_params = _select_params(VLLM_PARAMS if backend == "vllm" else SGLANG_PARAMS,
                                               {p.name for p in chosen_params})
                values = _ask_param_values(chosen_params, values)
                self._chosen_params = chosen_params  # type: ignore[attr-defined]
                self._param_values = values  # type: ignore[attr-defined]
                continue
            if action.startswith("进入编辑器"):
                edited = _editable("command> ", shlex.join(command))
                try:
                    command = shlex.split(edited.strip()) if edited.strip() else command
                except ValueError as exc:
                    print(f"命令解析失败：{exc}，保留勾选拼出的版本。")
            self.config.backend.command = command
            self._set_nested("backend", "command", command)
            return

    def _guess_model_path(self) -> str:
        """Best-effort guess for the inside-container model path the user wants."""
        if self.config.backend.command:
            for token in self.config.backend.command:
                if token.startswith("/") or "/" in token:
                    if token.startswith("--"):
                        continue
                    return token
        if self.config.backend.model_name:
            return self.config.backend.model_name
        return ""

    def _pick_model(self, backend: str) -> tuple[str, str]:
        """Let the user choose a model and return (container_path, hf_id_or_empty).

        Scans local HF / ModelScope caches and common model dirs.

        Container path resolution depends on the model source:
        - Hugging Face: use the HF id directly (e.g. Qwen/Qwen2.5-7B-Instruct).
          vllm / sglang resolves it from the mounted HF cache, so we don't
          have to know the snapshot hash.
        - ModelScope / Local: use the host absolute path AND warn the user
          that this path must be visible inside the container (the wizard
          only mounts the HF cache by default; add `--docker-arg=-v=...`).
        - Manual: caller types whatever container-visible path they want.
        """
        prompt_label = "模型路径" if backend == "vllm" else "模型路径（--model-path）"
        previous = self._guess_model_path()
        try:
            discovered = discover_model_paths(limit=30)
        except Exception:
            discovered = []

        manual_label = "手动输入容器内路径"
        if not discovered:
            print("未在本机扫到模型。已搜索 HF cache、ModelScope cache、/models、/mnt/models、/data/models 以及 $MODELSCOPE_CACHE / $LLM_BENCH_MODEL_DIRS。")
            while True:
                value = _text(prompt_label, previous)
                if value:
                    return value, ""
                print("模型路径不能为空，请重新输入（或输入 :back 返回上一步）。")

        labels = [f"{item['source']:<13} {item['name']}" for item in discovered]
        labels.append(manual_label)
        default = labels[0]
        if previous:
            for idx, item in enumerate(discovered):
                if item["name"].startswith(previous) or item["path"] == previous:
                    default = labels[idx]
                    break
        choice = _select(
            _title("选择模型", "(扫描自 HF cache / ModelScope cache / /mnt/models 等)"),
            labels,
            default,
        )
        if choice != manual_label:
            item = discovered[labels.index(choice)]
            source = item["source"]
            hf_id = item["name"].split("@", 1)[0]
            if source == "Hugging Face":
                # HF cache is mounted into the container by default; the HF id
                # is enough for vllm / sglang to find the weights.
                return hf_id, hf_id
            # ModelScope / Local: let the user edit how the host directory is
            # bind-mounted into the container. Default is host:host (same path
            # both sides) but anything in `host_dir:container_dir[:ro]` form is
            # accepted. The model's container path is derived from that edit.
            host_dir, default_container_path = _container_mount_for(item)
            default_mount = f"{host_dir}:{host_dir}"
            print(_title("配置挂载", f"({source} 模型: {item['path']})"))
            print("格式 host:container[:ro]；container 即容器内挂到的目录。")
            mount_entry = _editable_simple("挂载 (host:container)", default_mount)
            container_dir = _split_container_dir(mount_entry, host_dir)
            container_path = _rebase_path(item["path"], host_dir, container_dir) or default_container_path
            if mount_entry not in self.config.backend.extra_mounts:
                self.config.backend.extra_mounts.append(mount_entry)
                self._set_nested("backend", "extra_mounts", list(self.config.backend.extra_mounts))
                print(f"挂载: -v {mount_entry}   容器内模型路径: {container_path}")
            return container_path, hf_id if "/" in hf_id else ""

        while True:
            value = _text(prompt_label, previous)
            if value:
                return value, ""
            print("模型路径不能为空，请重新输入（或输入 :back 返回上一步）。")

    def _step_model_and_port(self) -> None:
        # `--served-model-name` (vllm/sglang startup flag) and `model_name`
        # (the OpenAI client's request field) MUST be the same string or the
        # server returns 404. Pre-fill the default in priority order:
        #   1. whatever the user already set with `--served-model-name`
        #      during parameter selection (single source of truth, no surprise)
        #   2. the auto-derived name (HF id as-is / last path segment)
        # Whatever the user types here is mirrored back to the container
        # command so both ends stay in sync.
        default = (
            self._served_model_name_from_command()
            or self.config.backend.model_name
            or self._auto_model_name()
        )
        # Loop on empty model_name instead of raising _BackRequested - the
        # latter throws away everything chosen in earlier steps.
        while True:
            model_name = _text("model_name", default)
            if model_name:
                break
            print("model_name 不能为空，请重新输入（或 :back 回上一步）。")
        self.config.backend.model_name = model_name
        self._set_nested("backend", "model_name", model_name)
        self._ensure_served_model_name(model_name)

        # Default the host port to whatever the container actually listens on,
        # so docker -p X:X matches the framework's --port. Otherwise the user
        # could end up with e.g. -p 8000:8000 while sglang listens on 30000
        # (its default) and every HTTP request 404s.
        container_port = self._container_listen_port()
        default_port = str(container_port or self.config.backend.port or 8000)
        # Loop on bad port input too; previously a typo bounced the user back
        # to _step_docker_args and forced them to redo every prior choice.
        while True:
            port_str = _text(
                "宿主机端口（同时也会作为 -p 容器端口转发，需与容器内监听端口一致）",
                default_port,
            )
            try:
                port = int(port_str)
                break
            except ValueError:
                print("端口必须是整数，请重新输入（或 :back 回上一步）。")
        self.config.backend.port = port
        # Keep --port in the container command in sync as well: changing the
        # host-side port now also rewrites the framework's --port arg so the
        # two values can never drift apart again.
        self._sync_container_port(port)
        self._set_nested("backend", "port", port)

    def _vllm_launcher_from_command(self) -> list[str] | None:
        """Detect which vllm launcher prefix the current command uses, if any."""
        cmd = self.config.backend.command
        for argv in VLLM_LAUNCHERS.values():
            if argv and list(cmd[: len(argv)]) == argv:
                return argv
        return None

    def _sglang_launcher_from_command(self) -> list[str] | None:
        """Detect which sglang launcher prefix the current command uses, if any."""
        cmd = self.config.backend.command
        for argv in SGLANG_LAUNCHERS.values():
            if list(cmd[: len(argv)]) == argv:
                return argv
        return None

    def _container_listen_port(self) -> int | None:
        """Read the --port value the framework will listen on inside the container."""
        cmd = self.config.backend.command
        try:
            idx = cmd.index("--port")
        except ValueError:
            return None
        if idx + 1 >= len(cmd):
            return None
        try:
            return int(cmd[idx + 1])
        except (TypeError, ValueError):
            return None

    def _sync_container_port(self, port: int) -> None:
        """Force the container command's --port to match the host-side port."""
        cmd = list(self.config.backend.command)
        try:
            idx = cmd.index("--port")
        except ValueError:
            return
        if idx + 1 >= len(cmd):
            return
        if cmd[idx + 1] == str(port):
            return
        old = cmd[idx + 1]
        cmd[idx + 1] = str(port)
        self.config.backend.command = cmd
        self._set_nested("backend", "command", cmd)
        print(f"已对齐容器内 --port: {old} -> {port}  (与宿主机端口转发一致)")

    def _served_model_name_from_command(self) -> str:
        """If `--served-model-name` is already in the container command, use its value."""
        cmd = self.config.backend.command
        try:
            idx = cmd.index("--served-model-name")
        except ValueError:
            return ""
        return cmd[idx + 1] if idx + 1 < len(cmd) else ""

    def _auto_model_name(self) -> str:
        """Derive an OpenAI-API-friendly model name from the container command."""
        cmd = list(self.config.backend.command)
        backend = self.config.backend.name
        if backend == "vllm":
            model_in_cmd = next((t for t in cmd if not t.startswith("--")), None)
        elif backend == "sglang":
            try:
                model_in_cmd = cmd[cmd.index("--model-path") + 1]
            except (ValueError, IndexError):
                model_in_cmd = None
        else:
            return self.config.backend.model_name or ""
        if not model_in_cmd:
            return self.config.backend.model_name or ""
        if model_in_cmd.startswith("/"):
            # Absolute container path -> use the last directory segment.
            return model_in_cmd.rstrip("/").rsplit("/", 1)[-1] or model_in_cmd
        # HF id-style; reuse as-is.
        return model_in_cmd

    def _ensure_served_model_name(self, model_name: str) -> None:
        cmd = list(self.config.backend.command)
        if not cmd:
            return
        # First non-flag token is the model path (vllm) or follows --model-path (sglang).
        backend = self.config.backend.name
        if backend == "vllm":
            model_in_cmd = next((t for t in cmd if not t.startswith("--")), None)
            served_flag = "--served-model-name"
        elif backend == "sglang":
            try:
                model_in_cmd = cmd[cmd.index("--model-path") + 1]
            except (ValueError, IndexError):
                model_in_cmd = None
            served_flag = "--served-model-name"
        else:
            return

        if model_in_cmd == model_name:
            return
        if served_flag in cmd:
            idx = cmd.index(served_flag)
            if idx + 1 < len(cmd) and cmd[idx + 1] == model_name:
                return
            cmd[idx + 1] = model_name
        else:
            cmd.extend([served_flag, model_name])
        self.config.backend.command = cmd
        self._set_nested("backend", "command", cmd)
        print(f"自动追加: {served_flag} {model_name}  (容器内模型路径 {model_in_cmd} 与 API model_name 不一致)")

    def _step_hf(self) -> None:
        hf_cache = _text(
            _title("HuggingFace cache 目录", "(挂载到容器内 /root/.cache/huggingface)"),
            self.config.backend.hf_cache or default_hf_cache(),
        )
        self.config.backend.hf_cache = hf_cache
        if hf_cache:
            self._set_nested("backend", "hf_cache", hf_cache)
        default_token = self.config.backend.hf_token or os.environ.get("HF_TOKEN") or ""
        hf_token = _text(
            "HuggingFace token（可留空，私有/受限模型才需要）",
            default_token,
        )
        if hf_token:
            self.config.backend.hf_token = hf_token
            self._set_nested("backend", "hf_token", hf_token)

    def _step_docker_args(self) -> None:
        # vllm / sglang need at minimum --gpus to see the GPU; --shm-size and
        # --ipc=host are standard for distributed inference. Pre-fill these
        # defaults so a clean wizard run can actually start the container.
        current = " ".join(self.config.backend.docker_args) if self.config.backend.docker_args else (
            "--gpus all --shm-size 16g --ipc=host"
        )
        print(_title("docker 参数", "(注入到 docker run；--gpus all 是 GPU 推理必需)"))
        text = _editable_simple("docker args (空格分隔)", current)
        args = shlex.split(text) if text.strip() else []
        self.config.backend.docker_args = args
        self._set_nested("backend", "docker_args", args)

    def _uses_local_model(self) -> bool:
        """True iff the container's model path is a bind-mounted absolute path.

        The "model path" lives in different positions for each backend:
          - vllm: positional, first non-flag token (e.g. `vllm serve <path>`).
          - sglang: value after `--model-path` (the rest are launcher args).
        Picking the first non-flag token for sglang would pick `python` and
        wrongly conclude the user is on HF id - reintroducing the HF cache
        / token wizard step. Always honour the explicit --model-path when set.
        """
        cmd = self.config.backend.command
        try:
            idx = cmd.index("--model-path")
            if idx + 1 < len(cmd):
                return cmd[idx + 1].startswith("/")
        except ValueError:
            pass
        for token in cmd:
            if token.startswith("--"):
                continue
            if token in _SGLANG_LAUNCHER_TOKENS or token in _VLLM_LAUNCHER_TOKENS:
                # launcher prefix tokens, not the model path itself.
                continue
            return token.startswith("/")
        return False

    def _step_profile_and_workload(self) -> None:
        # Show what each preset profile actually does so the user does not pick
        # blind. Labels are "<name>  concurrency=[..]  i=[..]  o=[..]  total=N".
        label_map = {name: _format_profile_label(name) for name in ("quick", "standard", "long-context")}
        label_map["custom"] = "custom  自定义所有参数"
        labels = list(label_map.values())
        current = self.config.workload.profile if self.config.workload.profile in label_map else "quick"
        choice = _select(_title("选择压测 profile"), labels, label_map[current])
        # Recover the canonical profile name from the displayed label.
        profile = next(name for name, lbl in label_map.items() if lbl == choice)
        if profile != "custom":
            apply_profile(self.config, profile)
        else:
            self.config.workload.profile = "custom"
        self._set_nested("workload", "profile", profile)
        # Always offer to override workload knobs. For presets the user can
        # press Enter to accept; for custom there is no preset to fall back on.
        self._override_workload(prompt_prefix="覆盖 " if profile != "custom" else "")

    def _override_workload(self, prompt_prefix: str = "") -> None:
        concurrency = _multi_int(
            f"{prompt_prefix}并发度（空格多选，Enter 确认）",
            [1, 2, 4, 8, 16, 32, 64],
            self.config.workload.concurrency or [1],
        )
        if concurrency:
            self.config.workload.concurrency = concurrency
            self._set_nested("workload", "concurrency", concurrency)
        input_tokens = _multi_int(
            f"{prompt_prefix}input tokens",
            [128, 512, 1024, 2048, 4096, 8192, 16384],
            self.config.workload.input_tokens or [512],
        )
        if input_tokens:
            self.config.workload.input_tokens = input_tokens
            self._set_nested("workload", "input_tokens", input_tokens)
        output_tokens = _multi_int(
            f"{prompt_prefix}output tokens",
            [32, 64, 128, 256, 512],
            self.config.workload.output_tokens or [128],
        )
        if output_tokens:
            self.config.workload.output_tokens = output_tokens
            self._set_nested("workload", "output_tokens", output_tokens)
        try:
            total_requests = int(_text(f"{prompt_prefix}total requests", str(self.config.workload.total_requests)))
        except ValueError:
            print("total requests 必须是整数，保留默认。")
            total_requests = self.config.workload.total_requests
        self.config.workload.total_requests = total_requests
        self._set_nested("workload", "total_requests", total_requests)

    def _step_prompt_and_api(self) -> None:
        api = _select(
            _title("OpenAI API 类型"),
            ["completions", "chat"],
            self.config.workload.api,
        )
        self.config.workload.api = api
        self._set_nested("workload", "api", api)
        stream = _select(
            "是否启用 streaming（用 SSE 获取真实 TTFT）",
            ["No", "Yes"],
            "Yes" if self.config.workload.stream else "No",
        )
        self.config.workload.stream = stream == "Yes"
        self._set_nested("workload", "stream", self.config.workload.stream)
        source = _select(
            "Prompt 来源",
            ["synthetic", "prompt-dir", "jsonl"],
            "jsonl" if self.config.workload.prompt_jsonl
            else ("prompt-dir" if self.config.workload.prompt_dir else "synthetic"),
        )
        if source == "synthetic":
            self.config.workload.prompt_jsonl = ""
            self.config.workload.prompt_dir = ""
            self.config.workload.mode = "fixed"
        elif source == "prompt-dir":
            prompt_dir = _text("prompt 目录", self.config.workload.prompt_dir or "examples/prompts")
            self.config.workload.prompt_dir = prompt_dir
            self.config.workload.prompt_jsonl = ""
            self.config.workload.mode = "prompt-dir"
            self._set_nested("workload", "prompt_dir", prompt_dir)
        else:
            prompt_jsonl = _text("prompt JSONL", self.config.workload.prompt_jsonl or "examples/workload.jsonl")
            self.config.workload.prompt_jsonl = prompt_jsonl
            self.config.workload.prompt_dir = ""
            self.config.workload.mode = "jsonl"
            self._set_nested("workload", "prompt_jsonl", prompt_jsonl)
        self._set_nested("workload", "mode", self.config.workload.mode)

    def _step_summary(self) -> None:
        backend_name = self.config.backend.name
        if backend_name in {"vllm", "sglang"}:
            preview = DockerServingBackend(backend_name).preview_command(self.config)
            backend_lines = [
                f"backend         : {backend_name}",
                f"image           : {self.config.backend.image}",
                f"port            : {self.config.backend.port}",
                f"model_name      : {self.config.backend.model_name}",
                f"hf_cache        : {self.config.backend.hf_cache}",
                f"hf_token        : {'<set>' if self.config.backend.hf_token else '<unset>'}",
                f"extra_mounts    : {', '.join(self.config.backend.extra_mounts) or '(none)'}",
                f"command         : {shlex.join(self.config.backend.command)}",
            ]
            preview_block = ["完整 docker 启动命令：", f"  {shlex.join(preview)}", ""]
        else:
            tx = self.config.transformers
            backend_lines = [
                f"backend         : {backend_name}",
                f"model_path      : {tx.model_path}",
                f"tokenizer_path  : {tx.tokenizer_path or '(= model_path)'}",
                f"torch_dtype     : {tx.torch_dtype}",
                f"device_map      : {tx.device_map}",
                f"quantization    : {tx.quantization or '(none)'}",
                f"trust_remote_code: {tx.trust_remote_code}",
                f"batch_size      : {tx.batch_size}",
            ]
            preview_block = []

        lines = [
            "",
            _title("确认配置"),
            "",
            *backend_lines,
            "",
            f"profile         : {self.config.workload.profile}",
            f"api / stream    : {self.config.workload.api} / stream={self.config.workload.stream}",
            f"concurrency     : {self.config.workload.concurrency}",
            f"input_tokens    : {self.config.workload.input_tokens}",
            f"output_tokens   : {self.config.workload.output_tokens}",
            f"total_requests  : {self.config.workload.total_requests}",
            "prompt          : "
            + (f"jsonl={self.config.workload.prompt_jsonl}"
               if self.config.workload.prompt_jsonl
               else f"dir={self.config.workload.prompt_dir}"
                    if self.config.workload.prompt_dir
                    else "synthetic"),
            f"output_dir      : {self.config.report.output_dir}",
            "",
            *preview_block,
            "Enter 开始执行；n 仅保存配置不执行；b 返回上一步。",
        ]
        print("\n".join(lines))
        while True:
            key = _read_key()
            if key == KEY_ENTER:
                self.should_start = True
                return
            if key.lower() == "n":
                self.should_start = False
                return
            if key == KEY_BACK:
                raise _BackRequested

    def _set_nested(self, section: str, key: str, value: Any) -> None:
        self.requested.setdefault(section, {})[key] = value


# ---------------------------------------------------------------------------
# NCCL comm wizard
# ---------------------------------------------------------------------------


class _CommWizard:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args

    def run(self) -> argparse.Namespace:
        steps: list[Callable[[], None]] = [
            self._step_image,
            self._step_docker_args,
            self._step_command,
            self._step_output,
            self._step_summary,
        ]
        snapshots: list[argparse.Namespace] = []
        idx = 0
        while idx < len(steps):
            snapshots = snapshots[:idx]
            snapshots.append(copy.deepcopy(self.args))
            try:
                steps[idx]()
                idx += 1
            except _BackRequested:
                if idx == 0:
                    print("已经是第一步。")
                    continue
                self.args = snapshots[idx - 1]
                idx -= 1
        return self.args

    def _step_image(self) -> None:
        images = discover_docker_images("nccl")
        labels = [_image_label(item) for item in images]
        manual_label = "手动输入"
        if labels:
            labels.append(manual_label)
            default = labels[0] if not self.args.image else next((label for label in labels if label.startswith(self.args.image)), labels[0])
            choice = _select(_title("选择 NCCL 镜像"), labels, default)
            if choice == manual_label:
                self.args.image = _text("NCCL 镜像", self.args.image or "nccl-tests:latest")
            else:
                self.args.image = images[labels.index(choice)]["name"]
        else:
            self.args.image = _text(_title("NCCL 镜像", "(本机未识别到 nccl 镜像)"), self.args.image or "nccl-tests:latest")

    def _step_docker_args(self) -> None:
        current = " ".join(self.args.docker_arg) if self.args.docker_arg else "--gpus all --shm-size 16g --ipc=host"
        text = _editable_simple(_title("docker 参数（空格分隔）"), current)
        self.args.docker_arg = shlex.split(text) if text.strip() else []

    def _step_command(self) -> None:
        default = " ".join(self.args.passthrough) if getattr(self.args, "passthrough", None) else (
            "/opt/nccl-tests/build/all_reduce_perf -b 8 -e 1G -f 2 -g 1 -n 100 -w 20"
        )
        print(_title("编辑容器内启动命令"))
        print("常用参数: -b 起始大小 / -e 结束大小 / -f 倍数 / -g GPU 数 / -n 迭代 / -w 预热")
        text = _editable("nccl> ", default)
        try:
            argv = shlex.split(text)
        except ValueError as exc:
            print(f"命令解析失败：{exc}，回退到默认。")
            argv = shlex.split(default)
        self.args.passthrough = argv

    def _step_output(self) -> None:
        self.args.output_dir = _text(_title("输出目录"), self.args.output_dir or "benchmark_output/comm_runs")
        self.args.run_name = _text("run name（可留空）", self.args.run_name or "")

    def _step_summary(self) -> None:
        passthrough = getattr(self.args, "passthrough", []) or []
        lines = [
            "",
            _title("确认 NCCL 配置"),
            "",
            f"image       : {self.args.image}",
            f"docker_args : {' '.join(self.args.docker_arg) if self.args.docker_arg else '(none)'}",
            f"command     : {shlex.join(passthrough)}",
            f"output_dir  : {self.args.output_dir}",
            f"run_name    : {self.args.run_name or '(auto)'}",
            "",
            "Enter 开始执行；b 返回上一步。",
        ]
        print("\n".join(lines))
        while True:
            key = _read_key()
            if key == KEY_ENTER:
                return
            if key == KEY_BACK:
                raise _BackRequested


def _title(label: str, hint: str = "") -> str:
    return label if not hint else f"{label}  {hint}"


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


def _select_params(catalog: list[ParamSpec], default_selected: set[str]) -> list[ParamSpec]:
    """Group catalog by category and let the user multi-select per category.

    Default selection is empty unless the caller passes a non-empty set (used
    when re-entering this step after editing the previous selection). A user
    may select zero parameters in any category by pressing Enter without
    Space.

    Returns the chosen ParamSpec objects in catalog order.
    """
    by_cat: dict[str, list[ParamSpec]] = {}
    for spec in catalog:
        by_cat.setdefault(spec.category, []).append(spec)

    chosen_set: set[str] = set()
    for category, group in by_cat.items():
        labels = [_format_param_label(spec) for spec in group]
        defaults = [labels[i] for i, spec in enumerate(group) if spec.name in default_selected]
        chosen_labels = _multi_select(
            f"参数 · {_translate_category(category)}（{len(group)} 项）",
            labels,
            defaults,
            allow_empty=True,
        )
        for lbl in chosen_labels:
            chosen_set.add(group[labels.index(lbl)].name)

    return [spec for spec in catalog if spec.name in chosen_set]


def _ask_param_values(params: list[ParamSpec], previous: dict[str, str | None]) -> dict[str, str | None]:
    """Ask the user for each selected parameter's value.

    Returns a map flag -> value-string, or None to indicate "do not include".
    For bool flags, value-string == "FLAG" means "emit the flag with no value".
    """
    values: dict[str, str | None] = {}
    for spec in params:
        previous_value = previous.get(spec.name)
        if spec.kind == "bool":
            current = "on" if previous_value == "FLAG" else "off"
            choice = _select(
                f"{spec.name}  {spec.description}",
                ["on", "off"],
                current,
            )
            values[spec.name] = "FLAG" if choice == "on" else None
            continue
        if spec.kind == "choice":
            options = list(spec.choices)
            sentinel = "(default)"
            options.append(sentinel)
            default = previous_value if previous_value in options else (spec.default if spec.default in options else sentinel)
            chosen = _select(
                f"{spec.name}  {spec.description}",
                options,
                default,
            )
            values[spec.name] = None if chosen == sentinel else chosen
            continue
        # str / int / float
        default = previous_value if previous_value is not None else spec.default
        value = _text(f"{spec.name}  {spec.description}", default)
        values[spec.name] = value or None
    return values


def _print_param_table(params: list["ParamSpec"], values: dict[str, str | None]) -> None:
    """Show the chosen flags in a 2-column aligned table for at-a-glance review."""
    rows: list[tuple[str, str]] = []
    for spec in params:
        value = values.get(spec.name)
        if value is None:
            continue
        rows.append((spec.name, "(flag)" if value == "FLAG" else str(value)))
    if not rows:
        return
    name_width = max(len(name) for name, _ in rows)
    print()
    print("已设置参数:")
    for name, val in rows:
        print(f"  {name.ljust(name_width)}  {val}")


def _format_profile_label(name: str) -> str:
    """Render a profile choice with its actual concurrency / token / total values."""
    cfg = PROFILES.get(name) or {}
    parts = [name]
    if "concurrency" in cfg:
        parts.append(f"concurrency={cfg['concurrency']}")
    if "input_tokens" in cfg:
        parts.append(f"i={cfg['input_tokens']}")
    if "output_tokens" in cfg:
        parts.append(f"o={cfg['output_tokens']}")
    if "total_requests" in cfg:
        parts.append(f"total={cfg['total_requests']}")
    return "  ".join(parts)


_BACKEND_IMAGE_KEYWORDS = {
    "vllm": ("vllm", "vllm-openai"),
    "sglang": ("sglang", "sgl-"),
}


def _filter_images_for_backend(images: list[dict[str, str]], backend: str) -> list[dict[str, str]]:
    """Keep only images whose name contains a backend-specific keyword."""
    keywords = _BACKEND_IMAGE_KEYWORDS.get(backend.lower(), ())
    if not keywords:
        return images
    return [item for item in images if any(kw in item["name"].lower() for kw in keywords)]


def _split_container_dir(mount_entry: str, fallback: str) -> str:
    """Pull `container` out of `host:container[:ro]`. Fallback if malformed."""
    parts = mount_entry.split(":")
    if len(parts) < 2 or not parts[1].strip():
        return fallback
    return parts[1].strip()


def _rebase_path(host_path: str, host_root: str, container_root: str) -> str:
    """Translate a host path under `host_root` to its equivalent under `container_root`."""
    from pathlib import Path as _Path
    try:
        rel = _Path(host_path).relative_to(host_root)
    except ValueError:
        return ""
    rel_str = str(rel)
    if rel_str == ".":
        return container_root
    return str(_Path(container_root) / rel_str)


def _container_mount_for(item: dict[str, str]) -> tuple[str, str]:
    """Choose what to bind-mount for a non-HF model and how the path looks inside.

    Returns (host_dir_to_mount, container_path_to_the_model). The container
    path equals the host path (we mount at the same location), so the user's
    command line stays exactly what they would write outside a container.

    Strategy:
    - For paths under `~/.cache/modelscope` mount the ModelScope cache root so
      vllm/sglang can resolve relative HF-style ids too.
    - For arbitrary local paths mount the model's parent dir (lets one mount
      cover sibling files like tokenizer / configs that live next to the
      weights).
    """
    from pathlib import Path as _Path
    host_path = _Path(item["path"])
    # Prefer mounting the ModelScope cache root so multiple models share one
    # bind mount. Detect it by path text, not Path.home(), so paths belonging
    # to other users (e.g. /home/u/.cache/modelscope/...) work too.
    parts = host_path.parts
    if ".cache" in parts and "modelscope" in parts:
        idx = parts.index(".cache")
        if idx + 1 < len(parts) and parts[idx + 1] == "modelscope":
            ms_root = _Path(*parts[: idx + 2])
            return str(ms_root), str(host_path)
    parent = host_path.parent
    if str(parent) in {"", "/"}:
        return str(host_path), str(host_path)
    return str(parent), str(host_path)


# vLLM: the official vllm/vllm-openai image sets ENTRYPOINT to
# ["vllm", "serve"], so no launcher prefix is needed. But third-party or
# custom images may have no ENTRYPOINT, requiring `vllm serve` explicitly.
# Alternatively, legacy scripts may use `python -m vllm.entrypoints.openai.api_server`.
VLLM_LAUNCHERS: dict[str, list[str]] = {
    "(无前缀，镜像 ENTRYPOINT 已是 vllm serve)  (推荐，官方 vllm/vllm-openai 镜像)": [],
    "vllm serve  (镜像无 ENTRYPOINT 时需要)": ["vllm", "serve"],
}
VLLM_DEFAULT_LAUNCHER = "(无前缀，镜像 ENTRYPOINT 已是 vllm serve)  (推荐，官方 vllm/vllm-openai 镜像)"
_VLLM_LAUNCHER_TOKENS = {tok for argv in VLLM_LAUNCHERS.values() for tok in argv if tok} | {"python3"}

# SGLang ships two equivalent launchers. The CLI `sglang serve` is the
# recommended entrypoint in 0.5+, the older `python -m sglang.launch_server`
# still works (and emits a deprecation warning).
SGLANG_LAUNCHERS: dict[str, list[str]] = {
    "sglang serve  (推荐，sglang 0.5+)": ["sglang", "serve"],
    "python -m sglang.launch_server  (老用法，兼容旧版本)": ["python", "-m", "sglang.launch_server"],
}
SGLANG_DEFAULT_LAUNCHER = "sglang serve  (推荐，sglang 0.5+)"
# Any token that can show up at the start of an sglang container command but
# is the launcher itself, not the model path. Used to skip past them when
# parsing the command back out (e.g. _uses_local_model).
_SGLANG_LAUNCHER_TOKENS = {tok for argv in SGLANG_LAUNCHERS.values() for tok in argv} | {"python3"}


def _build_command(
    backend: str,
    model_path: str,
    params: list[ParamSpec],
    values: dict[str, str | None],
    vllm_launcher: list[str] | None = None,
    sglang_launcher: list[str] | None = None,
) -> list[str]:
    if backend == "vllm":
        # 用户选择的启动器前缀。空列表表示镜像 ENTRYPOINT 已是 vllm serve，
        # 非空则显式写 vllm serve 或 python -m vllm.entrypoints.openai.api_server。
        launcher = vllm_launcher if vllm_launcher is not None else VLLM_LAUNCHERS[VLLM_DEFAULT_LAUNCHER]
        cmd: list[str] = [*launcher, model_path]
    else:
        # lmsysorg/sglang 镜像没有把 server 设为 entrypoint，必须自己带上启动器。
        launcher = sglang_launcher or SGLANG_LAUNCHERS[SGLANG_DEFAULT_LAUNCHER]
        cmd = [*launcher, "--model-path", model_path]
    for spec in params:
        value = values.get(spec.name)
        if value is None:
            continue
        if value == "FLAG":
            cmd.append(spec.name)
        else:
            cmd.extend([spec.name, str(value)])
    return cmd


def _format_param_label(spec: ParamSpec) -> str:
    suffix = ""
    if spec.kind == "bool":
        suffix = "  [flag]"
    elif spec.kind == "choice":
        suffix = f"  [{'|'.join(spec.choices)}]"
    default = f"  (默认: {spec.default})" if spec.default else ""
    return f"{spec.name}  {spec.description}{default}{suffix}"


def _translate_category(category: str) -> str:
    return {
        "core": "核心",
        "parallel": "并行",
        "memory": "显存与上下文",
        "performance": "性能与调度",
        "model": "模型与精度",
        "logging": "日志",
        "advanced": "高级",
    }.get(category, category)


def _default_comm_args() -> argparse.Namespace:
    return argparse.Namespace(
        image="",
        output_dir="benchmark_output/comm_runs",
        run_name="",
        gpus="all",
        timeout=1800,
        docker_arg=[],
        env=[],
        dry_run=False,
        interactive=False,
        passthrough=[],
    )


# ---------------------------------------------------------------------------
# Primitive UI helpers
# ---------------------------------------------------------------------------


class _BackRequested(Exception):
    pass


def _require_tty() -> None:
    if not sys.stdin.isatty() or not sys.stdout.isatty():
        raise RuntimeError(
            "当前环境不支持交互式输入。请使用 --config 或普通 CLI 参数。"
        )


def _select(title: str, options: list[str], default: str) -> str:
    selected = options.index(default) if default in options else 0
    renderer = _InlineRenderer()
    try:
        while True:
            renderer.draw(_option_lines(title, options, selected, set(), multi=False))
            key = _read_key()
            if key == KEY_UP:
                selected = (selected - 1) % len(options)
            elif key == KEY_DOWN:
                selected = (selected + 1) % len(options)
            elif key == KEY_BACK:
                raise _BackRequested
            elif key == KEY_ENTER:
                renderer.finish()
                value = options[selected]
                print(f"{title.splitlines()[0]}: {value}")
                return value
    finally:
        renderer.finish()


def _multi_select(title: str, options: list[str], defaults: list[str], allow_empty: bool = False) -> list[str]:
    selected = 0
    chosen = {idx for idx, value in enumerate(options) if value in defaults}
    renderer = _InlineRenderer()
    try:
        while True:
            renderer.draw(_option_lines(title, options, selected, chosen, multi=True))
            key = _read_key()
            if key == KEY_UP:
                selected = (selected - 1) % len(options)
            elif key == KEY_DOWN:
                selected = (selected + 1) % len(options)
            elif key == KEY_SPACE:
                chosen.symmetric_difference_update({selected})
            elif key == KEY_BACK:
                raise _BackRequested
            elif key == KEY_ENTER:
                renderer.finish()
                if chosen:
                    values = [options[idx] for idx in sorted(chosen)]
                elif allow_empty:
                    values = []
                else:
                    values = [options[selected]]
                head = title.splitlines()[0]
                if not values:
                    print(f"{head}: (无)")
                else:
                    print(f"{head}:")
                    for value in values:
                        print(f"  - {value}")
                return values
    finally:
        renderer.finish()


def _multi_int(title: str, options: list[int], defaults: list[int]) -> list[int]:
    values = _multi_select(title, [str(value) for value in options], [str(value) for value in defaults])
    return [int(value) for value in values]


def _text(title: str, default: str) -> str:
    if default:
        prompt = f"{title}（默认: {default}，直接 Enter 采用）: "
    else:
        prompt = f"* {title}: "
    value = _clean_text(input(prompt))
    if _is_back(value):
        raise _BackRequested
    return value or default


def _editable_simple(title: str, default: str) -> str:
    print(title)
    value = _editable("> ", default)
    if _is_back(value):
        raise _BackRequested
    return value or default


def _editable(prompt: str, default: str) -> str:
    if not default or not sys.stdin.isatty() or not sys.stdout.isatty():
        if default:
            print(default)
        return input(prompt)
    return _raw_editable(prompt, default)


def _raw_editable(prompt: str, default: str) -> str:
    buffer = list(default)
    cursor = len(buffer)
    scroll = 0

    def render() -> None:
        nonlocal scroll
        columns = shutil.get_terminal_size((120, 20)).columns
        view_width = max(10, columns - len(prompt) - 2)
        if cursor < scroll:
            scroll = cursor
        elif cursor > scroll + view_width:
            scroll = cursor - view_width
        visible = "".join(buffer[scroll:scroll + view_width])
        cursor_col = len(prompt) + (cursor - scroll)
        sys.stdout.write("\r\033[2K")
        sys.stdout.write(prompt + visible)
        right = len(prompt) + len(visible) - cursor_col
        if right > 0:
            sys.stdout.write(f"\033[{right}D")
        sys.stdout.flush()

    with _raw_terminal():
        render()
        while True:
            ch = sys.stdin.read(1)
            if ch in ("\x03", "\x1a"):
                raise KeyboardInterrupt
            if ch in ("\r", "\n"):
                # raw terminal disables ONLCR, so we must emit CR ourselves.
                sys.stdout.write("\r\n")
                sys.stdout.flush()
                return "".join(buffer)
            if ch in ("\x7f", "\b"):
                if cursor > 0:
                    del buffer[cursor - 1]
                    cursor -= 1
                render()
                continue
            if ch == "\x1b":
                seq = sys.stdin.read(2)
                if seq == "[C" and cursor < len(buffer):
                    cursor += 1
                elif seq == "[D" and cursor > 0:
                    cursor -= 1
                elif seq in ("[H", "OH"):
                    cursor = 0
                elif seq in ("[F", "OF"):
                    cursor = len(buffer)
                render()
                continue
            if ch == "\x01":
                cursor = 0
                render()
                continue
            if ch == "\x05":
                cursor = len(buffer)
                render()
                continue
            if ch.isprintable():
                buffer.insert(cursor, ch)
                cursor += 1
                render()


class _InlineRenderer:
    def __init__(self) -> None:
        self.line_count = 0

    def draw(self, lines: list[str]) -> None:
        self._clear_previous()
        sys.stdout.write("\n".join(lines) + "\n")
        sys.stdout.flush()
        self.line_count = len(lines)

    def finish(self) -> None:
        self._clear_previous()
        self.line_count = 0

    def _clear_previous(self) -> None:
        if self.line_count == 0:
            return
        sys.stdout.write(f"\033[{self.line_count}F")
        for _ in range(self.line_count):
            sys.stdout.write("\033[2K\033[1E")
        sys.stdout.write(f"\033[{self.line_count}F")
        sys.stdout.flush()


def _option_lines(title: str, options: list[str], selected: int, chosen: set[int], multi: bool) -> list[str]:
    head = title.splitlines()
    lines = [*head]
    for idx, option in enumerate(options):
        cursor = ">" if idx == selected else " "
        if multi:
            marker = "[✓]" if idx in chosen else "[ ]"
        else:
            marker = "[✓]" if idx == selected else "[ ]"
        lines.append(f"{cursor} {marker} {option}")
    return lines


def _image_label(image: dict[str, str]) -> str:
    suffix = f" ({image['size']})" if image.get("size") else ""
    return f"{image['name']}{suffix}"


def _read_key() -> str:
    with _raw_terminal():
        ch = sys.stdin.read(1)
        if ch == "\x1b":
            seq = sys.stdin.read(2)
            return _parse_key(ch, seq)
        return _parse_key(ch)


def _parse_key(ch: str, seq: str = "") -> str:
    if ch in ("\x03", "\x1a"):
        raise KeyboardInterrupt
    if ch == "\x1b":
        if seq == "[A":
            return KEY_UP
        if seq == "[B":
            return KEY_DOWN
        if seq == "[D":
            return KEY_BACK
        return ch + seq
    if ch in ("\r", "\n"):
        return KEY_ENTER
    if ch == " ":
        return KEY_SPACE
    if ch in ("\x7f", "\b"):
        return KEY_BACK
    if ch in ("b", "B"):
        return KEY_BACK
    if ch in ("k", "K"):
        return KEY_UP
    if ch in ("j", "J"):
        return KEY_DOWN
    return ch


@contextmanager
def _raw_terminal():
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _is_back(value: str) -> bool:
    normalized = value.replace("\ufffd", "").strip().lower()
    return normalized in {":back", "back"}


def _clean_text(value: str) -> str:
    cleaned = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", value)
    for literal in ("^[[A", "^[[B", "^[[C", "^[[D"):
        cleaned = cleaned.replace(literal, "")
    return cleaned.strip()
