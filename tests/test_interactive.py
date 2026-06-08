from contextlib import contextmanager
import os

import pytest

from llm_bench.config import BenchConfig
from llm_bench.interactive import (
    SGLANG_PARAMS,
    VLLM_PARAMS,
    _BackRequested,
    _build_command,
    VLLM_LAUNCHERS,
    _container_mount_for,
    _filter_images_for_backend,
    _InferWizard,
    _clean_text,
    _is_back,
    _option_lines,
    _parse_key,
    _raw_editable,
    _rebase_path,
    _split_container_dir,
)


def test_option_lines_renders_title_and_options():
    lines = _option_lines("选择", ["a"], 0, set(), multi=False)
    assert lines[0] == "选择"
    assert lines[-1] == "> [✓] a"


def test_replacement_back_text_requests_back():
    assert _is_back(":back") is True
    assert _is_back("\ufffdback") is True
    assert _is_back("anything") is False


def test_clean_text_strips_arrow_escape_sequences():
    assert _clean_text("\x1b[C") == ""
    assert _clean_text("^[[C") == ""
    assert _clean_text("serve --model X^[[D") == "serve --model X"


def _spec(catalog, name):
    return next(s for s in catalog if s.name == name)


def test_build_command_vllm_positional_model_path():
    # vllm/vllm-openai 镜像 entrypoint 已是 `vllm serve`，选择默认启动器（无前缀）时命令不能再带这个头。
    params = [_spec(VLLM_PARAMS, "--tensor-parallel-size"), _spec(VLLM_PARAMS, "--max-model-len")]
    values = {"--tensor-parallel-size": "2", "--max-model-len": "4096"}
    cmd = _build_command("vllm", "/models/qwen", params, values, vllm_launcher=[])
    assert cmd == [
        "/models/qwen",
        "--tensor-parallel-size", "2",
        "--max-model-len", "4096",
    ]
    assert cmd[:2] != ["vllm", "serve"]


def test_build_command_vllm_with_vllm_serve_launcher():
    # 非官方镜像无 ENTRYPOINT 时，需要显式加 `vllm serve` 前缀。
    params = [_spec(VLLM_PARAMS, "--tensor-parallel-size")]
    values = {"--tensor-parallel-size": "2"}
    cmd = _build_command("vllm", "/models/qwen", params, values, vllm_launcher=["vllm", "serve"])
    assert cmd == ["vllm", "serve", "/models/qwen", "--tensor-parallel-size", "2"]


def test_build_command_vllm_with_legacy_launcher():
    params = [_spec(VLLM_PARAMS, "--port")]
    values = {"--port": "8000"}
    cmd = _build_command(
        "vllm", "/models/qwen", params, values,
        vllm_launcher=["python", "-m", "vllm.entrypoints.openai.api_server"],
    )
    assert cmd == ["python", "-m", "vllm.entrypoints.openai.api_server", "/models/qwen", "--port", "8000"]


def test_build_command_sglang_uses_model_path_flag():
    # Default launcher in sglang 0.5+ is `sglang serve`; the legacy
    # `python -m sglang.launch_server` is still selectable via wizard.
    params = [_spec(SGLANG_PARAMS, "--tp")]
    cmd = _build_command("sglang", "/models/qwen", params, {"--tp": "4"})
    assert cmd == ["sglang", "serve", "--model-path", "/models/qwen", "--tp", "4"]


def test_build_command_sglang_accepts_legacy_launcher_override():
    params = [_spec(SGLANG_PARAMS, "--tp")]
    cmd = _build_command(
        "sglang",
        "/models/qwen",
        params,
        {"--tp": "4"},
        sglang_launcher=["python", "-m", "sglang.launch_server"],
    )
    assert cmd == ["python", "-m", "sglang.launch_server", "--model-path", "/models/qwen", "--tp", "4"]


def test_build_command_bool_flag_emits_no_value():
    params = [_spec(VLLM_PARAMS, "--trust-remote-code"), _spec(VLLM_PARAMS, "--enforce-eager")]
    values = {"--trust-remote-code": "FLAG", "--enforce-eager": None}
    cmd = _build_command("vllm", "/m", params, values)
    # FLAG -> emitted as a bare flag with no value; None -> omitted entirely.
    assert cmd == ["/m", "--trust-remote-code"]
    assert "FLAG" not in cmd
    assert "--enforce-eager" not in cmd


def test_build_command_skips_none_values():
    params = [_spec(VLLM_PARAMS, "--dtype"), _spec(VLLM_PARAMS, "--quantization")]
    values = {"--dtype": "bfloat16", "--quantization": None}
    cmd = _build_command("vllm", "/m", params, values)
    assert "--dtype" in cmd and "bfloat16" in cmd
    assert "--quantization" not in cmd


def test_param_names_are_upstream_exact():
    # No param name should have been accidentally renamed/translated.
    for spec in VLLM_PARAMS + SGLANG_PARAMS:
        assert spec.name.startswith("--")
        assert " " not in spec.name


@pytest.mark.parametrize("key", ["\x03", "\x1a"])
def test_ctrl_c_and_ctrl_z_interrupt(key):
    with pytest.raises(KeyboardInterrupt):
        _parse_key(key)


def test_back_text_input_raises_backrequested(monkeypatch):
    from llm_bench.interactive import _text

    monkeypatch.setattr("builtins.input", lambda prompt: ":back")
    with pytest.raises(_BackRequested):
        _text("test", "default")


def test_back_in_first_step_keeps_wizard_running(monkeypatch):
    # Verify _InferWizard handles back-at-first-step gracefully.
    wizard = _InferWizard(BenchConfig())

    calls = {"backend": 0, "image": 0, "command": 0, "model_port": 0, "hf": 0, "profile": 0, "prompt": 0, "summary": 0}

    def make_step(name, behaviour):
        def step(self):
            calls[name] += 1
            behaviour(self, calls[name])
        return step

    # Step sequence: ask back on step 2 once, then no-op everywhere; summary stops with n.
    monkeypatch.setattr(_InferWizard, "_step_backend", make_step("backend", lambda s, n: None))
    monkeypatch.setattr(_InferWizard, "_step_image", make_step("image", lambda s, n: (_ for _ in ()).throw(_BackRequested) if n == 1 else None))
    # _step_command was split into _step_pick_model + _step_pick_params +
    # _step_finalize_command so back-button granularity is sane; patch them all.
    monkeypatch.setattr(_InferWizard, "_step_command", make_step("command", lambda s, n: None))
    calls["pick_model"] = 0
    calls["pick_params"] = 0
    calls["finalize_command"] = 0
    monkeypatch.setattr(_InferWizard, "_step_pick_model", make_step("pick_model", lambda s, n: None))
    monkeypatch.setattr(_InferWizard, "_step_pick_params", make_step("pick_params", lambda s, n: None))
    monkeypatch.setattr(_InferWizard, "_step_finalize_command", make_step("finalize_command", lambda s, n: None))
    calls["docker_args"] = 0
    monkeypatch.setattr(_InferWizard, "_step_docker_args", make_step("docker_args", lambda s, n: None))
    monkeypatch.setattr(_InferWizard, "_step_model_and_port", make_step("model_port", lambda s, n: None))
    monkeypatch.setattr(_InferWizard, "_step_hf", make_step("hf", lambda s, n: None))
    monkeypatch.setattr(_InferWizard, "_step_profile_and_workload", make_step("profile", lambda s, n: None))
    monkeypatch.setattr(_InferWizard, "_step_prompt_and_api", make_step("prompt", lambda s, n: None))
    monkeypatch.setattr(_InferWizard, "_step_summary", make_step("summary", lambda s, n: setattr(s, "should_start", False)))

    _config, _requested, should_start = wizard.run()
    assert should_start is False
    assert calls["backend"] == 2  # re-ran after back
    assert calls["image"] == 2
    assert calls["summary"] == 1


def test_raw_editable_basic_cursor_editing(monkeypatch):
    class FakeInput:
        def __init__(self, data: str) -> None:
            self.data = data
            self.idx = 0

        def read(self, count: int) -> str:
            chunk = self.data[self.idx:self.idx + count]
            self.idx += count
            return chunk

        def isatty(self) -> bool:
            return True

        def fileno(self) -> int:
            return 0

    class FakeOutput:
        def __init__(self) -> None:
            self.parts: list[str] = []

        def write(self, text: str) -> None:
            self.parts.append(text)

        def flush(self) -> None:
            pass

        def isatty(self) -> bool:
            return True

    @contextmanager
    def fake_raw_terminal():
        yield

    monkeypatch.setattr("llm_bench.interactive._raw_terminal", fake_raw_terminal)
    monkeypatch.setattr("sys.stdin", FakeInput("\x1b[D\x7fZ\n"))
    monkeypatch.setattr("sys.stdout", FakeOutput())
    assert _raw_editable("edit> ", "abc") == "aZc"


def _wizard_with(command, backend="vllm"):
    cfg = BenchConfig()
    cfg.backend.name = backend
    cfg.backend.command = list(command)
    return _InferWizard(cfg)


# -----------------------------------------------------------------------------
# _steps_for: _step_hf 只在用 HF id 时出现（防回退死循环）
# -----------------------------------------------------------------------------

def test_steps_for_splits_command_into_three_substeps():
    # Regression for "press b rewinds the entire model selection" complaint.
    # The old monolithic _step_command did model + mount + params + edit in
    # one step, so backing out of _step_docker_args dumped you all the way
    # back to model selection. Now those three concerns are independent steps.
    w = _wizard_with(["/m"], backend="vllm")
    names = [s.__name__ for s in w._steps_for("vllm")]
    assert "_step_pick_model" in names
    assert "_step_pick_params" in names
    assert "_step_finalize_command" in names
    assert "_step_command" not in names
    # And the order: pick_model -> pick_params -> finalize_command -> docker_args
    pick_model_idx = names.index("_step_pick_model")
    pick_params_idx = names.index("_step_pick_params")
    finalize_idx = names.index("_step_finalize_command")
    docker_args_idx = names.index("_step_docker_args")
    assert pick_model_idx < pick_params_idx < finalize_idx < docker_args_idx


def test_steps_for_includes_hf_step_when_model_is_hf_id():
    w = _wizard_with(["Qwen/Qwen2.5-7B-Instruct", "--host", "0.0.0.0"])
    names = [s.__name__ for s in w._steps_for("vllm")]
    assert "_step_hf" in names


def test_steps_for_excludes_hf_step_when_model_is_local_path():
    w = _wizard_with(["/root/.cache/modelscope/m", "--host", "0.0.0.0"])
    names = [s.__name__ for s in w._steps_for("vllm")]
    assert "_step_hf" not in names


def test_steps_for_transformers_has_no_docker_steps():
    cfg = BenchConfig()
    cfg.backend.name = "transformers"
    w = _InferWizard(cfg)
    names = [s.__name__ for s in w._steps_for("transformers")]
    assert "_step_image" not in names
    assert "_step_docker_args" not in names
    assert "_step_command" not in names
    assert "_step_transformers" in names


# -----------------------------------------------------------------------------
# _uses_local_model：判断容器内模型是绝对路径还是 HF id
# -----------------------------------------------------------------------------

def test_uses_local_model_true_for_absolute_path():
    assert _wizard_with(["/root/.cache/modelscope/m", "--host", "0.0.0.0"])._uses_local_model() is True


def test_uses_local_model_false_for_hf_id():
    assert _wizard_with(["Qwen/Qwen2.5-7B-Instruct", "--host", "0.0.0.0"])._uses_local_model() is False


def test_uses_local_model_false_for_empty_command():
    assert _wizard_with([])._uses_local_model() is False


def test_uses_local_model_for_sglang_reads_model_path_value():
    # Regression: sglang's command starts with `python -m sglang.launch_server`
    # so the first non-flag token is `python`, not the model. Previously this
    # caused _uses_local_model() to wrongly return False (treating the model
    # as an HF id) and the HF cache/token wizard step would reappear even
    # though the user picked a local ModelScope path.
    sg_local = [
        "python", "-m", "sglang.launch_server",
        "--model-path", "/root/.cache/modelscope/hub/models/Qwen/Q",
        "--host", "0.0.0.0", "--port", "30000",
    ]
    w = _wizard_with(sg_local, backend="sglang")
    assert w._uses_local_model() is True
    assert "_step_hf" not in [s.__name__ for s in w._steps_for("sglang")]


def test_container_listen_port_reads_value_after_flag():
    w = _wizard_with(["/m", "--host", "0.0.0.0", "--port", "30000"])
    assert w._container_listen_port() == 30000


def test_container_listen_port_returns_none_when_flag_missing():
    w = _wizard_with(["/m", "--host", "0.0.0.0"])
    assert w._container_listen_port() is None


def test_sync_container_port_updates_existing_flag():
    # Regression: previously the wizard let docker -p X:X and the framework's
    # --port drift apart (e.g. -p 8000:8000 while sglang listened on 30000 by
    # default), causing every benchmark request to hit "nothing listening".
    w = _wizard_with([
        "python", "-m", "sglang.launch_server",
        "--model-path", "/m", "--host", "0.0.0.0", "--port", "30000",
    ], backend="sglang")
    w._sync_container_port(8001)
    cmd = w.config.backend.command
    idx = cmd.index("--port")
    assert cmd[idx + 1] == "8001"
    assert cmd.count("8001") == 1
    assert "30000" not in cmd


def test_sync_container_port_silent_when_flag_absent():
    w = _wizard_with(["/m"], backend="vllm")
    w._sync_container_port(8000)  # must not raise
    assert "--port" not in w.config.backend.command


def test_sglang_launcher_from_command_detects_each_variant():
    serve = _wizard_with(["sglang", "serve", "--model-path", "/m"], backend="sglang")
    assert serve._sglang_launcher_from_command() == ["sglang", "serve"]

    legacy = _wizard_with(
        ["python", "-m", "sglang.launch_server", "--model-path", "/m"], backend="sglang",
    )
    assert legacy._sglang_launcher_from_command() == ["python", "-m", "sglang.launch_server"]

    other = _wizard_with(["something", "else"], backend="sglang")
    assert other._sglang_launcher_from_command() is None


def test_vllm_launcher_from_command_detects_each_variant():
    serve = _wizard_with(["vllm", "serve", "/m", "--host", "0.0.0.0"], backend="vllm")
    assert serve._vllm_launcher_from_command() == ["vllm", "serve"]

    legacy = _wizard_with(
        ["python", "-m", "vllm.entrypoints.openai.api_server", "/m"], backend="vllm",
    )
    assert legacy._vllm_launcher_from_command() == ["python", "-m", "vllm.entrypoints.openai.api_server"]

    no_prefix = _wizard_with(["/m", "--host", "0.0.0.0"], backend="vllm")
    assert no_prefix._vllm_launcher_from_command() is None


def test_step_transformers_loops_on_bad_batch_size(monkeypatch):
    # Regression: previously raise _BackRequested kicked the user back to
    # _step_backend after a single typo, discarding all earlier choices.
    cfg = BenchConfig()
    cfg.backend.name = "transformers"
    w = _InferWizard(cfg)
    # Walk through every _text call the step makes; bad batch_size inputs
    # must trigger a re-prompt loop instead of raising _BackRequested.
    inputs = iter([
        "/m",                     # model_path (manual fallback after empty discover)
        "SKIP",                   # tokenizer_path -> empty (we map SKIP -> '')
        "auto",                   # device_map
        "abc",                    # batch_size: bad
        "xyz",                    # batch_size: bad again
        "8",                      # batch_size: ok
    ])

    def fake_text(label, default=""):
        try:
            value = next(inputs)
        except StopIteration:
            return default
        return "" if value == "SKIP" else value

    monkeypatch.setattr("llm_bench.interactive._text", fake_text)
    monkeypatch.setattr("llm_bench.interactive._select", lambda *a, **kw: a[2])
    monkeypatch.setattr("llm_bench.interactive.discover_model_paths", lambda limit=30: [])

    w._step_transformers()
    assert w.config.transformers.batch_size == 8
    assert w.config.transformers.model_path == "/m"


def test_step_transformers_uses_model_scanner(monkeypatch):
    cfg = BenchConfig()
    cfg.backend.name = "transformers"
    w = _InferWizard(cfg)
    discovered = [
        {"source": "Hugging Face", "name": "Qwen/Qwen2.5-7B", "path": "/root/.cache/hf"},
        {"source": "ModelScope", "name": "Qwen/Qwen3.5-9B", "path": "/root/.cache/modelscope/.../Qwen3.5-9B"},
    ]
    select_calls: list[tuple[str, list[str]]] = []

    def fake_select(title, options, default):
        select_calls.append((title, list(options)))
        if "选择模型" in title or "Select model" in title:
            return options[1]   # pick the ModelScope row
        return default

    monkeypatch.setattr("llm_bench.interactive.discover_model_paths", lambda limit=30: discovered)
    monkeypatch.setattr("llm_bench.interactive._select", fake_select)
    monkeypatch.setattr("llm_bench.interactive._text", lambda label, default="": default or "8")

    w._step_transformers()
    # Selecting the ModelScope entry should populate model_path with the host
    # absolute path (transformers runs in-process - no bind mount needed).
    assert w.config.transformers.model_path == "/root/.cache/modelscope/.../Qwen3.5-9B"


def test_uses_local_model_for_sglang_serve_launcher():
    # Regression: switching the default sglang launcher to `sglang serve`
    # must not regress _uses_local_model (it skips both launcher prefixes).
    w = _wizard_with(
        ["sglang", "serve", "--model-path", "/root/.cache/modelscope/m", "--host", "0.0.0.0"],
        backend="sglang",
    )
    assert w._uses_local_model() is True
    assert "_step_hf" not in [s.__name__ for s in w._steps_for("sglang")]


def test_uses_local_model_for_vllm_serve_launcher():
    # vllm serve launcher with local path should still detect local model.
    w = _wizard_with(
        ["vllm", "serve", "/root/.cache/modelscope/m", "--host", "0.0.0.0"],
        backend="vllm",
    )
    assert w._uses_local_model() is True
    assert "_step_hf" not in [s.__name__ for s in w._steps_for("vllm")]


def test_uses_local_model_for_sglang_hf_id_keeps_hf_step():
    sg_hf = [
        "python", "-m", "sglang.launch_server",
        "--model-path", "Qwen/Qwen2.5-7B-Instruct",
        "--host", "0.0.0.0",
    ]
    w = _wizard_with(sg_hf, backend="sglang")
    assert w._uses_local_model() is False
    assert "_step_hf" in [s.__name__ for s in w._steps_for("sglang")]


# -----------------------------------------------------------------------------
# _auto_model_name: 从容器命令推导 OpenAI client 用的 model 名
# -----------------------------------------------------------------------------

def test_auto_model_name_uses_last_segment_for_absolute_path():
    w = _wizard_with(["/root/.cache/modelscope/hub/models/Qwen/Qwen3.5-9B", "--host", "0.0.0.0"])
    assert w._auto_model_name() == "Qwen3.5-9B"


def test_auto_model_name_keeps_hf_id_intact():
    w = _wizard_with(["Qwen/Qwen2.5-7B-Instruct", "--host", "0.0.0.0"])
    assert w._auto_model_name() == "Qwen/Qwen2.5-7B-Instruct"


def test_auto_model_name_for_sglang_reads_model_path_flag():
    w = _wizard_with(
        ["python", "-m", "sglang.launch_server", "--model-path", "/x/foo", "--host", "0.0.0.0"],
        backend="sglang",
    )
    assert w._auto_model_name() == "foo"


def test_auto_model_name_falls_back_to_existing_model_name():
    cfg = BenchConfig()
    cfg.backend.name = "vllm"
    cfg.backend.model_name = "preset"
    cfg.backend.command = []
    assert _InferWizard(cfg)._auto_model_name() == "preset"


# -----------------------------------------------------------------------------
# _served_model_name_from_command + _ensure_served_model_name：双向同步
# -----------------------------------------------------------------------------

def test_served_model_name_from_command_returns_existing_value():
    w = _wizard_with(["/m", "--served-model-name", "custom"])
    assert w._served_model_name_from_command() == "custom"


def test_served_model_name_from_command_returns_empty_when_absent():
    w = _wizard_with(["/m", "--host", "0.0.0.0"])
    assert w._served_model_name_from_command() == ""


def test_served_model_name_from_command_handles_orphan_flag():
    w = _wizard_with(["/m", "--served-model-name"])  # value missing
    assert w._served_model_name_from_command() == ""


def test_ensure_served_model_name_appends_when_absent():
    w = _wizard_with(["/root/.cache/modelscope/m", "--host", "0.0.0.0"])
    w._ensure_served_model_name("alias")
    assert w.config.backend.command[-2:] == ["--served-model-name", "alias"]


def test_ensure_served_model_name_noop_when_already_matching_path():
    w = _wizard_with(["Qwen/Qwen2.5-7B-Instruct", "--host", "0.0.0.0"])
    w._ensure_served_model_name("Qwen/Qwen2.5-7B-Instruct")
    assert "--served-model-name" not in w.config.backend.command


def test_ensure_served_model_name_replaces_existing_alias():
    w = _wizard_with(["/m", "--served-model-name", "old"])
    w._ensure_served_model_name("new")
    idx = w.config.backend.command.index("--served-model-name")
    assert w.config.backend.command[idx + 1] == "new"
    assert w.config.backend.command.count("--served-model-name") == 1


def test_ensure_served_model_name_works_for_sglang():
    w = _wizard_with(
        ["python", "-m", "sglang.launch_server", "--model-path", "/x", "--host", "0.0.0.0"],
        backend="sglang",
    )
    w._ensure_served_model_name("alias")
    assert w.config.backend.command[-2:] == ["--served-model-name", "alias"]


# -----------------------------------------------------------------------------
# _filter_images_for_backend：按 backend 关键词过滤本地镜像
# -----------------------------------------------------------------------------

def test_filter_images_keeps_only_vllm_images():
    images = [
        {"name": "vllm-openai:latest"},
        {"name": "registry.example/inference/vllm-openai:v1"},
        {"name": "lmsysorg/sglang:latest"},
        {"name": "eipwork/kuboard:v3"},
        {"name": "ghcr.io/nccl-tests:cu12"},
    ]
    out = _filter_images_for_backend(images, "vllm")
    assert [i["name"] for i in out] == ["vllm-openai:latest", "registry.example/inference/vllm-openai:v1"]


def test_filter_images_keeps_only_sglang_images():
    images = [{"name": "vllm-openai:latest"}, {"name": "lmsysorg/sglang:latest"}, {"name": "sgl-runtime:v1"}]
    out = _filter_images_for_backend(images, "sglang")
    assert {i["name"] for i in out} == {"lmsysorg/sglang:latest", "sgl-runtime:v1"}


def test_filter_images_unknown_backend_returns_all():
    images = [{"name": "x"}, {"name": "y"}]
    assert _filter_images_for_backend(images, "transformers") == images


# -----------------------------------------------------------------------------
# 挂载路径推导：_split_container_dir / _rebase_path / _container_mount_for
# -----------------------------------------------------------------------------

def test_split_container_dir_basic():
    assert _split_container_dir("/host/.cache/ms:/models", "/fallback") == "/models"


def test_split_container_dir_with_ro_flag():
    assert _split_container_dir("/h:/c:ro", "/fallback") == "/c"


def test_split_container_dir_falls_back_when_malformed():
    assert _split_container_dir("only_one", "/fallback") == "/fallback"
    assert _split_container_dir("/h:", "/fallback") == "/fallback"


def test_rebase_path_under_root():
    assert _rebase_path("/a/b/c/d", "/a/b", "/x") == "/x/c/d"


def test_rebase_path_equal_to_root():
    assert _rebase_path("/a/b", "/a/b", "/x") == "/x"


def test_rebase_path_returns_empty_when_unrelated():
    assert _rebase_path("/somewhere/else", "/a/b", "/x") == ""


def test_container_mount_for_modelscope_uses_cache_root():
    item = {"source": "ModelScope", "name": "Qwen/Q@hash", "path": "/home/u/.cache/modelscope/hub/models/Qwen/Qwen3"}
    host_dir, cpath = _container_mount_for(item)
    assert host_dir == "/home/u/.cache/modelscope"
    assert cpath == "/home/u/.cache/modelscope/hub/models/Qwen/Qwen3"


def test_container_mount_for_modelscope_works_for_root_user():
    item = {"source": "ModelScope", "name": "q", "path": "/root/.cache/modelscope/hub/models/q"}
    host_dir, _cpath = _container_mount_for(item)
    assert host_dir == "/root/.cache/modelscope"


def test_container_mount_for_local_mounts_parent_dir():
    item = {"source": "Local", "name": "llama", "path": "/mnt/models/llama3-8b"}
    host_dir, cpath = _container_mount_for(item)
    assert host_dir == "/mnt/models"
    assert cpath == "/mnt/models/llama3-8b"


# -----------------------------------------------------------------------------
# _step_docker_args 默认值含 --gpus all（vllm 推理常踩坑）
# -----------------------------------------------------------------------------

def test_step_docker_args_default_includes_gpu_flags(monkeypatch):
    w = _wizard_with(["/m", "--host", "0.0.0.0"])
    captured: dict[str, str] = {}

    def fake_editable_simple(title, default):
        captured["default"] = default
        return default

    monkeypatch.setattr("llm_bench.interactive._editable_simple", fake_editable_simple)
    w._step_docker_args()
    assert "--gpus all" in captured["default"]
    assert "--shm-size 16g" in captured["default"]
    assert "--ipc=host" in captured["default"]
    assert w.config.backend.docker_args[:2] == ["--gpus", "all"]


def test_step_docker_args_preserves_existing(monkeypatch):
    w = _wizard_with(["/m"])
    w.config.backend.docker_args = ["--gpus", "1", "--runtime", "nvidia"]
    captured: dict[str, str] = {}
    monkeypatch.setattr("llm_bench.interactive._editable_simple", lambda t, d: captured.setdefault("d", d))
    w._step_docker_args()
    assert "--gpus 1" in captured["d"]
    assert "--runtime nvidia" in captured["d"]


def test_raw_editable_uses_horizontal_viewport(monkeypatch):
    class FakeInput:
        def read(self, count: int) -> str:
            return "\n"

        def isatty(self) -> bool:
            return True

        def fileno(self) -> int:
            return 0

    class FakeOutput:
        def __init__(self) -> None:
            self.parts: list[str] = []

        def write(self, text: str) -> None:
            self.parts.append(text)

        def flush(self) -> None:
            pass

        def isatty(self) -> bool:
            return True

    @contextmanager
    def fake_raw_terminal():
        yield

    output = FakeOutput()
    monkeypatch.setattr("llm_bench.interactive._raw_terminal", fake_raw_terminal)
    monkeypatch.setattr(
        "shutil.get_terminal_size",
        lambda *args, **kwargs: os.terminal_size((20, 20)),
    )
    monkeypatch.setattr("sys.stdin", FakeInput())
    monkeypatch.setattr("sys.stdout", output)

    assert _raw_editable("edit> ", "abcdefghijklmnopqrstuvwxyz") == "abcdefghijklmnopqrstuvwxyz"
    rendered = "".join(output.parts)
    assert "abcdefghijklmnopqrstuvwxyz" not in rendered
    assert "opqrstuvwxyz" in rendered
