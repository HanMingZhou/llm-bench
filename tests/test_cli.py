import argparse

from llm_bench import cli
from llm_bench.cli import _split_passthrough, build_parser, main


def test_keyboard_interrupt_exits_cleanly(monkeypatch, capsys):
    parser = argparse.ArgumentParser(prog="llm-bench")
    sub = parser.add_subparsers(dest="command")
    boom = sub.add_parser("boom")

    def raise_keyboard_interrupt(_args):
        raise KeyboardInterrupt

    boom.set_defaults(func=raise_keyboard_interrupt)
    monkeypatch.setattr(cli, "build_parser", lambda: parser)

    try:
        cli.main(["boom"])
    except SystemExit as exc:
        assert exc.code == 130
    else:
        raise AssertionError("expected SystemExit")

    captured = capsys.readouterr()
    assert captured.err == "interrupted by user\n"


def test_top_level_commands_registered():
    parser = build_parser()
    choices = parser._subparsers._group_actions[0].choices
    assert {
        "infer",
        "wizard",
        "check",
        "report",
        "list",
        "show",
        "compare",
        "gate",
        "baseline",
        "config",
        "comm",
        "cleanup",
        "self-test",
    }.issubset(set(choices))


def test_top_level_help_does_not_print_long_command_choices():
    parser = build_parser()
    help_text = parser.format_help()
    assert "{infer,wizard,check" not in help_text
    assert "usage: llm-bench [-h] command ..." in help_text


def test_split_passthrough_returns_both_sides():
    pre, post = _split_passthrough(
        ["infer", "--backend", "vllm", "--", "vllm", "serve", "/model"]
    )
    assert pre == ["infer", "--backend", "vllm"]
    assert post == ["vllm", "serve", "/model"]


def test_split_passthrough_returns_empty_post_when_no_separator():
    pre, post = _split_passthrough(["infer", "--backend", "vllm"])
    assert pre == ["infer", "--backend", "vllm"]
    assert post == []


def test_infer_parser_accepts_core_flags():
    parser = build_parser()
    args = parser.parse_args(
        [
            "infer",
            "--backend",
            "vllm",
            "--image",
            "vllm/vllm-openai:latest",
            "--model-name",
            "Qwen/Qwen2.5-7B-Instruct",
            "--port",
            "8000",
            "--workload-profile",
            "quick",
            "--no-save-request-metrics",
            "--include-samples",
            "--temperature",
            "0.2",
            "--top-p",
            "0.9",
            "--docker-arg=--shm-size=16g",
            "--keep-container",
        ]
    )
    assert args.backend == "vllm"
    assert args.image == "vllm/vllm-openai:latest"
    assert args.model_name == "Qwen/Qwen2.5-7B-Instruct"
    assert args.port == 8000
    assert args.workload_profile == "quick"
    assert args.save_request_metrics is False
    assert args.include_samples is True
    assert args.temperature == 0.2
    assert args.top_p == 0.9
    assert args.docker_arg == ["--shm-size=16g"]
    assert args.keep_container is True


def test_infer_rejects_unknown_backend():
    parser = build_parser()
    try:
        parser.parse_args(["infer", "--backend", "auto"])
    except SystemExit:
        return
    raise AssertionError("backend auto should not be accepted")


def test_infer_accepts_transformers_backend_options():
    parser = build_parser()
    args = parser.parse_args(
        [
            "infer",
            "--backend",
            "transformers",
            "--model-path",
            "/mnt/models/qwen",
            "--torch-dtype",
            "bfloat16",
            "--device-map",
            "cuda:0",
            "--quantization",
            "4bit",
            "--trust-remote-code",
            "--do-sample",
            "--temperature",
            "0.7",
            "--top-p",
            "0.9",
            "--top-k",
            "50",
            "--repetition-penalty",
            "1.1",
            "--num-beams",
            "1",
            "--batch-size",
            "2",
        ]
    )
    assert args.backend == "transformers"
    assert args.model_path == "/mnt/models/qwen"
    assert args.torch_dtype == "bfloat16"
    assert args.device_map == "cuda:0"
    assert args.quantization == "4bit"
    assert args.trust_remote_code is True
    assert args.do_sample is True
    assert args.temperature == 0.7
    assert args.top_p == 0.9
    assert args.top_k == 50
    assert args.repetition_penalty == 1.1
    assert args.num_beams == 1
    assert args.batch_size == 2


def test_main_attaches_passthrough_to_args(monkeypatch):
    captured = {}

    def fake_func(args):
        captured["passthrough"] = args.passthrough
        captured["model_name"] = args.model_name

    monkeypatch.setattr(
        cli,
        "build_parser",
        lambda: _parser_with_func(fake_func),
    )
    main(
        [
            "infer",
            "--backend",
            "vllm",
            "--image",
            "vllm:test",
            "--model-name",
            "Qwen/Qwen2.5-7B-Instruct",
            "--",
            "vllm",
            "serve",
            "/models/qwen",
            "--host",
            "0.0.0.0",
        ]
    )
    assert captured["passthrough"] == [
        "vllm", "serve", "/models/qwen", "--host", "0.0.0.0",
    ]
    assert captured["model_name"] == "Qwen/Qwen2.5-7B-Instruct"


def _parser_with_func(func):
    parser = argparse.ArgumentParser(prog="llm-bench")
    sub = parser.add_subparsers(dest="command")
    infer = sub.add_parser("infer")
    infer.add_argument("--backend")
    infer.add_argument("--image")
    infer.add_argument("--model-name")
    infer.set_defaults(func=func)
    return parser


def test_self_test_parser_accepts_prompt_dir_command():
    parser = build_parser()
    args = parser.parse_args(
        [
            "self-test",
            "--prompt-dir",
            "examples/prompts",
            "--concurrency",
            "1",
            "--total-requests",
            "3",
        ]
    )
    assert args.command == "self-test"
    assert args.prompt_dir == "examples/prompts"


def test_comm_all_reduce_accepts_docker_arg():
    parser = build_parser()
    args = parser.parse_args(
        [
            "comm",
            "all-reduce",
            "--image",
            "nccl-tests:latest",
            "--docker-arg=--gpus=all",
            "--docker-arg=--shm-size=16g",
            "--docker-arg=--ipc=host",
        ]
    )
    assert args.image == "nccl-tests:latest"
    assert args.docker_arg == ["--gpus=all", "--shm-size=16g", "--ipc=host"]
