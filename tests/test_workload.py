from pathlib import Path

from llm_bench.config import BenchConfig
from llm_bench.config import from_mapping
from llm_bench.workload import build_prompt, build_workload_requests


def test_build_prompt_token_counter_exact_words():
    def counter(text: str) -> int:
        return len(text.split())

    assert counter(build_prompt(128, counter)) == 128


def test_prompt_dir_recursive_filter(tmp_path: Path):
    (tmp_path / "a.md").write_text("root prompt", encoding="utf-8")
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "b.txt").write_text("nested prompt", encoding="utf-8")
    (nested / "skip.log").write_text("skip", encoding="utf-8")

    config = BenchConfig()
    config.workload.prompt_dir = str(tmp_path)
    config.workload.prompt_include = "*.txt,*.md"
    rows = build_workload_requests(config)

    files = {Path(row.metadata["prompt_file"]).name for row in rows}
    assert files == {"a.md", "b.txt"}


def test_json_prompt_file(tmp_path: Path):
    (tmp_path / "request.json").write_text(
        '{"prompt": "hello", "max_tokens": 17, "metadata": {"case": "x"}}',
        encoding="utf-8",
    )
    config = BenchConfig()
    config.workload.prompt_dir = str(tmp_path)
    rows = build_workload_requests(config)
    assert rows[0].output_tokens == 17
    assert rows[0].metadata["case"] == "x"


def test_yaml_scalar_workload_values_are_normalized():
    config = from_mapping({"workload": {"input_tokens": 512, "output_tokens": 128, "concurrency": 4}})

    assert config.workload.input_tokens == [512]
    assert config.workload.output_tokens == [128]
    assert config.workload.concurrency == [4]
