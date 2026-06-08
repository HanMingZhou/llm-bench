from __future__ import annotations

import json
import fnmatch
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from llm_bench.config import BenchConfig


@dataclass
class WorkloadRequest:
    prompt: str
    input_tokens: int
    output_tokens: int
    metadata: dict[str, Any] = field(default_factory=dict)
    messages: list[dict[str, str]] | None = None


def build_workload_requests(
    config: BenchConfig,
    token_counter: Callable[[str], int] | None = None,
) -> list[WorkloadRequest]:
    if config.workload.prompt_jsonl:
        return load_jsonl_workload(Path(config.workload.prompt_jsonl), config, token_counter)
    if config.workload.prompt_dir:
        return load_prompt_dir_workload(Path(config.workload.prompt_dir), config, token_counter)
    requests = []
    for input_tokens in config.workload.input_tokens:
        prompt = build_prompt(input_tokens, token_counter)
        measured_tokens = token_counter(prompt) if token_counter else input_tokens
        for output_tokens in config.workload.output_tokens:
            requests.append(
                WorkloadRequest(
                    prompt=prompt,
                    input_tokens=measured_tokens,
                    output_tokens=output_tokens,
                    metadata={"source": "synthetic", "target_input_tokens": input_tokens},
                )
            )
    return requests


def load_jsonl_workload(
    path: Path,
    config: BenchConfig,
    token_counter: Callable[[str], int] | None = None,
) -> list[WorkloadRequest]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows: list[WorkloadRequest] = []
    default_output = config.workload.output_tokens[0] if config.workload.output_tokens else 128
    with path.open("r", encoding="utf-8") as f:
        for index, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{index}: invalid JSON ({exc})") from exc
            prompt = _extract_prompt(data)
            messages = _extract_messages(data)
            max_tokens = int(data.get("max_tokens") or data.get("output_tokens") or default_output)
            metadata = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
            metadata = {
                **metadata,
                "source": "jsonl",
                "jsonl_line": index,
            }
            rows.append(
                WorkloadRequest(
                    prompt=prompt,
                    input_tokens=token_counter(prompt) if token_counter else _estimate_tokens(prompt),
                    output_tokens=max_tokens,
                    metadata=metadata,
                    messages=messages,
                )
            )
    if not rows:
        raise ValueError(f"No workload rows found in {path}")
    return rows


def load_prompt_dir_workload(
    path: Path,
    config: BenchConfig,
    token_counter: Callable[[str], int] | None = None,
) -> list[WorkloadRequest]:
    if not path.exists():
        raise FileNotFoundError(path)
    if not path.is_dir():
        raise NotADirectoryError(path)
    rows: list[WorkloadRequest] = []
    default_output = config.workload.output_tokens[0] if config.workload.output_tokens else 128
    files = _prompt_files(path, config)
    for file_path in files:
        if file_path.suffix.lower() == ".jsonl":
            rows.extend(load_jsonl_workload(file_path, config, token_counter))
            continue
        if file_path.suffix.lower() == ".json":
            try:
                data = json.loads(file_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{file_path}: invalid JSON ({exc})") from exc
            prompt = _extract_prompt(data)
            messages = _extract_messages(data)
            output_tokens = int(data.get("max_tokens") or data.get("output_tokens") or default_output)
            metadata = data.get("metadata") if isinstance(data.get("metadata"), dict) else {}
        else:
            prompt = file_path.read_text(encoding="utf-8")
            messages = None
            output_tokens = default_output
            metadata = {}
        metadata = {
            **metadata,
            "source": "prompt-dir",
            "prompt_file": str(file_path),
        }
        rows.append(
            WorkloadRequest(
                prompt=prompt,
                input_tokens=token_counter(prompt) if token_counter else _estimate_tokens(prompt),
                output_tokens=output_tokens,
                metadata=metadata,
                messages=messages,
            )
        )
    if not rows:
        raise ValueError(f"No prompt files found in {path}")
    return rows


def build_prompt(input_tokens: int, token_counter: Callable[[str], int] | None = None) -> str:
    base = _base_long_text()
    if token_counter is None:
        words = base.split()
        repeats = max(input_tokens // max(len(words), 1) + 1, 1)
        return " ".join((words * repeats)[: max(input_tokens, 1)])

    prompt = base
    while token_counter(prompt) < input_tokens:
        prompt += "\n\n" + base
    tokens = token_counter(prompt)
    if tokens <= input_tokens * 1.05:
        return prompt

    # Binary search by word count. It is not perfect for every tokenizer, but keeps prompts close.
    words = prompt.split()
    low, high = 1, len(words)
    best = prompt
    while low <= high:
        mid = (low + high) // 2
        candidate = " ".join(words[:mid])
        count = token_counter(candidate)
        if count <= input_tokens:
            best = candidate
            low = mid + 1
        else:
            high = mid - 1
    return best


def _extract_prompt(data: dict[str, Any]) -> str:
    if "prompt" in data:
        return str(data["prompt"])
    if "messages" in data and isinstance(data["messages"], list):
        parts = []
        for message in data["messages"]:
            if isinstance(message, dict):
                role = message.get("role", "user")
                content = message.get("content", "")
                parts.append(f"{role}: {content}")
        return "\n".join(parts)
    raise ValueError("JSONL row must include prompt or messages")


def _extract_messages(data: dict[str, Any]) -> list[dict[str, str]] | None:
    messages = data.get("messages")
    if not isinstance(messages, list):
        return None
    normalized = []
    for message in messages:
        if isinstance(message, dict):
            normalized.append(
                {
                    "role": str(message.get("role", "user")),
                    "content": str(message.get("content", "")),
                }
            )
    return normalized or None


def _prompt_files(path: Path, config: BenchConfig) -> list[Path]:
    includes = [p.strip() for p in config.workload.prompt_include.split(",") if p.strip()]
    excludes = [p.strip() for p in config.workload.prompt_exclude.split(",") if p.strip()]
    iterator = path.rglob("*") if config.workload.prompt_dir_recursive else path.iterdir()
    files = []
    for candidate in iterator:
        if not candidate.is_file():
            continue
        rel = str(candidate.relative_to(path))
        if includes and not any(fnmatch.fnmatch(candidate.name, pat) or fnmatch.fnmatch(rel, pat) for pat in includes):
            continue
        if excludes and any(fnmatch.fnmatch(candidate.name, pat) or fnmatch.fnmatch(rel, pat) for pat in excludes):
            continue
        files.append(candidate)
    return sorted(files)


def _estimate_tokens(prompt: str) -> int:
    return max(len(prompt.split()), 1)


def _base_long_text() -> str:
    return (
        "This benchmark prompt contains a realistic mixture of operational notes, "
        "system observations, incident context, configuration details, and user requests. "
        "The goal is to create stable input pressure for prefill, attention, and KV cache behavior. "
        "A production inference service often receives long documents, logs, source code snippets, "
        "customer support histories, retrieval augmented context, and structured instructions. "
        "The assistant should read the supplied material, preserve important facts, identify risks, "
        "summarize tradeoffs, and produce a concise answer grounded in the provided context."
    )
