from __future__ import annotations

from typing import Callable

from llm_bench.config import BenchConfig


def build_token_counter(config: BenchConfig) -> Callable[[str], int] | None:
    """Try to build an exact token counter by loading the tokenizer locally.

    With the new thin-wrapper design the tool does not know the local model
    path on host. We try the HuggingFace model id from `backend.model_name`
    against the host HF cache; if `transformers` is not installed or the
    tokenizer is not cached, we return None and let workloads fall back to
    word-count estimation.
    """
    model_name = config.backend.model_name
    if not model_name:
        return None
    try:
        from transformers import AutoTokenizer
    except ImportError:
        return None
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
    except Exception:
        return None

    def count(text: str) -> int:
        return len(tokenizer.encode(text, add_special_tokens=False))

    return count
