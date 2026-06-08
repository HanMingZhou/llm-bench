import types

from llm_bench.backends.transformers_backend import _from_pretrained_kwargs, _generate_defaults, _torch_dtype
from llm_bench.config import TransformersConfig


def _fake_torch():
    # Minimal stand-in exposing the dtype attributes the backend reads.
    return types.SimpleNamespace(float16="f16", bfloat16="bf16", float32="f32")


def test_torch_dtype_mapping():
    torch = _fake_torch()
    assert _torch_dtype(torch, "float16") == "f16"
    assert _torch_dtype(torch, "bfloat16") == "bf16"
    assert _torch_dtype(torch, "float32") == "f32"
    # Unknown falls back to bfloat16.
    assert _torch_dtype(torch, "weird") == "bf16"


def test_from_pretrained_kwargs_use_upstream_names():
    tx = TransformersConfig(
        torch_dtype="bfloat16",
        device_map="cuda:0",
        trust_remote_code=True,
        revision="main",
        low_cpu_mem_usage=True,
    )
    kwargs = _from_pretrained_kwargs(tx, _fake_torch())
    assert set(kwargs) >= {"torch_dtype", "device_map", "trust_remote_code", "revision", "low_cpu_mem_usage"}
    assert kwargs["device_map"] == "cuda:0"
    assert kwargs["trust_remote_code"] is True
    assert kwargs["torch_dtype"] == "bf16"


def test_from_pretrained_kwargs_quantization_4bit():
    tx = TransformersConfig(quantization="4bit")
    kwargs = _from_pretrained_kwargs(tx, _fake_torch())
    assert kwargs.get("load_in_4bit") is True
    assert "load_in_8bit" not in kwargs


def test_from_pretrained_kwargs_quantization_8bit():
    tx = TransformersConfig(quantization="int8")
    kwargs = _from_pretrained_kwargs(tx, _fake_torch())
    assert kwargs.get("load_in_8bit") is True
    assert "load_in_4bit" not in kwargs


def test_from_pretrained_kwargs_no_quantization():
    tx = TransformersConfig(quantization="")
    kwargs = _from_pretrained_kwargs(tx, _fake_torch())
    assert "load_in_4bit" not in kwargs
    assert "load_in_8bit" not in kwargs


def test_generate_defaults_pull_sampling_from_workload():
    tx = TransformersConfig(do_sample=True, top_k=40, repetition_penalty=1.2, num_beams=2)
    defaults = _generate_defaults(tx, temperature=0.7, top_p=0.95)
    assert defaults == {
        "do_sample": True,
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": 40,
        "repetition_penalty": 1.2,
        "num_beams": 2,
    }
