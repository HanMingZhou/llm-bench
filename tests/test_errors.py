from llm_bench.errors import classify_error


def test_classify_error():
    assert classify_error("CUDA out of memory") == "oom"
    assert classify_error("address already in use") == "port_in_use"
    assert classify_error("health check timed out") == "timeout"
    assert classify_error("executable file not found") == "command_not_found"
    assert classify_error("HTTP Error 404 status 404: model not found") == "http_404"
    assert classify_error("HTTP Error 429 status 429: rate limit") == "http_429"
    assert classify_error("pull access denied for image") == "image_missing"
    assert classify_error("tokenizer load failed: tokenizer.json not found") == "tokenizer_load_error"
    assert classify_error("model load failed: can't load model checkpoint") == "model_load_error"
    assert classify_error("CUDA driver initialization failed") == "cuda_runtime"
    assert classify_error("NCCL unhandled system error") == "nccl_error"
