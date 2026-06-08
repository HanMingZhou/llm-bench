from __future__ import annotations


def classify_error(message: object) -> str:
    text = str(message or "").lower()
    if not text:
        return "unknown"
    if "out of memory" in text or "cuda oom" in text or "oom" in text:
        return "oom"
    if "nccl" in text and ("error" in text or "failed" in text or "unhandled" in text):
        return "nccl_error"
    if "cuda" in text and ("driver" in text or "runtime" in text or "initialization" in text):
        return "cuda_runtime"
    if "address already in use" in text or "port is already allocated" in text:
        return "port_in_use"
    if "timed out" in text or "timeout" in text:
        return "timeout"
    if "connection refused" in text or "connection reset" in text:
        return "connection_error"
    if "docker daemon" in text or "cannot connect to the docker" in text:
        return "docker_daemon"
    if "docker" in text and ("permission denied" in text or "got permission denied" in text):
        return "docker_permission"
    if "image" in text and ("not found" in text or "no such image" in text):
        return "image_missing"
    if "pull access denied" in text or "repository does not exist" in text:
        return "image_missing"
    if "health check" in text:
        return "health_check_timeout"
    if "executable file not found" in text or "command not found" in text:
        return "command_not_found"
    if "http error 429" in text or "status 429" in text:
        return "http_429"
    if "http error 404" in text or "status 404" in text:
        return "http_404"
    if "http error 400" in text or "status 400" in text:
        return "http_400"
    if "http error 5" in text or "status 5" in text:
        return "http_5xx"
    if "http error 4" in text or "status 4" in text:
        return "http_4xx"
    if "config.json" in text and ("not found" in text or "does not exist" in text):
        return "model_config_missing"
    if "tokenizer" in text and ("not found" in text or "does not exist" in text or "can't load" in text):
        return "tokenizer_load_error"
    if "can't load" in text and ("model" in text or "checkpoint" in text):
        return "model_load_error"
    if "safetensors" in text or "pytorch_model.bin" in text or ("weight" in text and "not found" in text):
        return "model_weights_missing"
    if "no such file" in text or "not found" in text:
        return "not_found"
    if "invalid mount config" in text or ("mount" in text and "error" in text):
        return "mount_error"
    if "container" in text and "exited" in text:
        return "container_exit"
    return "other"
