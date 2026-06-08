from __future__ import annotations

import os
import socket
import shutil
import subprocess
from pathlib import Path
from typing import Any

from llm_bench.config import BenchConfig


def inspect_runtime(config: BenchConfig) -> dict[str, Any]:
    docker_required = config.backend.name in {"vllm", "sglang"}
    gpu_required = config.backend.name in {"vllm", "sglang", "transformers"}
    docker = (
        inspect_docker(config.backend.image)
        if docker_required
        else {"installed": True, "daemon_ok": True, "image_exists": True, "ok": True, "error": None, "image": ""}
    )
    port = (
        inspect_port("0.0.0.0", config.backend.port)
        if docker_required
        else {"available": True}
    )
    gpu = inspect_gpu()
    disk = inspect_disk(config.report.output_dir)
    return {
        "docker": docker,
        "port": port,
        "gpu": gpu,
        "disk": disk,
        "docker_required": docker_required,
        "gpu_required": gpu_required,
    }


def enforce_runtime_requirements(config: BenchConfig, runtime: dict[str, Any]) -> None:
    if config.skip_env_check or config.backend.name == "dry-run":
        return
    failures: list[str] = []
    docker = runtime.get("docker") or {}
    port = runtime.get("port") or {}
    gpu = runtime.get("gpu") or {}
    disk = runtime.get("disk") or {}
    if config.backend.name in {"vllm", "sglang"}:
        if not docker.get("installed"):
            failures.append("docker is not installed or not found in PATH")
        elif not docker.get("daemon_ok"):
            failures.append(f"docker daemon is not available: {docker.get('error')}")
        elif config.backend.image and not docker.get("image_exists"):
            failures.append(f"docker image does not exist locally: {config.backend.image}")
        if not port.get("available"):
            failures.append(f"port is not available: 0.0.0.0:{config.backend.port}")
    needs_gpu = config.backend.name in {"vllm", "sglang", "transformers"}
    if config.backend.name == "transformers" and "cpu" in (config.transformers.device_map or "").lower():
        needs_gpu = False
    if needs_gpu and not gpu.get("gpu_available"):
        failures.append("GPU is not visible through nvidia-smi")
    if disk and disk.get("free_gb", 0) < 1:
        failures.append(f"output disk free space is too low: {disk.get('free_gb')} GB")
    if failures:
        raise RuntimeError("; ".join(failures))


def inspect_docker(image: str) -> dict[str, Any]:
    docker_path = shutil.which("docker")
    result: dict[str, Any] = {
        "installed": docker_path is not None,
        "path": docker_path,
        "version": None,
        "daemon_ok": False,
        "image": image,
        "image_exists": False,
        "ok": False,
        "error": None,
    }
    if not docker_path:
        result["error"] = "docker command not found"
        return result

    version = _run([docker_path, "--version"])
    result["version"] = version["stdout"].strip() if version["returncode"] == 0 else None
    ps = _run([docker_path, "ps"])
    result["daemon_ok"] = ps["returncode"] == 0
    if ps["returncode"] != 0:
        result["error"] = ps["stderr"].strip() or ps["stdout"].strip()
        return result

    if image:
        inspect = _run([docker_path, "image", "inspect", image])
        result["image_exists"] = inspect["returncode"] == 0
        if inspect["returncode"] != 0:
            result["error"] = inspect["stderr"].strip() or inspect["stdout"].strip()
    else:
        result["image_exists"] = True

    result["ok"] = result["installed"] and result["daemon_ok"] and result["image_exists"]
    return result


def discover_docker_images(backend: str = "", limit: int = 50) -> list[dict[str, str]]:
    docker_path = shutil.which("docker")
    if not docker_path:
        return []
    proc = _run([docker_path, "image", "ls", "--format", "{{.Repository}}:{{.Tag}}\t{{.ID}}\t{{.Size}}"])
    if proc["returncode"] != 0:
        return []

    images: list[dict[str, str]] = []
    seen: set[str] = set()
    for line in proc["stdout"].splitlines():
        parts = line.split("\t")
        if not parts:
            continue
        name = parts[0].strip()
        if not name or name.startswith("<none>:") or name in seen:
            continue
        seen.add(name)
        images.append(
            {
                "name": name,
                "id": parts[1].strip() if len(parts) > 1 else "",
                "size": parts[2].strip() if len(parts) > 2 else "",
            }
        )

    return sorted(images, key=lambda image: (_image_rank(image["name"], backend), image["name"]))[:limit]


def inspect_port(host: str, port: int) -> dict[str, Any]:
    bind_host = "0.0.0.0" if host in {"0.0.0.0", "::"} else host
    result = {"host": host, "port": port, "available": False, "error": None}
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((bind_host, int(port)))
        result["available"] = True
    except OSError as exc:
        result["error"] = str(exc)
    finally:
        sock.close()
    return result


def _image_rank(image: str, backend: str) -> int:
    text = image.lower()
    backend_text = backend.lower()
    if backend_text and backend_text in text:
        return 0
    if backend_text == "vllm" and "openai" in text:
        return 1
    if backend_text == "sglang" and "sgl" in text:
        return 1
    if "nccl" in text:
        return 3
    return 2


def inspect_gpu() -> dict[str, Any]:
    nvidia_smi = shutil.which("nvidia-smi")
    result: dict[str, Any] = {
        "nvidia_smi": nvidia_smi,
        "gpu_available": False,
        "gpu_count": 0,
        "gpus": [],
        "error": None,
    }
    if not nvidia_smi:
        result["error"] = "nvidia-smi command not found"
        return result
    proc = _run([nvidia_smi, "--query-gpu=index,name,memory.total", "--format=csv,noheader,nounits"])
    if proc["returncode"] != 0:
        result["error"] = proc["stderr"].strip() or proc["stdout"].strip()
        return result
    gpus = []
    for line in proc["stdout"].splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 3:
            gpus.append({"index": parts[0], "name": parts[1], "memory_total_mb": parts[2]})
    result["gpus"] = gpus
    result["gpu_count"] = len(gpus)
    result["gpu_available"] = bool(gpus)
    return result


def inspect_disk(path: str) -> dict[str, Any]:
    target = Path(path)
    existing = target if target.exists() else target.parent
    while not existing.exists() and existing != existing.parent:
        existing = existing.parent
    usage = shutil.disk_usage(existing)
    return {
        "path": str(path),
        "free_gb": round(usage.free / (1024**3), 3),
        "total_gb": round(usage.total / (1024**3), 3),
        "used_gb": round(usage.used / (1024**3), 3),
    }


def inspect_model(model_path: str, model_name: str) -> dict[str, Any]:
    candidates = model_candidates(model_path, model_name)
    discovered: list[dict[str, str]] = []

    if not model_path and not model_name:
        discovered = discover_model_paths(limit=20, roots=[Path.cwd()])
        if len(discovered) == 1:
            return {
                "exists": True,
                "resolved_path": discovered[0]["path"],
                "searched_paths": [str(Path.cwd())],
                "candidate_models": discovered,
            }
        if discovered:
            return {
                "exists": False,
                "resolved_path": None,
                "searched_paths": [str(Path.cwd())],
                "candidate_models": discovered,
                "error": "model is ambiguous in current directory",
            }

    for candidate in candidates:
        if _looks_like_model_dir(candidate):
            return {
                "exists": True,
                "resolved_path": str(candidate),
                "searched_paths": [str(p) for p in candidates],
                "candidate_models": [],
            }
        if candidate.exists() and candidate.is_dir():
            nested = discover_model_paths(limit=20, roots=[candidate])
            discovered.extend(nested)
            if len(nested) == 1:
                return {
                    "exists": True,
                    "resolved_path": nested[0]["path"],
                    "searched_paths": [str(p) for p in candidates],
                    "candidate_models": nested,
                }
    return {
        "exists": False,
        "resolved_path": None,
        "searched_paths": [str(p) for p in candidates],
        "candidate_models": discovered,
    }


def discover_model_paths(limit: int = 50, roots: list[Path] | None = None) -> list[dict[str, str]]:
    candidates: list[dict[str, str]] = []
    seen: set[str] = set()
    for root in roots or _default_model_roots():
        for path in _walk_model_dirs(root):
            key = _path_key(path)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(
                {
                    "source": _model_source(path),
                    "name": _model_name(path),
                    "path": str(path),
                }
            )
            if len(candidates) >= limit:
                return sorted(candidates, key=lambda item: (item["source"], item["name"], item["path"]))
    return sorted(candidates, key=lambda item: (item["source"], item["name"], item["path"]))


def model_candidates(model_path: str, model_name: str) -> list[Path]:
    home = Path.home()
    raw = model_path
    if not raw and model_name:
        model_name_path = Path(os.path.expandvars(os.path.expanduser(model_name)))
        if model_name_path.is_absolute() or model_name_path.exists():
            raw = model_name
    names = [n for n in [model_name] if n]
    if model_path:
        model_path_obj = Path(model_path)
        if not model_path_obj.is_absolute():
            names.append(model_path)
        path_name = Path(model_path).name
        if path_name and path_name != model_path:
            names.append(path_name)
    candidates: list[Path] = []
    if raw:
        expanded = Path(os.path.expandvars(os.path.expanduser(raw)))
        candidates.append(expanded)

    for name in names:
        normalized = name.strip("/")
        if not normalized:
            continue
        candidates.extend(
            [
                home / ".cache" / "modelscope" / "hub" / "models" / normalized,
                home / ".cache" / "modelscope" / "hub" / normalized,
                home / ".cache" / "huggingface" / "hub" / normalized,
            ]
        )
        if "/" in normalized:
            org, repo = normalized.split("/", 1)
            candidates.append(home / ".cache" / "huggingface" / "hub" / f"models--{org}--{repo}")

    expanded_candidates: list[Path] = []
    for candidate in candidates:
        snapshots = candidate / "snapshots"
        if candidate.name.startswith("models--") and snapshots.exists():
            expanded_candidates.extend(_snapshot_dirs(snapshots))
        expanded_candidates.append(candidate)

    deduped = []
    seen = set()
    for candidate in expanded_candidates:
        key = str(candidate)
        if key not in seen:
            deduped.append(candidate)
            seen.add(key)
    return deduped


def _default_model_roots() -> list[Path]:
    home = Path.home()
    roots: list[Path] = []
    env_roots = [
        os.environ.get("MODELSCOPE_CACHE"),
        os.environ.get("HUGGINGFACE_HUB_CACHE"),
        os.environ.get("TRANSFORMERS_CACHE"),
    ]
    if os.environ.get("HF_HOME"):
        env_roots.append(str(Path(os.environ["HF_HOME"]) / "hub"))
    if os.environ.get("LLM_BENCH_MODEL_DIRS"):
        env_roots.extend(os.environ["LLM_BENCH_MODEL_DIRS"].split(os.pathsep))
    env_roots.extend(
        [
            str(home / ".cache" / "modelscope" / "hub" / "models"),
            str(home / ".cache" / "modelscope" / "hub"),
            str(home / ".cache" / "huggingface" / "hub"),
            "/models",
            "/mnt/models",
            "/data/models",
        ]
    )
    seen: set[str] = set()
    for raw in env_roots:
        if not raw:
            continue
        path = Path(os.path.expandvars(os.path.expanduser(raw)))
        key = str(path)
        if key not in seen:
            roots.append(path)
            seen.add(key)
    return roots


def _walk_model_dirs(root: Path, max_depth: int = 4) -> list[Path]:
    root = Path(root)
    if not root.exists() or not root.is_dir():
        return []
    found: list[Path] = []
    stack: list[tuple[Path, int]] = [(root, 0)]
    ignored = {"blobs", ".locks", ".lock", "__pycache__"}
    while stack:
        current, depth = stack.pop()
        if _looks_like_model_dir(current):
            found.append(current)
            continue
        if depth >= max_depth:
            continue
        try:
            children = [child for child in current.iterdir() if child.is_dir()]
        except OSError:
            continue
        for child in sorted(children, reverse=True):
            if child.name in ignored:
                continue
            stack.append((child, depth + 1))
    return found


def _looks_like_model_dir(path: Path) -> bool:
    marker_names = {
        "config.json",
        "tokenizer.json",
        "tokenizer.model",
        "pytorch_model.bin",
        "model.safetensors",
    }
    try:
        names = {child.name for child in path.iterdir()}
    except OSError:
        return False
    if names & marker_names:
        return True
    return any(name.endswith((".safetensors", ".bin")) for name in names)


def _snapshot_dirs(snapshots: Path) -> list[Path]:
    try:
        dirs = [child for child in snapshots.iterdir() if child.is_dir()]
    except OSError:
        return []
    model_dirs = [child for child in dirs if _looks_like_model_dir(child)]
    return sorted(model_dirs or dirs, reverse=True)


def _path_key(path: Path) -> str:
    try:
        return str(path.resolve())
    except OSError:
        return str(path)


def _model_source(path: Path) -> str:
    text = str(path)
    if ".cache/huggingface" in text or "huggingface" in text:
        return "Hugging Face"
    if ".cache/modelscope" in text or "modelscope" in text:
        return "ModelScope"
    return "Local"


def _model_name(path: Path) -> str:
    for part in path.parts:
        if part.startswith("models--"):
            repo = part.removeprefix("models--").replace("--", "/")
            if path.parent.name == "snapshots":
                return f"{repo}@{path.name[:8]}"
            return repo
    if path.parent.name == "snapshots" and path.parent.parent.name.startswith("models--"):
        repo = path.parent.parent.name.removeprefix("models--").replace("--", "/")
        return f"{repo}@{path.name[:8]}"
    if path.parent.name not in {"hub", "models", "snapshots"}:
        return f"{path.parent.name}/{path.name}"
    return path.name


def _run(cmd: list[str]) -> dict[str, Any]:
    try:
        proc = subprocess.run(cmd, text=True, capture_output=True, timeout=10)
        return {
            "returncode": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
    except Exception as exc:
        return {
            "returncode": 1,
            "stdout": "",
            "stderr": str(exc),
        }
