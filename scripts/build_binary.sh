#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

# Sanity check: we MUST be in the project root (pyproject.toml + llm_bench/).
# Otherwise setuptools falls back to a 0.0.0 "UNKNOWN" package and the
# subsequent PyInstaller spec resolves `Path.cwd() / llm_bench / __main__.py`
# to a non-existent file.
if [[ ! -f pyproject.toml || ! -d llm_bench ]]; then
  echo "build_binary.sh: project root not found (no pyproject.toml + llm_bench/)" >&2
  echo "  cwd: $PWD" >&2
  exit 2
fi
# Pull the project name out of pyproject.toml so an obviously-broken file
# (missing [project] section, wrong path mirrored from another repo) fails
# fast instead of producing UNKNOWN-0.0.0 wheels later.
if ! grep -qE '^name\s*=\s*"autoctl-llm-bench"' pyproject.toml; then
  echo "build_binary.sh: pyproject.toml is missing the autoctl-llm-bench [project] name; refusing to build." >&2
  echo "  hint: are you running the script from a stale clone? cwd=$PWD" >&2
  exit 2
fi

target="current"
fresh_venv=0
# 定义清华源镜像地址
PIP_INDEX_URL="https://pypi.tuna.tsinghua.edu.cn/simple"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target)
      target="${2:?missing value for --target}"
      shift 2
      ;;
    --fresh-venv)
      fresh_venv=1
      shift
      ;;
    --help|-h)
      cat <<'EOF'
Usage:
  bash scripts/build_binary.sh
  bash scripts/build_binary.sh --target current
  bash scripts/build_binary.sh --target linux/amd64
  bash scripts/build_binary.sh --fresh-venv

Targets:
  current       Build for the current OS/CPU with PyInstaller.
  linux/amd64   Build a Linux x86_64 binary through Docker.
  linux/arm64   Build a Linux arm64 binary through Docker.

Build environment selection (for "current" target):
  1. If $VIRTUAL_ENV is set, use that python. PyInstaller is auto-installed.
  2. Otherwise fallback to uv-run.
  3. Otherwise fallback to system python3 + pip.
EOF
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

host_os="$(uname -s)"
host_arch="$(uname -m)"

echo "host: ${host_os}/${host_arch}"
echo "target: ${target}"

case "${target}" in
  current)
    echo "building for current platform with PyInstaller"
    if [[ "${fresh_venv}" -eq 0 && -n "${VIRTUAL_ENV:-}" ]]; then
      py="${VIRTUAL_ENV}/bin/python"
      echo "using activated venv: ${VIRTUAL_ENV}"
      if ! "${py}" -c 'import PyInstaller' >/dev/null 2>&1; then
        echo "PyInstaller not found in venv; installing via Tsinghua mirror..."
        # 1. 局部 venv 安装时使用清华源
        "${py}" -m pip install -i "${PIP_INDEX_URL}" --quiet "pyinstaller>=6.0"
      fi
      "${py}" -m PyInstaller --clean --noconfirm packaging/llm-bench.spec
    elif command -v uv >/dev/null 2>&1; then
      echo "using uv (no active venv detected, or --fresh-venv set)"
      # 2. uv 运行时使用 --index-url 参数指定清华源
      uv run --index-url "${PIP_INDEX_URL}" --extra binary pyinstaller --clean --noconfirm packaging/llm-bench.spec
    else
      echo "using system python3 (no venv, no uv)"
      # 3. 系统 python3 安装时使用清华源
      python3 -m pip install -i "${PIP_INDEX_URL}" ".[binary]"
      python3 -m PyInstaller --clean --noconfirm packaging/llm-bench.spec
    fi
    ;;
  linux/amd64|linux/arm64)
    if ! command -v docker >/dev/null 2>&1; then
      echo "docker is required for --target ${target}" >&2
      exit 1
    fi
    echo "building Linux binary in Docker for ${target}"
    # 4. Docker 内部通过环境变量 PIP_INDEX_URL 全局配置清华源
    docker run --rm \
      --platform "${target}" \
      -v "$PWD:/work" \
      -w /work \
      -e PIP_INDEX_URL="${PIP_INDEX_URL}" \
      python:3.12-slim \
      bash -lc '
        python -m pip install --upgrade pip
        python -m pip install ".[binary]"
        python -m PyInstaller --clean --noconfirm packaging/llm-bench.spec
      '
    ;;
  *)
    echo "unsupported target: ${target}" >&2
    exit 2
    ;;
esac

echo "binary: dist/llm-bench"
file dist/llm-bench || true