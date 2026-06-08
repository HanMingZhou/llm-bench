# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_submodules
from pathlib import Path


project_root = Path.cwd()
hiddenimports = collect_submodules("llm_bench")

# Deliberately exclude heavy ML / GPU deps from the dist binary even when
# they happen to be installed in the build env (e.g. you ran the build
# inside a venv that has torch for the transformers backend). Reasons:
#   - torch+CUDA libs alone are >= 1.5 GB; including them would push the
#     single-file binary past 2 GB.
#   - vllm/sglang backends never import torch from the binary itself - the
#     model runs inside the docker container.
#   - The transformers backend is documented as "use the source + venv";
#     `python -m llm_bench` is the supported entrypoint there.
# matplotlib stays in because the Markdown report's chart rendering needs
# it; that costs ~30 MB and is worth it.
HEAVY_ML_EXCLUDES = [
    "torch",
    "torchvision",
    "torchaudio",
    "transformers",
    "tokenizers",
    "accelerate",
    "bitsandbytes",
    "safetensors",
    "triton",
    "huggingface_hub",
    "sentencepiece",
    "datasets",
    "peft",
    "vllm",
    "sglang",
]

a = Analysis(
    [str(project_root / "llm_bench" / "__main__.py")],
    pathex=[str(project_root)],
    binaries=[],
    datas=[],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=HEAVY_ML_EXCLUDES,
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="llm-bench",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
