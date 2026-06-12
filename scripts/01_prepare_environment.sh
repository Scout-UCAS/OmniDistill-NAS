#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00_common.sh"

print_step "Step 1: prepare project-local environment"

run_python - <<'PY'
from pathlib import Path

for path in [
    "vendor/python",
    "hf_cache/models",
    "hf_cache/datasets",
    "outputs",
    "checkpoints",
]:
    Path(path).mkdir(parents=True, exist_ok=True)
print("project_dirs_ok=True")
PY

missing="$(
run_python - <<'PY'
import importlib.util
import importlib.metadata

missing = []

def require_import(module_name, package_name=None):
    try:
        __import__(module_name)
    except Exception:
        missing.append(package_name or module_name)

if importlib.util.find_spec("torch") is None:
    missing.append("torch")
require_import("numpy")
try:
    import scipy  # noqa: F401
    from scipy.optimize import milp  # noqa: F401
except Exception:
    missing.append("scipy")
require_import("datasets")
require_import("PIL", "pillow")
for module_name, package_name in [
    ("accelerate", "accelerate"),
    ("safetensors", "safetensors"),
    ("sentencepiece", "sentencepiece"),
    ("timm", "timm"),
    ("einops", "einops"),
]:
    if importlib.util.find_spec(module_name) is None:
        missing.append(package_name)
if importlib.util.find_spec("transformers") is None:
    missing.append("transformers>=4.57,<5")
else:
    version = importlib.metadata.version("transformers")
    major, minor, *_ = [int(part) for part in version.split(".")[:2]]
    if (major, minor) < (4, 57):
        missing.append("transformers>=4.57,<5")
print(",".join(missing))
PY
)"

if [[ -n "${missing}" ]]; then
  print_step "Installing missing Python packages into vendor/python: ${missing}"
  if [[ "${missing}" == *torch* ]]; then
    echo "torch is missing. Install a CUDA-compatible torch in the base environment first." >&2
    exit 1
  fi
  run_python -m pip install \
    --target vendor/python \
    --upgrade \
    --force-reinstall \
    --no-cache-dir \
    "numpy" \
    "scipy" \
    "huggingface-hub>=0.30.0,<1.0" \
    "pillow" \
    "accelerate" \
    "safetensors" \
    "sentencepiece" \
    "timm" \
    "einops" \
    "datasets==5.0.0" \
    "transformers>=4.57,<5"
  export PYTHONPATH="${PROJECT_ROOT}/vendor/python:${PROJECT_ROOT}:${PYTHONPATH:-}"
fi

run_python - <<'PY'
from __future__ import annotations

import importlib.metadata
import re
import shutil
from pathlib import Path

vendor = Path("vendor/python")
if not vendor.exists():
    raise SystemExit


def normalized_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def version_key(version: str) -> tuple[object, ...]:
    return tuple(int(part) if part.isdigit() else part for part in re.split(r"([0-9]+)", version))


records: dict[str, list[tuple[Path, str]]] = {}
for path in list(vendor.glob("*.dist-info")) + list(vendor.glob("*.egg-info")):
    try:
        dist = importlib.metadata.Distribution.at(path)
        name = dist.metadata["Name"]
        version = dist.version
    except Exception:
        parts = path.name.rsplit(".", 1)[0].split("-")
        if len(parts) < 2:
            continue
        name = parts[0]
        version = parts[1]
    records.setdefault(normalized_name(name), []).append((path, version))

for entries in records.values():
    if len(entries) <= 1:
        continue
    keep_path, _keep_version = max(entries, key=lambda item: version_key(item[1]))
    for path, _version in entries:
        if path == keep_path:
            continue
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
        print(f"removed_stale_vendor_metadata={path.name}")
PY

should_install_fla="$(
run_python - <<'PY'
import os
import sys

value = os.environ.get("INSTALL_FLA", "auto").strip().lower()
if value in {"0", "false", "no", "off"}:
    print("0")
    raise SystemExit
if value in {"1", "true", "yes", "on"}:
    print("1")
    raise SystemExit
try:
    import torch
except Exception:
    print("0")
    raise SystemExit
print("1" if sys.platform.startswith("linux") and torch.cuda.is_available() else "0")
PY
)"

if [[ "${should_install_fla}" == "1" ]]; then
  print_step "Prepare flash-linear-attention source"
  FLA_DIR="${FLA_DIR:-${PROJECT_ROOT}/vendor/flash-linear-attention}"
  if [[ ! -d "${FLA_DIR}/.git" ]]; then
    rm -rf "${FLA_DIR}"
    git clone --depth 1 https://github.com/fla-org/flash-linear-attention.git "${FLA_DIR}"
  elif [[ "${UPDATE_FLA:-0}" =~ ^(1|true|yes|on)$ ]]; then
    git -C "${FLA_DIR}" pull --ff-only
  fi
  export PYTHONPATH="${FLA_DIR}:${PROJECT_ROOT}/vendor/python:${PROJECT_ROOT}:${PYTHONPATH:-}"
fi

print_step "Environment versions"
run_python - <<'PY'
import importlib
import importlib.util

import torch

print("python_imports_ok=True")
print("torch", torch.__version__)
print("cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
for name in ["numpy", "scipy", "transformers", "datasets", "huggingface_hub", "accelerate", "timm", "einops"]:
    module = importlib.import_module(name)
    print(name, getattr(module, "__version__", "unknown"))
fla_spec = importlib.util.find_spec("fla")
print("fla_available", fla_spec is not None)
PY
