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
if importlib.util.find_spec("torch") is None:
    missing.append("torch")
if importlib.util.find_spec("datasets") is None:
    missing.append("datasets")
if importlib.util.find_spec("PIL") is None:
    missing.append("pillow")
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

print_step "Environment versions"
run_python - <<'PY'
import importlib

import torch

print("python_imports_ok=True")
print("torch", torch.__version__)
print("cuda", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
for name in ["transformers", "datasets", "huggingface_hub", "accelerate", "timm", "einops"]:
    module = importlib.import_module(name)
    print(name, getattr(module, "__version__", "unknown"))
PY
