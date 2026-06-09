#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

if [[ -z "${PYTHON:-}" ]]; then
  if [[ -x /root/miniconda3/bin/python ]]; then
    PYTHON=/root/miniconda3/bin/python
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON=python3
  else
    PYTHON=python
  fi
fi

export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"
export HF_HOME="${HF_HOME:-${PROJECT_ROOT}/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export HF_XET_CACHE="${HF_XET_CACHE:-${HF_HOME}/xet}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${PROJECT_ROOT}/.cache}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
DISTILL_NAS_USE_VENDOR_PYTHON="${DISTILL_NAS_USE_VENDOR_PYTHON:-${PUZZLE_USE_VENDOR_PYTHON:-auto}}"
DISTILL_NAS_VENDOR_NEEDED=0
if [[ "${DISTILL_NAS_USE_VENDOR_PYTHON}" == "auto" ]] && [[ -d "${PROJECT_ROOT}/vendor/python" ]]; then
  if [[ "$(uname -s)" == "Linux" ]]; then
    DISTILL_NAS_VENDOR_NEEDED=1
  elif "${PYTHON}" - "${PROJECT_ROOT}/vendor/python" <<'PY'
import importlib.util
import sys

vendor_path = sys.argv[1]
missing = []
for module_name in ["datasets", "transformers", "accelerate", "PIL", "safetensors", "sentencepiece", "timm", "einops"]:
    if importlib.util.find_spec(module_name) is None:
        missing.append(module_name)
if not missing:
    sys.exit(1)
sys.path.insert(0, vendor_path)
try:
    for module_name in missing:
        __import__(module_name)
except Exception:
    sys.exit(1)
sys.exit(0)
PY
  then
    DISTILL_NAS_VENDOR_NEEDED=1
  fi
fi
if [[ "${DISTILL_NAS_USE_VENDOR_PYTHON}" =~ ^(1|true|yes|on)$ ]] || {
  [[ "${DISTILL_NAS_USE_VENDOR_PYTHON}" == "auto" ]] && {
    [[ "$(uname -s)" == "Linux" ]] || [[ "${DISTILL_NAS_VENDOR_NEEDED}" == "1" ]]
  }
}; then
  export PYTHONPATH="${PROJECT_ROOT}/vendor/python:${PROJECT_ROOT}:${PYTHONPATH:-}"
else
  export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
fi

mkdir -p vendor/python hf_cache/models hf_cache/datasets outputs checkpoints

print_step() {
  printf '\n==> %s\n' "$1"
}

run_python() {
  "${PYTHON}" "$@"
}
