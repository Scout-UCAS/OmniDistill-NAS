#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKFLOW_WORKDIR="${WORKFLOW_WORKDIR:-${PWD}}"
mkdir -p "${WORKFLOW_WORKDIR}"
WORKFLOW_WORKDIR="$(cd "${WORKFLOW_WORKDIR}" && pwd)"
DISTILL_NAS_VENDOR_DIR="${DISTILL_NAS_VENDOR_DIR:-${WORKFLOW_WORKDIR}/vendor}"
DISTILL_NAS_VENDOR_PYTHON="${DISTILL_NAS_VENDOR_PYTHON:-${DISTILL_NAS_VENDOR_DIR}/python}"
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
export WORKFLOW_WORKDIR
export DISTILL_NAS_VENDOR_DIR
export DISTILL_NAS_VENDOR_PYTHON
export WORKFLOW_OUTPUT_DIR="${WORKFLOW_OUTPUT_DIR:-${STAGE_DIR:-${WORKFLOW_WORKDIR}/outputs/distill_nas_workflow}}"
export HF_HOME="${HF_HOME:-${WORKFLOW_WORKDIR}/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export HF_XET_CACHE="${HF_XET_CACHE:-${HF_HOME}/xet}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${WORKFLOW_WORKDIR}/.cache}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
DISTILL_NAS_USE_VENDOR_PYTHON="${DISTILL_NAS_USE_VENDOR_PYTHON:-${PUZZLE_USE_VENDOR_PYTHON:-auto}}"
DISTILL_NAS_VENDOR_NEEDED=0
if [[ "${DISTILL_NAS_USE_VENDOR_PYTHON}" == "auto" ]] && [[ -d "${DISTILL_NAS_VENDOR_PYTHON}" ]]; then
  if [[ "$(uname -s)" == "Linux" ]]; then
    DISTILL_NAS_VENDOR_NEEDED=1
  elif "${PYTHON}" - "${DISTILL_NAS_VENDOR_PYTHON}" <<'PY'
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
  export PYTHONPATH="${DISTILL_NAS_VENDOR_PYTHON}:${PROJECT_ROOT}:${PYTHONPATH:-}"
else
  export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
fi

mkdir -p "${DISTILL_NAS_VENDOR_PYTHON}" "${HF_HOME}/models" "${HF_HOME}/datasets" "${WORKFLOW_WORKDIR}/outputs" "${WORKFLOW_WORKDIR}/checkpoints"

print_step() {
  printf '\n==> %s\n' "$1"
}

run_python() {
  "${PYTHON}" "$@"
}

is_real_workflow_backend() {
  case "${WORKFLOW_BACKEND:-toy}" in
    qwen|model|real|llm|vlm|vla) return 0 ;;
    *) return 1 ;;
  esac
}

build_model_args() {
  MODEL_ARGS=(
    --model-id "${MODEL_ID:-Qwen/Qwen3-0.6B}"
    --model-kind "${MODEL_KIND:-auto}"
    --device "${DEVICE:-auto}"
    --dtype "${DTYPE:-auto}"
    --cache-dir "${MODEL_CACHE_DIR:-${HF_HOME}/models}"
    --prompt-source "${PROMPT_SOURCE:-built_in}"
    --seq-len "${SEQ_LEN:-128}"
    --max-prompts "${MAX_PROMPTS:-2}"
    --mmlu-dataset "${MMLU_DATASET:-cais/mmlu}"
    --mmlu-subject "${MMLU_SUBJECT:-abstract_algebra}"
    --mmlu-split "${MMLU_SPLIT:-test}"
    --dataset-cache-dir "${DATASET_CACHE_DIR:-${HF_HOME}/datasets}"
    --vlm-blank-image-size "${VLM_BLANK_IMAGE_SIZE:-224}"
  )
  if [[ -n "${DATASET_NAME:-}" ]]; then MODEL_ARGS+=(--dataset-name "${DATASET_NAME}"); fi
  if [[ -n "${DATASET_CONFIG:-}" ]]; then MODEL_ARGS+=(--dataset-config "${DATASET_CONFIG}"); fi
  if [[ -n "${DATASET_SPLIT:-}" ]]; then MODEL_ARGS+=(--dataset-split "${DATASET_SPLIT}"); fi
  if [[ -n "${DATASET_PATH:-}" ]]; then MODEL_ARGS+=(--dataset-path "${DATASET_PATH}"); fi
  if [[ -n "${DATASET_TASK:-}" ]]; then MODEL_ARGS+=(--dataset-task "${DATASET_TASK}"); fi
  if [[ -n "${DATASET_IMAGE_ROOT:-}" ]]; then MODEL_ARGS+=(--dataset-image-root "${DATASET_IMAGE_ROOT}"); fi
  if [[ -n "${IMAGE_PATH:-}" ]]; then MODEL_ARGS+=(--image-path "${IMAGE_PATH}"); fi
  if [[ "${INCLUDE_DATASET_TARGET:-0}" =~ ^(1|true|yes|on)$ ]]; then MODEL_ARGS+=(--include-dataset-target); fi
  if [[ "${ALLOW_BLANK_IMAGE:-0}" =~ ^(1|true|yes|on)$ ]]; then MODEL_ARGS+=(--allow-blank-image); fi
  if [[ "${NO_VLM_GENERATION_PROMPT:-0}" =~ ^(1|true|yes|on)$ ]]; then MODEL_ARGS+=(--no-vlm-generation-prompt); fi
}
