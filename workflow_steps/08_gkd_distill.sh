#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00_common.sh"

WORKFLOW_OUTPUT_DIR="${WORKFLOW_OUTPUT_DIR:-${STAGE_DIR:-outputs/distill_nas_workflow}}"
ASSEMBLY_OUTPUT_DIR="${ASSEMBLY_OUTPUT_DIR:-${WORKFLOW_OUTPUT_DIR}/07_model_assembly}"
GKD_OUTPUT_DIR="${GKD_OUTPUT_DIR:-${WORKFLOW_OUTPUT_DIR}/08_global_knowledge_distillation}"
DEVICE="${DEVICE:-auto}"
ASSEMBLED_MODEL_PTH="${ASSEMBLED_MODEL_PTH:-${ASSEMBLY_OUTPUT_DIR}/assembled_model.pth}"
GKD_OUTPUT_PTH="${GKD_OUTPUT_PTH:-${GKD_OUTPUT_DIR}/gkd_model.pth}"
GKD_SUMMARY_JSON="${GKD_SUMMARY_JSON:-${GKD_OUTPUT_DIR}/summary.json}"
GKD_STEPS="${GKD_STEPS:-2}"
GKD_LR="${GKD_LR:-1e-4}"
SEQ_LEN="${SEQ_LEN:-}"
BATCH_SIZE="${BATCH_SIZE:-2}"
NUM_BATCHES="${NUM_BATCHES:-4}"
OPD_WEIGHT="${OPD_WEIGHT:-0.0}"
OPD_MAX_NEW_TOKENS="${OPD_MAX_NEW_TOKENS:-0}"
WORKFLOW_BACKEND="${WORKFLOW_BACKEND:-toy}"

GKD_ARGS=()
if [[ "${INCLUDE_LM_LOSS:-0}" =~ ^(1|true|yes|on)$ ]]; then
  GKD_ARGS+=(--include-lm-loss)
fi
if [[ -n "${OPD_TEMPERATURE:-}" ]]; then
  GKD_ARGS+=(--opd-temperature "${OPD_TEMPERATURE}")
fi
if [[ -n "${OPD_TOP_K:-}" ]]; then
  GKD_ARGS+=(--opd-top-k "${OPD_TOP_K}")
fi

print_step "Step 8: GKD global distillation"
if is_real_workflow_backend; then
  build_model_args
  if [[ "${SAVE_FULL_STATE_DICT:-0}" =~ ^(1|true|yes|on)$ ]]; then
    GKD_ARGS+=(--save-full-state-dict)
  fi
  BASE_CMD=(
    scripts/run_staged_model_pipeline.py gkd
    "${MODEL_ARGS[@]}"
    --assembled-pth "${ASSEMBLED_MODEL_PTH}"
    --gkd-steps "${GKD_STEPS}"
    --lr "${GKD_LR}"
    --opd-weight "${OPD_WEIGHT}"
    --opd-max-new-tokens "${OPD_MAX_NEW_TOKENS}"
    --output-pth "${GKD_OUTPUT_PTH}"
    --summary-json "${GKD_SUMMARY_JSON}"
  )
else
  TOY_SEQ_LEN="${SEQ_LEN:-16}"
  BASE_CMD=(
    scripts/run_staged_toy_pipeline.py gkd
    --device "${DEVICE}"
    --assembled-pth "${ASSEMBLED_MODEL_PTH}"
    --seq-len "${TOY_SEQ_LEN}"
    --batch-size "${BATCH_SIZE}"
    --num-batches "${NUM_BATCHES}"
    --gkd-steps "${GKD_STEPS}"
    --lr "${GKD_LR}"
    --opd-weight "${OPD_WEIGHT}"
    --opd-max-new-tokens "${OPD_MAX_NEW_TOKENS}"
    --output-pth "${GKD_OUTPUT_PTH}"
    --summary-json "${GKD_SUMMARY_JSON}"
  )
fi
if [[ "${#GKD_ARGS[@]}" -gt 0 ]]; then
  run_python "${BASE_CMD[@]}" "${GKD_ARGS[@]}"
else
  run_python "${BASE_CMD[@]}"
fi
