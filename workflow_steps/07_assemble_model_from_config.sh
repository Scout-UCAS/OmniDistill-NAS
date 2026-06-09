#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00_common.sh"

WORKFLOW_OUTPUT_DIR="${WORKFLOW_OUTPUT_DIR:-${STAGE_DIR:-outputs/distill_nas_workflow}}"
BLD_OUTPUT_DIR="${BLD_OUTPUT_DIR:-${WORKFLOW_OUTPUT_DIR}/04_bld_block_library}"
MIP_OUTPUT_DIR="${MIP_OUTPUT_DIR:-${WORKFLOW_OUTPUT_DIR}/06_mip_topk_architecture_configs}"
ASSEMBLY_OUTPUT_DIR="${ASSEMBLY_OUTPUT_DIR:-${WORKFLOW_OUTPUT_DIR}/07_model_assembly}"
BLD_OUTPUT_PTH="${BLD_OUTPUT_PTH:-${BLD_OUTPUT_DIR}/block_library.pth}"
MIP_TOPK_JSON="${MIP_TOPK_JSON:-${MIP_OUTPUT_DIR}/topk_architecture_configs.json}"
CONFIG_RANK="${CONFIG_RANK:-0}"
ASSEMBLED_MODEL_PTH="${ASSEMBLED_MODEL_PTH:-${ASSEMBLY_OUTPUT_DIR}/assembled_model.pth}"
ASSEMBLED_SUMMARY_JSON="${ASSEMBLED_SUMMARY_JSON:-${ASSEMBLY_OUTPUT_DIR}/summary.json}"
WORKFLOW_BACKEND="${WORKFLOW_BACKEND:-toy}"

CONFIG_ARGS=(--configs-json "${MIP_TOPK_JSON}" --config-rank "${CONFIG_RANK}")
if [[ -n "${CONFIG_PATH:-}" ]]; then
  CONFIG_ARGS=(--config-path "${CONFIG_PATH}")
fi

print_step "Step 7: assemble model from selected architecture config"
if is_real_workflow_backend; then
  build_model_args
  ASSEMBLE_ARGS=()
  if [[ "${SKIP_FORWARD_CHECK:-0}" =~ ^(1|true|yes|on)$ ]]; then
    ASSEMBLE_ARGS+=(--skip-forward-check)
  fi
  if [[ "${SAVE_FULL_STATE_DICT:-0}" =~ ^(1|true|yes|on)$ ]]; then
    ASSEMBLE_ARGS+=(--save-full-state-dict)
  fi
  run_python scripts/run_staged_model_pipeline.py assemble \
    "${MODEL_ARGS[@]}" \
    --bld-pth "${BLD_OUTPUT_PTH}" \
    "${CONFIG_ARGS[@]}" \
    --output-pth "${ASSEMBLED_MODEL_PTH}" \
    --summary-json "${ASSEMBLED_SUMMARY_JSON}" \
    "${ASSEMBLE_ARGS[@]}"
else
  run_python scripts/run_staged_toy_pipeline.py assemble \
    --bld-pth "${BLD_OUTPUT_PTH}" \
    "${CONFIG_ARGS[@]}" \
    --output-pth "${ASSEMBLED_MODEL_PTH}" \
    --summary-json "${ASSEMBLED_SUMMARY_JSON}"
fi
