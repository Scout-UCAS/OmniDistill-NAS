#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00_common.sh"

WORKFLOW_OUTPUT_DIR="${WORKFLOW_OUTPUT_DIR:-${STAGE_DIR:-outputs/distill_nas_workflow}}"
BLD_OUTPUT_DIR="${BLD_OUTPUT_DIR:-${WORKFLOW_OUTPUT_DIR}/04_bld_block_library}"
NAS_OUTPUT_DIR="${NAS_OUTPUT_DIR:-${WORKFLOW_OUTPUT_DIR}/05_nas_layer_scoring}"
DEVICE="${DEVICE:-auto}"
SCORE_BATCHES="${SCORE_BATCHES:-2}"
BATCH_SIZES="${BATCH_SIZES:-1,2,4}"
BLD_OUTPUT_PTH="${BLD_OUTPUT_PTH:-${BLD_OUTPUT_DIR}/block_library.pth}"
NAS_IMPORTANCE_JSON="${NAS_IMPORTANCE_JSON:-${NAS_OUTPUT_DIR}/layer_importance.json}"
WORKFLOW_BACKEND="${WORKFLOW_BACKEND:-toy}"

print_step "Step 5: NAS replace-layer scoring and layer importance"
if is_real_workflow_backend; then
  build_model_args
  MODEL_SCORE_ARGS=()
  if [[ "${NO_SKIP_UNAVAILABLE_FLA:-0}" =~ ^(1|true|yes|on)$ ]]; then
    MODEL_SCORE_ARGS+=(--no-skip-unavailable-fla)
  fi
  run_python scripts/run_staged_model_pipeline.py score \
    "${MODEL_ARGS[@]}" \
    --bld-pth "${BLD_OUTPUT_PTH}" \
    --score-batches "${SCORE_BATCHES}" \
    --batch-sizes "${BATCH_SIZES}" \
    --output-json "${NAS_IMPORTANCE_JSON}" \
    "${MODEL_SCORE_ARGS[@]}"
else
  run_python scripts/run_staged_toy_pipeline.py score \
    --device "${DEVICE}" \
    --bld-pth "${BLD_OUTPUT_PTH}" \
    --score-batches "${SCORE_BATCHES}" \
    --batch-sizes "${BATCH_SIZES}" \
    --output-json "${NAS_IMPORTANCE_JSON}"
fi
