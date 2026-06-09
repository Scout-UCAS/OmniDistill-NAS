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

print_step "Step 5: NAS replace-layer scoring and layer importance"
run_python scripts/run_staged_toy_pipeline.py score \
  --device "${DEVICE}" \
  --bld-pth "${BLD_OUTPUT_PTH}" \
  --score-batches "${SCORE_BATCHES}" \
  --batch-sizes "${BATCH_SIZES}" \
  --output-json "${NAS_IMPORTANCE_JSON}"
