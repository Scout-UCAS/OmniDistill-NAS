#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00_common.sh"

WORKFLOW_OUTPUT_DIR="${WORKFLOW_OUTPUT_DIR:-${STAGE_DIR:-outputs/distill_nas_workflow}}"
GKD_OUTPUT_DIR="${GKD_OUTPUT_DIR:-${WORKFLOW_OUTPUT_DIR}/08_global_knowledge_distillation}"
EVAL_OUTPUT_DIR="${EVAL_OUTPUT_DIR:-${WORKFLOW_OUTPUT_DIR}/09_evaluation}"
ARTIFACT_PTH="${EVAL_ARTIFACT_PTH:-${GKD_OUTPUT_DIR}/gkd_model.pth}"
EVAL_OUTPUT_JSON="${EVAL_OUTPUT_JSON:-${EVAL_OUTPUT_DIR}/metrics.json}"
EVAL_DEVICE="${EVAL_DEVICE:-cpu}"
EVAL_NUM_BATCHES="${EVAL_NUM_BATCHES:-4}"

print_step "Step 9: evaluate distilled artifact"
run_python tools/evaluate_artifact.py \
  --artifact-pth "${ARTIFACT_PTH}" \
  --device "${EVAL_DEVICE}" \
  --num-batches "${EVAL_NUM_BATCHES}" \
  --output-json "${EVAL_OUTPUT_JSON}"
