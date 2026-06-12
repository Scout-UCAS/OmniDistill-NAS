#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00_common.sh"

WORKFLOW_OUTPUT_DIR="${WORKFLOW_OUTPUT_DIR:-${STAGE_DIR:-outputs/distill_nas_workflow}}"
NAS_OUTPUT_DIR="${NAS_OUTPUT_DIR:-${WORKFLOW_OUTPUT_DIR}/05_nas_layer_scoring}"
MIP_OUTPUT_DIR="${MIP_OUTPUT_DIR:-${WORKFLOW_OUTPUT_DIR}/06_mip_topk_architecture_configs}"
NAS_IMPORTANCE_JSON="${NAS_IMPORTANCE_JSON:-${NAS_OUTPUT_DIR}/layer_importance.json}"
MIP_TOPK_JSON="${MIP_TOPK_JSON:-${MIP_OUTPUT_DIR}/topk_architecture_configs.json}"
MIP_CONFIG_DIR="${MIP_CONFIG_DIR:-${MIP_OUTPUT_DIR}/configs}"
TOP_K="${TOP_K:-3}"
BATCH_SIZES="${BATCH_SIZES:-1,2,4}"
MEMORY_FRACTION="${MEMORY_FRACTION:-0.82}"
RUNTIME_FRACTION="${RUNTIME_FRACTION:-0.82}"
DIVERSITY_ALPHA="${DIVERSITY_ALPHA:-0.75}"
OBJECTIVE_MODE="${OBJECTIVE_MODE:-score}"
SCORE_WEIGHT="${SCORE_WEIGHT:-1.0}"
MEMORY_WEIGHT="${MEMORY_WEIGHT:-0.0}"
RUNTIME_WEIGHT="${RUNTIME_WEIGHT:-0.0}"
WORKFLOW_BACKEND="${WORKFLOW_BACKEND:-toy}"

OBJECTIVE_ARGS=(
  --objective-mode "${OBJECTIVE_MODE}"
  --score-weight "${SCORE_WEIGHT}"
  --memory-weight "${MEMORY_WEIGHT}"
  --runtime-weight "${RUNTIME_WEIGHT}"
)
if [[ "${NO_NORMALIZE_OBJECTIVES:-0}" =~ ^(1|true|yes|on)$ ]]; then
  OBJECTIVE_ARGS+=(--no-normalize-objectives)
fi

print_step "Step 6: MIP search for top-K architecture configs"
if is_real_workflow_backend; then
  run_python tools/run_staged_model_pipeline.py mip \
    --scores-json "${NAS_IMPORTANCE_JSON}" \
    --output-json "${MIP_TOPK_JSON}" \
    --config-dir "${MIP_CONFIG_DIR}" \
    --top-k "${TOP_K}" \
    --batch-sizes "${BATCH_SIZES}" \
    --memory-fraction "${MEMORY_FRACTION}" \
    --runtime-fraction "${RUNTIME_FRACTION}" \
    --diversity-alpha "${DIVERSITY_ALPHA}" \
    "${OBJECTIVE_ARGS[@]}"
else
  run_python tools/run_staged_toy_pipeline.py mip \
    --scores-json "${NAS_IMPORTANCE_JSON}" \
    --output-json "${MIP_TOPK_JSON}" \
    --config-dir "${MIP_CONFIG_DIR}" \
    --top-k "${TOP_K}" \
    --batch-sizes "${BATCH_SIZES}" \
    --memory-fraction "${MEMORY_FRACTION}" \
    --runtime-fraction "${RUNTIME_FRACTION}" \
    --diversity-alpha "${DIVERSITY_ALPHA}" \
    "${OBJECTIVE_ARGS[@]}"
fi
