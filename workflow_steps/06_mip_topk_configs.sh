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

print_step "Step 6: MIP search for top-K architecture configs"
run_python scripts/run_staged_toy_pipeline.py mip \
  --scores-json "${NAS_IMPORTANCE_JSON}" \
  --output-json "${MIP_TOPK_JSON}" \
  --config-dir "${MIP_CONFIG_DIR}" \
  --top-k "${TOP_K}" \
  --batch-sizes "${BATCH_SIZES}" \
  --memory-fraction "${MEMORY_FRACTION}" \
  --runtime-fraction "${RUNTIME_FRACTION}" \
  --diversity-alpha "${DIVERSITY_ALPHA}"
