#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00_common.sh"

WORKFLOW_OUTPUT_DIR="${WORKFLOW_OUTPUT_DIR:-${STAGE_DIR:-outputs/distill_nas_workflow}}"
GKD_OUTPUT_DIR="${GKD_OUTPUT_DIR:-${WORKFLOW_OUTPUT_DIR}/08_global_knowledge_distillation}"
PROFILE_OUTPUT_DIR="${PROFILE_OUTPUT_DIR:-${WORKFLOW_OUTPUT_DIR}/10_profiling}"
ARTIFACT_PTH="${PROFILE_ARTIFACT_PTH:-${GKD_OUTPUT_DIR}/gkd_model.pth}"
PROFILE_OUTPUT_JSON="${PROFILE_OUTPUT_JSON:-${PROFILE_OUTPUT_DIR}/profile.json}"
PROFILE_DEVICE="${PROFILE_DEVICE:-cpu}"
PROFILE_BATCH_SIZES="${PROFILE_BATCH_SIZES:-1,2,4}"
PROFILE_WARMUP="${PROFILE_WARMUP:-1}"
PROFILE_STEPS="${PROFILE_STEPS:-5}"

print_step "Step 10: profile distilled artifact"
run_python tools/profile_artifact.py \
  --artifact-pth "${ARTIFACT_PTH}" \
  --device "${PROFILE_DEVICE}" \
  --batch-sizes "${PROFILE_BATCH_SIZES}" \
  --warmup "${PROFILE_WARMUP}" \
  --steps "${PROFILE_STEPS}" \
  --output-json "${PROFILE_OUTPUT_JSON}"
