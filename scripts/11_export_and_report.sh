#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00_common.sh"

WORKFLOW_OUTPUT_DIR="${WORKFLOW_OUTPUT_DIR:-${STAGE_DIR:-outputs/distill_nas_workflow}}"
GKD_OUTPUT_DIR="${GKD_OUTPUT_DIR:-${WORKFLOW_OUTPUT_DIR}/08_global_knowledge_distillation}"
EXPORT_OUTPUT_DIR="${EXPORT_OUTPUT_DIR:-${WORKFLOW_OUTPUT_DIR}/11_export}"
ARTIFACT_PTH="${EXPORT_ARTIFACT_PTH:-${GKD_OUTPUT_DIR}/gkd_model.pth}"
REPORT_MD="${REPORT_MD:-${WORKFLOW_OUTPUT_DIR}/report.md}"

EXPORT_ARGS=()
if [[ "${EXPORT_NO_STATE_DICT:-0}" =~ ^(1|true|yes|on)$ ]]; then
  EXPORT_ARGS+=(--no-state-dict)
fi

print_step "Step 11: export artifact and generate report"
if [[ "${#EXPORT_ARGS[@]}" -gt 0 ]]; then
  run_python tools/export_artifact.py \
    --artifact-pth "${ARTIFACT_PTH}" \
    --export-dir "${EXPORT_OUTPUT_DIR}" \
    "${EXPORT_ARGS[@]}"
else
  run_python tools/export_artifact.py \
    --artifact-pth "${ARTIFACT_PTH}" \
    --export-dir "${EXPORT_OUTPUT_DIR}"
fi
run_python tools/generate_report.py \
  --workflow-dir "${WORKFLOW_OUTPUT_DIR}" \
  --output-md "${REPORT_MD}"
