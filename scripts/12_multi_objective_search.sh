#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00_common.sh"

WORKFLOW_OUTPUT_DIR="${WORKFLOW_OUTPUT_DIR:-${STAGE_DIR:-outputs/distill_nas_workflow}}"
NAS_OUTPUT_DIR="${NAS_OUTPUT_DIR:-${WORKFLOW_OUTPUT_DIR}/05_nas_layer_scoring}"
MIP_OUTPUT_DIR="${MIP_OUTPUT_DIR:-${WORKFLOW_OUTPUT_DIR}/06_mip_topk_architecture_configs}"
NAS_IMPORTANCE_JSON="${NAS_IMPORTANCE_JSON:-${NAS_OUTPUT_DIR}/layer_importance.json}"
MULTI_OBJECTIVE_JSON="${MULTI_OBJECTIVE_JSON:-${MIP_OUTPUT_DIR}/multi_objective_search.json}"
PARETO_CONFIG_DIR="${PARETO_CONFIG_DIR:-${MIP_OUTPUT_DIR}/pareto_configs}"
MULTI_OBJECTIVE_REPORT_MD="${MULTI_OBJECTIVE_REPORT_MD:-${MIP_OUTPUT_DIR}/multi_objective_report.md}"
PARETO_SVG="${PARETO_SVG:-${MIP_OUTPUT_DIR}/pareto_front.svg}"
BATCH_SIZES="${BATCH_SIZES:-1,2,4}"
MEMORY_FRACTION="${MEMORY_FRACTION:-0.82}"
RUNTIME_FRACTION="${RUNTIME_FRACTION:-0.82}"
WEIGHT_GRID="${WEIGHT_GRID:-}"
GRID_RESOLUTION="${GRID_RESOLUTION:-4}"
PARETO_MODE="${PARETO_MODE:-auto}"
MAX_EXHAUSTIVE_COMBINATIONS="${MAX_EXHAUSTIVE_COMBINATIONS:-200000}"

MO_ARGS=()
if [[ -n "${WEIGHT_GRID}" ]]; then
  MO_ARGS+=(--weight-grid "${WEIGHT_GRID}")
fi
if [[ "${NO_NORMALIZE_OBJECTIVES:-0}" =~ ^(1|true|yes|on)$ ]]; then
  MO_ARGS+=(--no-normalize-objectives)
fi

print_step "Step 12: multi-objective weight sweep and Pareto report"
if [[ "${#MO_ARGS[@]}" -gt 0 ]]; then
  run_python tools/run_multi_objective_search.py \
    --scores-json "${NAS_IMPORTANCE_JSON}" \
    --output-json "${MULTI_OBJECTIVE_JSON}" \
    --config-dir "${PARETO_CONFIG_DIR}" \
    --report-md "${MULTI_OBJECTIVE_REPORT_MD}" \
    --plot-svg "${PARETO_SVG}" \
    --batch-sizes "${BATCH_SIZES}" \
    --memory-fraction "${MEMORY_FRACTION}" \
    --runtime-fraction "${RUNTIME_FRACTION}" \
    --grid-resolution "${GRID_RESOLUTION}" \
    --pareto-mode "${PARETO_MODE}" \
    --max-exhaustive-combinations "${MAX_EXHAUSTIVE_COMBINATIONS}" \
    "${MO_ARGS[@]}"
else
  run_python tools/run_multi_objective_search.py \
    --scores-json "${NAS_IMPORTANCE_JSON}" \
    --output-json "${MULTI_OBJECTIVE_JSON}" \
    --config-dir "${PARETO_CONFIG_DIR}" \
    --report-md "${MULTI_OBJECTIVE_REPORT_MD}" \
    --plot-svg "${PARETO_SVG}" \
    --batch-sizes "${BATCH_SIZES}" \
    --memory-fraction "${MEMORY_FRACTION}" \
    --runtime-fraction "${RUNTIME_FRACTION}" \
    --grid-resolution "${GRID_RESOLUTION}" \
    --pareto-mode "${PARETO_MODE}" \
    --max-exhaustive-combinations "${MAX_EXHAUSTIVE_COMBINATIONS}"
fi
