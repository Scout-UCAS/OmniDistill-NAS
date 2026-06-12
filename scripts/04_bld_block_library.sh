#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00_common.sh"

WORKFLOW_OUTPUT_DIR="${WORKFLOW_OUTPUT_DIR:-${STAGE_DIR:-outputs/distill_nas_workflow}}"
BLD_OUTPUT_DIR="${BLD_OUTPUT_DIR:-${WORKFLOW_OUTPUT_DIR}/04_bld_block_library}"
DEVICE="${DEVICE:-auto}"
SEQ_LEN="${SEQ_LEN:-}"
BATCH_SIZE="${BATCH_SIZE:-2}"
NUM_BATCHES="${NUM_BATCHES:-4}"
ATTENTION_VARIANTS="${ATTENTION_VARIANTS:-parent_attn,mha_attn,quant_mha_attn,mqa_attn,gqa_kv2,linear_attn,noop_attn}"
LAYER_VARIANTS="${LAYER_VARIANTS:-parent,skip_attn,skip_mlp,skip_both}"
BLD_OUTPUT_PTH="${BLD_OUTPUT_PTH:-${BLD_OUTPUT_DIR}/block_library.pth}"
BLD_SUMMARY_JSON="${BLD_SUMMARY_JSON:-${BLD_OUTPUT_DIR}/summary.json}"
WORKFLOW_BACKEND="${WORKFLOW_BACKEND:-toy}"

print_step "Step 4: BLD block-library training"
if is_real_workflow_backend; then
  BLD_STEPS="${BLD_STEPS:-0}"
  MODEL_VARIANTS="${MODEL_VARIANTS:-${VARIANTS:-parent,skip_attn,skip_mlp,skip_both,all_core_attn,all_fla}}"
  MAX_LAYERS="${MAX_LAYERS:-2}"
  LAYER_STRIDE="${LAYER_STRIDE:-1}"
  build_model_args
  MODEL_SEARCH_ARGS=()
  if [[ "${NO_SKIP_UNAVAILABLE_FLA:-0}" =~ ^(1|true|yes|on)$ ]]; then
    MODEL_SEARCH_ARGS+=(--no-skip-unavailable-fla)
  fi
  run_python tools/run_staged_model_pipeline.py bld \
    "${MODEL_ARGS[@]}" \
    --max-layers "${MAX_LAYERS}" \
    --layer-stride "${LAYER_STRIDE}" \
    --variants "${MODEL_VARIANTS}" \
    --fla-mode "${FLA_MODE:-chunk}" \
    --fla-feature-map "${FLA_FEATURE_MAP:-elu}" \
    --bld-steps "${BLD_STEPS}" \
    --lr "${BLD_LR:-1e-4}" \
    --output-pth "${BLD_OUTPUT_PTH}" \
    --summary-json "${BLD_SUMMARY_JSON}" \
    "${MODEL_SEARCH_ARGS[@]}"
else
  BLD_STEPS="${BLD_STEPS:-1}"
  TOY_SEQ_LEN="${SEQ_LEN:-16}"
  run_python tools/run_staged_toy_pipeline.py bld \
    --device "${DEVICE}" \
    --seq-len "${TOY_SEQ_LEN}" \
    --batch-size "${BATCH_SIZE}" \
    --num-batches "${NUM_BATCHES}" \
    --bld-steps "${BLD_STEPS}" \
    --attention-variants "${ATTENTION_VARIANTS}" \
    --layer-variants "${LAYER_VARIANTS}" \
    --output-pth "${BLD_OUTPUT_PTH}" \
    --summary-json "${BLD_SUMMARY_JSON}"
fi
