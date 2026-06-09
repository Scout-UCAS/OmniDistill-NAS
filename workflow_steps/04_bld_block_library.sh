#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00_common.sh"

WORKFLOW_OUTPUT_DIR="${WORKFLOW_OUTPUT_DIR:-${STAGE_DIR:-outputs/distill_nas_workflow}}"
BLD_OUTPUT_DIR="${BLD_OUTPUT_DIR:-${WORKFLOW_OUTPUT_DIR}/04_bld_block_library}"
DEVICE="${DEVICE:-auto}"
BLD_STEPS="${BLD_STEPS:-1}"
SEQ_LEN="${SEQ_LEN:-16}"
BATCH_SIZE="${BATCH_SIZE:-2}"
NUM_BATCHES="${NUM_BATCHES:-4}"
ATTENTION_VARIANTS="${ATTENTION_VARIANTS:-parent_attn,mha_attn,quant_mha_attn,mqa_attn,gqa_kv2,linear_attn,noop_attn}"
LAYER_VARIANTS="${LAYER_VARIANTS:-parent,skip_attn,skip_mlp,skip_both}"
BLD_OUTPUT_PTH="${BLD_OUTPUT_PTH:-${BLD_OUTPUT_DIR}/block_library.pth}"
BLD_SUMMARY_JSON="${BLD_SUMMARY_JSON:-${BLD_OUTPUT_DIR}/summary.json}"

print_step "Step 4: BLD block-library training"
run_python scripts/run_staged_toy_pipeline.py bld \
  --device "${DEVICE}" \
  --seq-len "${SEQ_LEN}" \
  --batch-size "${BATCH_SIZE}" \
  --num-batches "${NUM_BATCHES}" \
  --bld-steps "${BLD_STEPS}" \
  --attention-variants "${ATTENTION_VARIANTS}" \
  --layer-variants "${LAYER_VARIANTS}" \
  --output-pth "${BLD_OUTPUT_PTH}" \
  --summary-json "${BLD_SUMMARY_JSON}"
