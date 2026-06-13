#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00_common.sh"

TOY_BLD_STEPS="${TOY_BLD_STEPS:-1}"
TOY_SCORE_BATCHES="${TOY_SCORE_BATCHES:-1}"
TOY_ATTENTION_VARIANTS="${TOY_ATTENTION_VARIANTS:-all_attention}"
TOY_LAYER_VARIANTS="${TOY_LAYER_VARIANTS:-parent,skip_attn,skip_mlp,skip_both}"
TOY_LOG="${TOY_LOG:-${WORKFLOW_WORKDIR}/outputs/tiny_nas_quick.log}"

print_step "Step 3: smoke test the tiny distillation NAS pipeline"
run_python tools/run_tiny_nas.py \
  --quick \
  --bld-steps "${TOY_BLD_STEPS}" \
  --score-batches "${TOY_SCORE_BATCHES}" \
  --attention-variants "${TOY_ATTENTION_VARIANTS}" \
  --layer-variants "${TOY_LAYER_VARIANTS}" | tee "${TOY_LOG}"

echo "toy_log=${TOY_LOG}"
