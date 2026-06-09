# Qwen3-0.6B Linear Attention 同级候选运行记录

记录日期：2026-06-09  
远端工作目录：`/root/autodl-tmp/PUZZLE`  
远端服务器：`ssh -p 33052 root@connect.nma1.seetacloud.com`

> 说明：本文不记录 SSH 密码。模型权重、`transformers`/`datasets` 依赖、FLA 源码、JSON 输出和 `.pth` 输出均放在 `/root/autodl-tmp/PUZZLE` 下。

## 1. 本次目标

将 linear attention 家族提升为和 MHA/MQA/GQA/MFA/MLA/MKA 同一层级的 NAS attention 候选，而不是只作为单独的 FLA 附属集合。

新增/整理后的别名：

```text
all_linear_attn =
  linear_attn
  fla_linear_attn
  fla_gated_linear_attn
  fla_based_linear_attn
  fla_rebased_linear_attn
  fla_deltanet_attn
  fla_gated_deltanet_attn
  fla_kimi_delta_attn

all_core_attn =
  parent_attn
  mha_attn
  quant_mha_attn
  mqa_attn
  gqa_kv2
  mfa_kv2
  mla_kv2
  mka_attn
  all_linear_attn
  noop_attn

all_attention =
  parent / skip variants
  all_core_attn
  fla_multiscale_retention_attn
  fla_mla_attn
  fla_native_sparse_attn
  fla_moba_attn
```

## 2. 本地验证命令

### 2.1 语法检查

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m py_compile \
  distill_nas_core/blocks.py \
  distill_nas_core/search_space.py \
  distill_nas_core/resources.py \
  scripts/run_qwen3_attention_search.py \
  scripts/run_tiny_nas.py \
  test_suite/test_blocks.py
```

输出：无输出，表示通过。

### 2.2 单元测试

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s test_suite
```

输出：

```text
Ran 9 tests in 1.003s
OK
```

### 2.3 Tiny NAS 全量候选

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/run_tiny_nas.py \
  --quick \
  --bld-steps 0 \
  --score-batches 1 \
  --attention-variants all_attention \
  --layer-variants parent,skip_attn,skip_mlp,skip_both
```

关键输出：

```text
generated_candidates=176
architecture:
  L0:quant_mha_attn+ffn_50
  L1:quant_mha_attn+parent_ffn
```

## 3. 远端验证命令

### 3.1 远端单元测试

```bash
cd /root/autodl-tmp/PUZZLE
PYTHONDONTWRITEBYTECODE=1 /root/miniconda3/bin/python -m unittest discover -s test_suite
```

输出：

```text
Ran 9 tests in 1.406s
OK
```

### 3.2 远端 Tiny NAS 全量候选

```bash
cd /root/autodl-tmp/PUZZLE
PYTHONDONTWRITEBYTECODE=1 /root/miniconda3/bin/python scripts/run_tiny_nas.py \
  --quick \
  --bld-steps 0 \
  --score-batches 1 \
  --attention-variants all_attention \
  --layer-variants parent,skip_attn,skip_mlp,skip_both
```

关键输出：

```text
generated_candidates=176
device=cuda
architecture:
  L0:quant_mha_attn+ffn_50
  L1:quant_mha_attn+parent_ffn
```

### 3.3 Qwen3 + MMLU：Qwen-native 与 linear 同级候选

```bash
cd /root/autodl-tmp/PUZZLE
WORKFLOW_BACKEND=qwen \
MODEL_ID=Qwen/Qwen3-0.6B \
DEVICE=gpu \
PROMPT_SOURCE=mmlu \
MODEL_VARIANTS=parent,parent_attn,mha_attn,quant_mha_attn,mqa_attn,gqa_kv2,mfa_kv2,mla_kv2,mka_attn,all_linear_attn,noop_attn \
MAX_LAYERS=1 \
MAX_PROMPTS=2 \
bash workflow_steps/04_bld_block_library.sh
WORKFLOW_BACKEND=qwen bash workflow_steps/05_nas_layer_importance.sh
WORKFLOW_BACKEND=qwen bash workflow_steps/06_mip_topk_configs.sh
```

默认候选：

```text
parent,parent_attn,mha_attn,quant_mha_attn,mqa_attn,gqa_kv2,mfa_kv2,mla_kv2,mka_attn,all_linear_attn,noop_attn
```

输出：

```text
outputs/qwen3_0_6b_mmlu_smoke.json
outputs/qwen3_0_6b_mmlu_smoke_attention_scores/
checkpoints/qwen3_0_6b_mmlu_smoke.pth
```

确认结果：

```text
variant_count=18
has_quant_mha=True
has_linear_family=True
selected=L0:linear_attn
```

### 3.4 Qwen3 + MMLU：全量 attention 候选

```bash
cd /root/autodl-tmp/PUZZLE
WORKFLOW_BACKEND=qwen \
MODEL_ID=Qwen/Qwen3-0.6B \
DEVICE=gpu \
PROMPT_SOURCE=mmlu \
MODEL_VARIANTS=parent,skip_attn,skip_mlp,skip_both,all_attention \
MAX_LAYERS=1 \
MAX_PROMPTS=2 \
bash workflow_steps/04_bld_block_library.sh
WORKFLOW_BACKEND=qwen bash workflow_steps/05_nas_layer_importance.sh
WORKFLOW_BACKEND=qwen bash workflow_steps/06_mip_topk_configs.sh
```

后台运行时使用的日志：

```text
outputs/logs/step06_all_attention_20260609_161605.log
```

关键输出：

```text
selected:
  L0:linear_attn

wrote=outputs/qwen3_0_6b_mmlu_all_attention_smoke_final.json
wrote_attention_scores=outputs/qwen3_0_6b_mmlu_all_attention_smoke_final_attention_scores
wrote_pth=checkpoints/qwen3_0_6b_mmlu_all_attention_smoke_final.pth
```

确认结果：

```text
variant_count=25
score_json_count=26
has_quant_mha=True
has_linear_family=True
```

当前 FLA 源码版本中以下候选被记录为 skipped：

```text
fla_mla_attn
fla_moba_attn
```

其中 `fla_mla_attn` 是构造参数与当前 `MultiheadLatentAttention` 版本不匹配；`fla_moba_attn` 是当前 `vendor/flash-linear-attention-v0.4.2` 中没有 `fla.layers.MoBA`。

### 3.5 重新生成 attention 级输出汇总

```bash
cd /root/autodl-tmp/PUZZLE
WORKFLOW_BACKEND=qwen bash workflow_steps/07_assemble_model_from_config.sh
```

关键输出：

```text
outputs/qwen3_0_6b_mmlu_smoke.json
  num_scores 18
  selected ['L0:linear_attn']
  attention_score_dir outputs/qwen3_0_6b_mmlu_smoke_attention_scores

outputs/qwen3_0_6b_mmlu_all_attention_smoke_final.json
  num_scores 23
  selected ['L0:linear_attn']
  skipped_variants ['fla_mla_attn', 'fla_moba_attn']
  attention_score_dir outputs/qwen3_0_6b_mmlu_all_attention_smoke_final_attention_scores
```

## 4. 主要产出路径

```text
outputs/qwen3_0_6b_mmlu_smoke.json
outputs/qwen3_0_6b_mmlu_smoke_attention_scores/
outputs/qwen3_0_6b_mmlu_smoke_attention_scores/_summary.csv
outputs/qwen3_0_6b_mmlu_smoke_attention_scores/_summary.json
checkpoints/qwen3_0_6b_mmlu_smoke.pth

outputs/qwen3_0_6b_mmlu_all_attention_smoke_final.json
outputs/qwen3_0_6b_mmlu_all_attention_smoke_final_attention_scores/
outputs/qwen3_0_6b_mmlu_all_attention_smoke_final_attention_scores/_summary.csv
outputs/qwen3_0_6b_mmlu_all_attention_smoke_final_attention_scores/_summary.json
checkpoints/qwen3_0_6b_mmlu_all_attention_smoke_final.pth
```

每个候选均有单独 JSON，例如：

```text
outputs/qwen3_0_6b_mmlu_all_attention_smoke_final_attention_scores/quant_mha_attn.json
outputs/qwen3_0_6b_mmlu_all_attention_smoke_final_attention_scores/linear_attn.json
outputs/qwen3_0_6b_mmlu_all_attention_smoke_final_attention_scores/fla_linear_attn.json
outputs/qwen3_0_6b_mmlu_all_attention_smoke_final_attention_scores/fla_mla_attn.json
outputs/qwen3_0_6b_mmlu_all_attention_smoke_final_attention_scores/fla_moba_attn.json
```

`fla_mla_attn.json` 和 `fla_moba_attn.json` 的 `status` 为 `skipped`，具体原因记录在 `skipped_reason` 字段以及 `_summary.csv` 中。
