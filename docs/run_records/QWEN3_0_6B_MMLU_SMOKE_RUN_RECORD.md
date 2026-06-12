# Qwen3-0.6B MMLU Smoke Test 记录

日期：2026-06-09

## 代码改动

- `tools/run_qwen3_attention_search.py` 新增 MMLU prompt source：
  - `--prompt-source mmlu`
  - `--mmlu-dataset`
  - `--mmlu-subject`
  - `--mmlu-split`
  - `--dataset-cache-dir`
- MMLU 数据默认缓存到项目目录：`hf_cache/datasets`。
- 修复 FLA 候选的若干 Qwen3-0.6B 适配参数：
  - `fla_gated_linear_attn` / `fla_multiscale_retention_attn` 不再传不兼容的 `elu` feature map。
  - `fla_based_linear_attn` / `fla_rebased_linear_attn` 使用 FLA 类内部匹配的 head 维度，并使用 `parallel` mode。
  - `fla_native_sparse_attn` 使用 `num_kv_heads=1` 和 `window_size=0`，避免不满足 NSA head-group 约束以及缺少 `flash_attn` 时调用 sliding-window 分支。

## 本地验证

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m py_compile tools/run_qwen3_attention_search.py
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests
PYTHONDONTWRITEBYTECODE=1 python3 tools/run_qwen3_attention_search.py --help
```

结果：

```text
Ran 7 tests in 0.701s
OK
```

## 远端环境

工作目录：

```text
/root/autodl-tmp/PUZZLE
```

环境检查：

```text
torch 2.3.0+cu121
cuda True NVIDIA A800 80GB PCIe
transformers 4.52.4
datasets 5.0.0
huggingface_hub 0.36.2
```

`datasets` 安装到了项目目录：

```bash
/root/miniconda3/bin/python -m pip install --target vendor/python --upgrade --no-cache-dir datasets
/root/miniconda3/bin/python -m pip install --target vendor/python --upgrade --no-cache-dir "huggingface-hub>=0.30.0,<1.0"
```

第二条命令用于修正 `datasets` 默认拉取的 `huggingface-hub 1.x` 与 `transformers 4.52.4` 不兼容的问题。

## MMLU 数据加载验证

```bash
HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 \
PYTHONDONTWRITEBYTECODE=1 /root/miniconda3/bin/python - <<'PY'
from pathlib import Path
from tools.run_qwen3_attention_search import load_mmlu_prompts
prompts = load_mmlu_prompts("cais/mmlu", "abstract_algebra", "test", 1, Path("hf_cache/datasets"))
print(len(prompts))
print(prompts[0].splitlines()[2])
PY
```

输出：

```text
1
Question: Find the degree for the given field extension Q(sqrt(2), sqrt(3), sqrt(18)) over Q.
```

## Qwen Attention MMLU Smoke

命令：

```bash
HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 \
PYTHONDONTWRITEBYTECODE=1 /root/miniconda3/bin/python tools/run_qwen3_attention_search.py \
  --model-id Qwen/Qwen3-0.6B \
  --device gpu \
  --dtype bf16 \
  --prompt-source mmlu \
  --mmlu-dataset cais/mmlu \
  --mmlu-subject abstract_algebra \
  --mmlu-split test \
  --max-prompts 2 \
  --max-layers 1 \
  --seq-len 256 \
  --variants parent,parent_attn,mha_attn,mqa_attn,gqa_kv2,mfa_kv2,mla_kv2,mka_attn,linear_attn,noop_attn \
  --output outputs/qwen3_0_6b_mmlu_smoke.json \
  --pth-output checkpoints/qwen3_0_6b_mmlu_smoke.pth
```

输出摘要：

```text
selected_batch_size: 1
total_kl_score: 1.1080771684646606
selected: ["L0:linear_attn"]
wrote: /root/autodl-tmp/PUZZLE/outputs/qwen3_0_6b_mmlu_smoke.json
wrote_pth: /root/autodl-tmp/PUZZLE/checkpoints/qwen3_0_6b_mmlu_smoke.pth
```

该次运行实际打分 10 个候选：

```text
parent
parent_attn
mha_attn
mqa_attn
gqa_kv2
mfa_kv2
mla_kv2
mka_attn
linear_attn
noop_attn
```

## All Attention MMLU Smoke

命令：

```bash
HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 \
PYTHONDONTWRITEBYTECODE=1 /root/miniconda3/bin/python tools/run_qwen3_attention_search.py \
  --model-id Qwen/Qwen3-0.6B \
  --device gpu \
  --dtype bf16 \
  --prompt-source mmlu \
  --mmlu-dataset cais/mmlu \
  --mmlu-subject abstract_algebra \
  --mmlu-split test \
  --max-prompts 1 \
  --max-layers 1 \
  --seq-len 128 \
  --variants parent,skip_attn,skip_mlp,skip_both,all_attention \
  --output outputs/qwen3_0_6b_mmlu_all_attention_smoke_final.json \
  --pth-output checkpoints/qwen3_0_6b_mmlu_all_attention_smoke_final.pth
```

输出摘要：

```text
selected_batch_size: 1
total_kl_score: 1.3698714971542358
selected: ["L0:linear_attn"]
wrote: /root/autodl-tmp/PUZZLE/outputs/qwen3_0_6b_mmlu_all_attention_smoke_final.json
wrote_pth: /root/autodl-tmp/PUZZLE/checkpoints/qwen3_0_6b_mmlu_all_attention_smoke_final.pth
```

实际打分 22 个候选：

```text
parent
skip_attn
skip_mlp
skip_both
parent_attn
mha_attn
mqa_attn
gqa_kv2
mfa_kv2
mla_kv2
mka_attn
linear_attn
noop_attn
fla_linear_attn
fla_gated_linear_attn
fla_based_linear_attn
fla_rebased_linear_attn
fla_deltanet_attn
fla_gated_deltanet_attn
fla_kimi_delta_attn
fla_multiscale_retention_attn
fla_native_sparse_attn
```

仍被跳过的候选：

```text
fla_mla_attn: FLA MultiheadLatentAttention 需要 flash_attn，当前远端环境未安装。
fla_moba_attn: 当前 vendor/flash-linear-attention-v0.4.2 的 fla.layers 未导出 MoBA。
```

## 产物

```text
outputs/qwen3_0_6b_mmlu_smoke.json
checkpoints/qwen3_0_6b_mmlu_smoke.pth
outputs/qwen3_0_6b_mmlu_all_attention_smoke_final.json
checkpoints/qwen3_0_6b_mmlu_all_attention_smoke_final.pth
outputs/qwen3_0_6b_mmlu_nsa_strict_smoke.json
checkpoints/qwen3_0_6b_mmlu_nsa_strict_smoke.pth
```

注意：这些 smoke test 使用 MMLU prompt 作为 NAS replace-layer KL 搜索输入，不是完整 MMLU accuracy 评测。
