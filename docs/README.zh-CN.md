# OmniDistill-NAS

本仓库是 **OmniDistill-NAS**，根据 [`distillation_nas_paper.pdf`](../distillation_nas_paper.pdf) 中的论文 **Distillation-Based NAS for Inference-Optimized LLMs** 实现了一个轻量、可本地运行的蒸馏式架构搜索框架。

论文原方法面向 Llama/Nemotron 等大模型，并依赖真实硬件上的推理 profiling 数据。本实现保留论文的核心算法流程，同时提供两个后端：小型 causal Transformer 用于快速本地验证，Qwen 风格 LLM/VLM/VLA 后端用于真实模型的分阶段 BLD、NAS 打分、MIP、组装和 GKD/OPD。

## 实现内容

当前实现覆盖论文中的主要阶段：

1. 构建 attention 和 FFN 的候选 block library。
2. 使用训练前初始化方法生成候选子模块。
3. 使用归一化 MSE 进行 Blockwise Local Distillation，也就是 BLD。
4. 使用 replace-1-block KL divergence 或 LM loss 给候选块打分。
5. 估计参数内存、KV-cache 内存和 runtime 成本。
6. 使用 MILP 求解“每层选择一个候选块”的多约束架构搜索问题。
7. 可选执行 Global Knowledge Distillation，也就是 GKD，loss 为 hidden cosine loss 加 logits KL loss，并可额外加入 OPD。

## 快速运行

```bash
python3 tools/run_tiny_nas.py --quick
```

示例会在可用时自动选择 CUDA 或 MPS。运行后会输出：

- 当前使用的 device
- 生成的候选 block 数量
- MIP 选择的 batch size
- 架构总 KL 分数
- 总内存估计
- runtime/throughput proxy
- 每一层最终选择的候选块

示例输出类似：

```text
generated_candidates=32
device=mps
selected_batch_size=1
total_kl_score=0.000358
total_memory_bytes=34176
total_runtime_proxy=0.00024269
throughput_proxy=65928.27
architecture:
  L0:parent_attn+ffn_50 score=0.000169
  L1:parent_attn+ffn_50 score=0.000189
```

## 验证

```bash
python3 -m compileall distill_nas_core scripts tools tests
python3 -m unittest discover -s tests
```

贡献前的本地 CI 检查可以直接运行：

```bash
python3 tools/check_project.py
```

如果不希望生成 `__pycache__`，可以使用：

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests
PYTHONDONTWRITEBYTECODE=1 python3 tools/run_tiny_nas.py --quick
```

## 分步脚本

完整流程拆在 `scripts/` 下，每一步都有独立的 `.sh` 文件；04-08
按照论文阶段组织为 BLD、NAS 打分、MIP、组装和 GKD。更详细的脚本清单和
输出位置见 [scripts.md](scripts.md)：

```bash
bash scripts/01_prepare_environment.sh
bash scripts/02_validate_project.sh
bash scripts/03_smoke_tiny_nas.sh
bash scripts/04_bld_block_library.sh
bash scripts/05_nas_layer_importance.sh
bash scripts/06_mip_topk_configs.sh
bash scripts/07_assemble_model_from_config.sh
bash scripts/08_gkd_distill.sh
```

也可以直接运行：

```bash
bash scripts/run_all.sh
```

## 平台化实验入口

除了 shell 脚本，现在也可以用 JSON/YAML 实验配置运行，并自动跳过已经存在的
阶段产物：

```bash
omnidistill run --config configs/toy_experiment.json
omnidistill run --config configs/toy_experiment.json --from-stage evaluate
omnidistill status --workflow-dir outputs/distill_nas_workflow
```

默认 toy 配置在 04-08 论文阶段后增加：

- `evaluate`：输出 `09_evaluation/metrics.json`。
- `profile`：输出 `10_profiling/profile.json`，包含实测 latency。
- `export`：输出 `11_export/manifest.json` 和可移植 artifact。
- `report`：输出 `report.md`。

默认后端是 `toy`，适合快速检查。真实 Qwen/VLM/VLA 分阶段流程使用：

```bash
WORKFLOW_BACKEND=qwen \
MODEL_ID=Qwen/Qwen3-0.6B \
DEVICE=gpu \
MODEL_VARIANTS=parent,skip_attn,skip_mlp,skip_both,all_core_attn \
MAX_LAYERS=2 \
MAX_PROMPTS=2 \
bash scripts/run_all.sh
```

常用覆盖参数示例：

```bash
TOP_K=5 CONFIG_RANK=1 GKD_STEPS=20 bash scripts/run_all.sh
```

## 主要模块

- `distill_nas_core.blocks`：小型 causal attention、FFN、no-op/linear 子模块和 Transformer block。
- `distill_nas_core.search_space`：NAS 搜索空间定义，以及 MHA、MQA、GQA、MFA、MLA、MKA、linear attention、FFN pruning、linear FFN、no-op 的初始化。
- `distill_nas_core.library`：coupled BLD 和 decoupled BLD 的 block library 构建。
- `distill_nas_core.distill`：BLD/GKD/可选 OPD 的 loss 和训练循环。
- `distill_nas_core.scoring`：replace-1-block KL divergence 和 LM loss 打分。
- `distill_nas_core.resources`：参数内存、KV-cache 内存、runtime profiling/估计。
- `distill_nas_core.mip`：混合整数规划架构搜索，包含 diversity constraint 和小规模 exhaustive fallback。
- `distill_nas_core.toy`：用于 demo 的小型 causal language model。
- `distill_nas_core.experiment`：实验配置、resume/cache 和阶段执行。
- `distill_nas_core.evaluation`、`profiler`、`export`、`reporting`：评测、实测 profiling、导出和报告生成。
- `distill_nas_core.data_adapters`、`quantization`、`distributed`、`vla`：数据集、量化校准、设备计划和 VLA rollout 的扩展点。

## 多目标搜索

默认 MIP 仍然是“最小化候选 score，并把 memory/runtime 作为硬约束”。如果希望把
内存和延迟也加入目标函数，可以使用 weighted 模式：

```bash
OBJECTIVE_MODE=weighted SCORE_WEIGHT=1.0 MEMORY_WEIGHT=0.25 RUNTIME_WEIGHT=0.25 \
bash scripts/06_mip_topk_configs.sh
```

对应 CLI 参数是 `--objective-mode weighted --score-weight ... --memory-weight ...
--runtime-weight ...`。默认会归一化 score、memory、runtime，避免不同单位的数值尺度直接主导结果。

如果想一次性查看多组权重下的取舍空间，可以运行 Pareto sweep 报告：

```bash
python3 tools/run_multi_objective_search.py \
  --scores-json outputs/distill_nas_workflow/05_nas_layer_scoring/layer_importance.json
```

该命令会输出 `multi_objective_search.json`、Pareto 架构配置、`multi_objective_report.md`
和 `pareto_front.svg`。小规模搜索空间会精确枚举 Pareto 前沿，大规模空间会使用权重
sweep 的非支配解作为近似前沿。

## 与论文的对应关系

论文中的 NAS 框架由三大阶段组成：

1. **Crafting puzzle pieces**：对每个 block 的候选 attention/FFN 子模块进行 BLD，形成 block library。
2. **Assembling puzzle architecture**：根据 block quality score、runtime、memory、KV-cache 等约束，用 MIP 组装最终架构。
3. **Uptraining**：用 GKD 进行端到端蒸馏，改善不同 block 组合后的兼容性。

本实现中对应关系如下：

- 阶段 1：`distill_nas_core.library`、`distill_nas_core.distill.local_distill_block`
- 阶段 2：`distill_nas_core.scoring`、`distill_nas_core.resources`、`distill_nas_core.mip`
- 阶段 3：`distill_nas_core.distill.global_knowledge_distillation`

## GKD 中加入 OPD

`global_knowledge_distillation` 默认仍然使用原来的离线 GKD，也就是 teacher/student 在同一批 token 上计算 logits KL 和 hidden cosine loss。需要启用文本 OPD 时，传入正数 `opd_weight`，设置正数 `opd_max_new_tokens`，并保证 batch 中有 `input_ids`。

文本 OPD 的流程是：先让 student 用当前策略从 prompt 继续采样，再让 teacher 和 student 对这些 student 生成的 token 计算 log-prob，额外优化 sampled reverse-KL：

```text
log p_student(token) - log p_teacher(token)
```

用法示例：

```python
from distill_nas_core.distill import global_knowledge_distillation

losses = global_knowledge_distillation(
    teacher,
    student,
    prompt_batches,
    steps=100,
    lr=1e-4,
    opd_weight=0.25,
    opd_max_new_tokens=32,
    opd_temperature=1.0,
    opd_top_k=50,
)
```

该实现参考了 Thinking Machines 的 on-policy distillation 思路：https://thinkingmachines.ai/blog/on-policy-distillation/

VLA 模型还支持 action-space OPD：`action_logits`/`predicted_action_logits` 使用 student 采样动作上的 sampled reverse-KL；连续 `actions`/`predicted_actions` 使用 action MSE，属于 action-space distillation 近似，不执行环境 rollout。因此 GKD 不再只依赖语言 token logits。

## Attention 候选

toy distillation NAS pipeline 中的 layer 级候选包含：

```text
parent
skip_attn
skip_mlp
skip_both
```

attention 候选包含：

```text
parent_attn
mha_attn
quant_mha_attn
mqa_attn
gqa_kv*
mfa_attn / mfa_kv*
mla_attn / mla_kv*
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
fla_mla_attn
fla_native_sparse_attn
fla_moba_attn
```

其中 MHA/MQA/GQA 通过调整 `num_kv_heads` 实现；`quant_mha_attn` 使用 int8 对称量化后的 MHA 投影权重；MFA 使用分组低秩 K/V 投影；MLA 使用共享 latent 重建 K/V；MKA 使用共享 Key 和低秩 Value。FLA 名称在通用 toy pipeline 中对应轻量 PyTorch 等价候选，包括 kernel linear attention、gated linear attention、retention 和局部稀疏 attention，因此也会真实 forward，并作为 `AttentionSpec` 进入 BLD、replace-1-block 评分和 MIP 搜索。`all_linear_attn` 会展开为 `linear_attn` 以及 FLA 的 linear/delta family，`all_core_attn` 会把这些 linear attention 与 MHA/MQA/GQA/MFA/MLA/MKA 放在同一层级。`tools/run_tiny_nas.py` 默认使用 `--attention-variants all_attention` 和 `--layer-variants parent,skip_attn,skip_mlp,skip_both`，也可以改成 `all_qwen_attn`、`all_linear_attn`、`all_core_attn`、`all_fla` 或逗号分隔的候选名。

## 注意事项

本仓库没有内置 Llama/Nemotron 权重、真实训练语料或 H100/4090 的硬件测量数据。因此快速 demo 使用 toy model 和 runtime proxy。真实 Qwen/VLM/VLA 后端已经能加载模型、替换语言解码器层并生成分阶段产物；要迁移到其他模型族，还需要补充：

- 模型 block 的适配层
- 真实 token 数据
- 目标硬件上的 block runtime 测量
- 对应推理引擎的非均匀 block 支持

## Qwen3-0.6B 真实模型示例

推荐优先使用分阶段 workflow：

```bash
WORKFLOW_BACKEND=qwen \
MODEL_ID=Qwen/Qwen3-0.6B \
DEVICE=gpu \
MODEL_VARIANTS=parent,skip_attn,skip_mlp,skip_both,all_core_attn,all_fla \
MAX_LAYERS=2 \
MAX_PROMPTS=2 \
bash scripts/run_all.sh
```

04-08 会分别输出 `block_library.pth`、`layer_importance.json`、top-K config、`assembled_model.pth` 和 `gkd_model.pth`。默认保存的是 delta checkpoint：模型 ID、架构 config 和候选替换层权重。checkpoint 恢复默认严格校验 key；只有明确需要跨候选定义加载旧权重时，才设置 `ALLOW_PARTIAL_CHECKPOINT_LOAD=1`。只有设置 `SAVE_FULL_STATE_DICT=1` 时才保存完整模型 state dict。

`tools/run_qwen3_attention_search.py` 会加载 HuggingFace 上的 `Qwen/Qwen3-0.6B`，在 GPU 上对若干 Transformer 层执行 NAS 候选搜索：

```bash
python3 tools/run_qwen3_attention_search.py \
  --model-id Qwen/Qwen3-0.6B \
  --device gpu \
  --max-layers 8 \
  --max-prompts 2 \
  --seq-len 128
```

该示例会实际计算 replace-1-layer KL 分数，然后用 MIP 在真实 Qwen attention 候选、skip 候选和 FLA 候选中为每个搜索层选择一个候选。`all_qwen_attn` 会展开为：

```text
parent_attn
mha_attn
quant_mha_attn
mqa_attn
gqa_kv2
mfa_kv2
mla_kv2
mka_attn
linear_attn
noop_attn
```

`all_linear_attn` 会展开为：

```text
linear_attn
fla_linear_attn
fla_gated_linear_attn
fla_based_linear_attn
fla_rebased_linear_attn
fla_deltanet_attn
fla_gated_deltanet_attn
fla_kimi_delta_attn
```

`all_fla` 会展开为：

```text
fla_linear_attn
fla_gated_linear_attn
fla_based_linear_attn
fla_rebased_linear_attn
fla_deltanet_attn
fla_gated_deltanet_attn
fla_kimi_delta_attn
fla_multiscale_retention_attn
fla_mla_attn
fla_native_sparse_attn
fla_moba_attn
```

其中 MHA/量化 MHA/MQA/GQA/MFA/MLA/MKA/linear/no-op 候选均已在 Qwen decoder layer 中真实 forward，并尽量复用原 Qwen attention 的 q/k/v/o、q/k norm 和 RoPE。FLA 候选也会替换原始 self-attention，并尽量用原 Qwen attention 的 q/k/v/o 权重初始化。`scripts/01_prepare_environment.sh` 在 Linux GPU 环境下会默认把 `fla-org/flash-linear-attention` clone 到当前工作区的 `vendor/flash-linear-attention`；如需跳过可设置 `INSTALL_FLA=0`，如需校验时强制要求 FLA 可设置 `REQUIRE_FLA=1`。当前环境缺少某个 FLA 类或 CUDA 依赖时默认跳过该候选；如需严格失败，可加 `--no-skip-unavailable-fla`。结果默认写入：

```text
outputs/qwen3_0_6b_layer_skip_search.json
checkpoints/qwen3_0_6b_layer_skip_search.pth
```

模型权重缓存默认位于当前工作区的 `hf_cache/models`。本地安装的 `transformers` 等依赖可放在 `vendor/python`，脚本会优先从该目录导入。

如需显式指定候选集合：

```bash
python3 tools/run_qwen3_attention_search.py \
  --variants parent,skip_attn,skip_mlp,skip_both,all_core_attn,all_fla
```

也可以用 MMLU 小样本作为真实输入数据进行 smoke test。该路径需要可选依赖 `datasets`，数据集缓存默认写入 `hf_cache/datasets`：

```bash
python3 tools/run_qwen3_attention_search.py \
  --model-id Qwen/Qwen3-0.6B \
  --device gpu \
  --prompt-source mmlu \
  --mmlu-dataset cais/mmlu \
  --mmlu-subject abstract_algebra \
  --mmlu-split test \
  --max-prompts 2 \
  --max-layers 1 \
  --seq-len 256 \
  --variants parent,parent_attn,mha_attn,quant_mha_attn,mqa_attn,gqa_kv2,mfa_kv2,mla_kv2,mka_attn,all_linear_attn,noop_attn
```

## VLM 支持

同一个真实模型脚本也支持 Qwen 风格 VLM，例如 Qwen3-VL。使用
`--model-kind vlm` 时，脚本会用 `AutoProcessor` 和
`AutoModelForImageTextToText` 加载模型，并自动定位 VLM 外层封装里的
language decoder layers。NAS 候选只替换语言解码器层，vision encoder 和
multimodal projector 保持原样。

内置 smoke prompt 在不传 `--image-path` 时会自动生成一张空白 RGB 图片；
数据集模式默认要求样本里有真实图片字段，只有显式传 `--allow-blank-image`
时才允许缺图 fallback。如果要使用真实图片，可以传单个路径或逗号分隔的多个路径。
Qwen3-VL 需要 transformers 能识别 `qwen3_vl` 架构，`scripts/01_prepare_environment.sh`
会在当前版本过低时把 `transformers>=4.57,<5` 和 `pillow` 安装到
`vendor/python`：

```bash
python3 tools/run_qwen3_attention_search.py \
  --model-id Qwen/Qwen3-VL-2B-Instruct \
  --model-kind vlm \
  --device gpu \
  --max-prompts 1 \
  --max-layers 1 \
  --seq-len 512 \
  --variants parent,parent_attn,mha_attn,quant_mha_attn,mqa_attn,linear_attn,noop_attn \
  --output outputs/qwen3_vl_attention_smoke.json \
  --pth-output checkpoints/qwen3_vl_attention_smoke.pth
```

如果你的环境中已经有 `Qwen3-VL-0.6B` 或其他更小的 VLM checkpoint，只需要把
`--model-id` 换成对应路径或 HuggingFace id 即可。

## VLA 支持和常用数据集

真实模型脚本现在也支持 `--model-kind vla`，面向 OpenVLA 这类
Vision-Language-Action 模型。VLA 路径和 VLM 一样，会用 `AutoProcessor`
加多模态模型加载器，并自动定位语言 decoder layers；NAS 候选只替换语言解码器，
vision encoder、multimodal projector/action head 保持不变。
评分时会优先使用 teacher/student 的动作输出：`action_logits` 用 KL，
连续 `action_mean` 用 MSE，`actions` 或 `predicted_actions` 作为确定性动作值继续用 MSE；如果模型只暴露普通 `logits`，则退回到语言 logits KL。GKD/OPD 中，如果模型同时暴露 `action_mean` 和 `action_log_std`，连续动作会使用 student 采样动作上的 sampled reverse-KL；只有确定性动作张量时默认退回 MSE，设置 `STRICT_ACTION_OPD=1` 可禁止这种近似。VLA 本地数据中的数值 `action` 会作为 `actions` tensor 保存在 batch 里，`state`/`proprio` 会作为 `proprio` tensor 保留；不支持这些字段的普通 HF 模型会在 forward fallback 时自动过滤。

```bash
python3 tools/run_qwen3_attention_search.py \
  --model-id openvla/openvla-7b \
  --model-kind vla \
  --prompt-source built_in \
  --max-prompts 1 \
  --max-layers 1 \
  --seq-len 512 \
  --variants parent,parent_attn,mha_attn,quant_mha_attn,mqa_attn,linear_attn,noop_attn \
  --output outputs/openvla_attention_smoke.json \
  --pth-output checkpoints/openvla_attention_smoke.pth
```

内置常用数据集别名分三类：

- LLM：`mmlu`、`mmlu_pro`、`hellaswag`、`arc_challenge`、`arc_easy`、`gsm8k`、`boolq`、`winogrande`、`truthfulqa`
- VLM：`vqav2`、`okvqa`、`gqa`、`textvqa`、`scienceqa`、`vizwiz`、`coco_caption`
- VLA：`libero`、`lerobot_libero`、`lerobot_pusht`、`aloha_transfer_cube`、`aloha_insertion`，以及 `bridge_v2`、`rt1`、`open_x_embodiment`、`droid` 这类数据集家族别名；后几类公开托管不统一，需要额外传 `--dataset-name` 或 `--dataset-path`

通用 HuggingFace 数据集或本地 JSON/JSONL/CSV/Parquet 也可以使用：

```bash
python3 tools/run_qwen3_attention_search.py \
  --model-id Qwen/Qwen3-VL-2B-Instruct \
  --model-kind vlm \
  --prompt-source dataset \
  --dataset-name lmms-lab/TextVQA \
  --dataset-split validation \
  --dataset-task vlm \
  --max-prompts 2
```

本地 VLA 数据建议包含 `instruction`、`image_path`、`action` 等字段，图片字段
可以是嵌套结构、路径或 byte 编码；数据集样本缺图时默认会报错，然后传入：

```bash
python3 tools/run_qwen3_attention_search.py \
  --model-id openvla/openvla-7b \
  --model-kind vla \
  --prompt-source dataset \
  --dataset-path data/robot_samples.jsonl \
  --dataset-task vla \
  --dataset-image-root data/images \
  --max-prompts 2
```
