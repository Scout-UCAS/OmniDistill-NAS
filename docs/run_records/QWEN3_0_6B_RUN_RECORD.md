# Qwen3-0.6B Puzzle 远端运行记录

记录日期：2026-06-09  
远端工作目录：`/root/autodl-tmp/PUZZLE`  
远端服务器：`ssh -p 33052 root@connect.nma1.seetacloud.com`  

> 说明：SSH 密码不写入本文档。所有模型权重、Python 依赖库、JSON 输出和 `.pth` 输出均已放在 `/root/autodl-tmp/PUZZLE` 下。

## 1. 涉及脚本

### 1.1 真实模型运行脚本

路径：

```text
examples/run_qwen3_attention_search.py
```

作用：

- 加载 `Qwen/Qwen3-0.6B`。
- 优先从项目目录下的 `vendor/python` 导入 `transformers` 等库。
- 将 HuggingFace 缓存默认指向项目目录下的 `hf_cache`。
- 在 GPU 上计算 no-op-only grouped replace-1-layer KL 分数。
- 在 `parent`、`skip_attn`、`skip_mlp`、`skip_both` 四种候选中做 MIP 搜索。
- 保存 JSON 和 `.pth` 结果。

关键项目内路径：

```text
/root/autodl-tmp/PUZZLE/vendor/python
/root/autodl-tmp/PUZZLE/hf_cache/models
/root/autodl-tmp/PUZZLE/outputs/qwen3_0_6b_layer_skip_search.json
/root/autodl-tmp/PUZZLE/checkpoints/qwen3_0_6b_layer_skip_search.pth
```

### 1.2 MIP 求解模块

路径：

```text
puzzle_nas/mip.py
```

作用：

- 定义 `SearchCandidate`、`SearchConstraints`、`NasSolution`。
- 调用 SciPy MILP 求解每层选择一个候选块的 grouped-knapsack 问题。
- 支持 memory、runtime/latency、throughput 和 diversity constraint。

## 2. 远端环境探测

### 2.1 GPU 和 Python 探测命令

```bash
ssh -tt -p 33052 root@connect.nma1.seetacloud.com \
  'cd /root && mkdir -p /root/autodl-tmp/PUZZLE && \
   nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader && \
   python3 --version && pwd && df -h /root/autodl-tmp'
```

输出：

```text
NVIDIA A800 80GB PCIe, 81920 MiB, 81153 MiB
bash: line 1: python3: command not found
```

结论：

- GPU：`NVIDIA A800 80GB PCIe`
- 远端无 `python3` 命令，但存在 `/root/miniconda3/bin/python`。

### 2.2 Conda Python 和依赖探测命令

```bash
ssh -tt -p 33052 root@connect.nma1.seetacloud.com \
  'cd /root/autodl-tmp/PUZZLE && /root/miniconda3/bin/python - <<'"'"'PY'"'"'
import sys
print(sys.executable)
print(sys.version)
mods = ["torch", "transformers", "scipy", "numpy", "accelerate", "sentencepiece", "tiktoken"]
for m in mods:
    try:
        mod = __import__(m)
        print(m, "OK", getattr(mod, "__version__", ""))
    except Exception as e:
        print(m, "NO", type(e).__name__, str(e)[:120])
import torch
print("cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu", torch.cuda.get_device_name(0))
    print("capability", torch.cuda.get_device_capability(0))
PY'
```

输出：

```text
/root/miniconda3/bin/python
3.12.3 | packaged by Anaconda, Inc. | (main, May  6 2024, 19:46:43) [GCC 11.2.0]
torch OK 2.3.0+cu121
transformers NO ModuleNotFoundError No module named 'transformers'
scipy NO ModuleNotFoundError No module named 'scipy'
numpy OK 1.26.4
accelerate NO ModuleNotFoundError No module named 'accelerate'
sentencepiece NO ModuleNotFoundError No module named 'sentencepiece'
tiktoken NO ModuleNotFoundError No module named 'tiktoken'
cuda True
gpu NVIDIA A800 80GB PCIe
capability (8, 0)
```

## 3. 同步项目到服务器

### 3.1 首次同步命令

```bash
rsync -az --delete \
  --exclude '__pycache__' \
  --exclude '.git' \
  -e 'ssh -p 33052 -o StrictHostKeyChecking=no' \
  ./ root@connect.nma1.seetacloud.com:/root/autodl-tmp/PUZZLE/
```

输出：

```text
# rsync 正常完成，无错误输出
```

### 3.2 后续同步命令

为避免删除远端项目内依赖、模型缓存和结果文件，后续同步排除了 `vendor/python`、`hf_cache`、`checkpoints`、`outputs`：

```bash
rsync -az \
  --exclude '__pycache__' \
  --exclude '.git' \
  --exclude 'vendor/python' \
  --exclude 'hf_cache' \
  --exclude 'checkpoints' \
  --exclude 'outputs' \
  -e 'ssh -p 33052 -o StrictHostKeyChecking=no' \
  ./ root@connect.nma1.seetacloud.com:/root/autodl-tmp/PUZZLE/
```

输出：

```text
# rsync 正常完成，无错误输出
```

## 4. 项目内依赖安装

### 4.1 创建项目内目录

```bash
ssh -tt -p 33052 root@connect.nma1.seetacloud.com \
  'cd /root/autodl-tmp/PUZZLE && \
   mkdir -p vendor/python hf_cache/models checkpoints outputs'
```

### 4.2 安装依赖到 `vendor/python`

```bash
ssh -tt -p 33052 root@connect.nma1.seetacloud.com \
  'cd /root/autodl-tmp/PUZZLE && \
   /root/miniconda3/bin/pip install -q \
     --target vendor/python \
     --upgrade \
     --no-deps \
     "transformers==4.52.4" accelerate safetensors tokenizers sentencepiece \
     tiktoken huggingface-hub regex requests filelock packaging pyyaml tqdm \
     numpy scipy'
```

输出：

```text
WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager...
```

### 4.3 修正 `tokenizers` 版本

原因：`transformers==4.52.4` 要求 `tokenizers>=0.21,<0.22`。

```bash
ssh -tt -p 33052 root@connect.nma1.seetacloud.com \
  'cd /root/autodl-tmp/PUZZLE && \
   rm -rf vendor/python/tokenizers vendor/python/tokenizers-* && \
   /root/miniconda3/bin/pip install -q \
     --target vendor/python \
     --upgrade \
     --no-deps "tokenizers>=0.21,<0.22"'
```

输出：

```text
WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager...
```

### 4.4 修正 `huggingface-hub` 版本

原因：`transformers==4.52.4` 要求 `huggingface-hub>=0.30.0,<1.0`。

```bash
ssh -tt -p 33052 root@connect.nma1.seetacloud.com \
  'cd /root/autodl-tmp/PUZZLE && \
   rm -rf vendor/python/huggingface_hub vendor/python/huggingface_hub-* vendor/python/huggingface-hub-* && \
   /root/miniconda3/bin/pip install -q \
     --target vendor/python \
     --upgrade \
     --no-deps "huggingface-hub>=0.30,<1.0"'
```

输出：

```text
WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager...
```

### 4.5 项目内依赖验证

```bash
ssh -tt -p 33052 root@connect.nma1.seetacloud.com \
  'cd /root/autodl-tmp/PUZZLE && \
   PYTHONDONTWRITEBYTECODE=1 /root/miniconda3/bin/python - <<'"'"'PY'"'"'
import sys, os
from pathlib import Path
root = Path("/root/autodl-tmp/PUZZLE")
sys.path.insert(0, str(root / "vendor/python"))
os.environ["HF_HOME"] = str(root / "hf_cache")
import transformers, tokenizers, huggingface_hub, torch
print("transformers", transformers.__version__, transformers.__file__)
print("tokenizers", tokenizers.__version__, tokenizers.__file__)
print("huggingface_hub", huggingface_hub.__version__, huggingface_hub.__file__)
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("hf_home", os.environ["HF_HOME"])
PY'
```

输出：

```text
transformers 4.52.4 /root/autodl-tmp/PUZZLE/vendor/python/transformers/__init__.py
tokenizers 0.21.4 /root/autodl-tmp/PUZZLE/vendor/python/tokenizers/__init__.py
huggingface_hub 0.36.2 /root/autodl-tmp/PUZZLE/vendor/python/huggingface_hub/__init__.py
torch 2.3.0+cu121 cuda True
hf_home /root/autodl-tmp/PUZZLE/hf_cache
```

## 5. 项目测试

### 5.1 本地测试命令

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest
```

输出：

```text
.....
----------------------------------------------------------------------
Ran 5 tests in 0.483s

OK
```

### 5.2 远端测试命令

```bash
ssh -tt -p 33052 root@connect.nma1.seetacloud.com \
  'cd /root/autodl-tmp/PUZZLE && \
   PYTHONDONTWRITEBYTECODE=1 /root/miniconda3/bin/python -m unittest'
```

输出：

```text
.....
----------------------------------------------------------------------
Ran 5 tests in 1.636s

OK
```

## 6. Qwen3-0.6B 实际运行

### 6.1 执行命令

```bash
ssh -tt -p 33052 root@connect.nma1.seetacloud.com \
  'cd /root/autodl-tmp/PUZZLE && \
   HF_ENDPOINT=https://hf-mirror.com \
   HF_HUB_DISABLE_XET=1 \
   PYTHONDONTWRITEBYTECODE=1 \
   /root/miniconda3/bin/python examples/run_qwen3_attention_search.py \
     --model-id Qwen/Qwen3-0.6B \
     --device gpu \
     --max-layers 8 \
     --max-prompts 2 \
     --seq-len 128 \
     --cache-dir /root/autodl-tmp/PUZZLE/hf_cache/models \
     --output /root/autodl-tmp/PUZZLE/outputs/qwen3_0_6b_layer_skip_search.json \
     --pth-output /root/autodl-tmp/PUZZLE/checkpoints/qwen3_0_6b_layer_skip_search.pth'
```

### 6.2 下载输出摘要

首次运行时模型文件下载到项目内缓存：

```text
tokenizer_config.json: 9.73kB
vocab.json: 2.78MB
merges.txt: 1.67MB
tokenizer.json: 11.4MB
config.json: 726B
model.safetensors: 1.50G
generation_config.json: 239B
```

缓存目录：

```text
/root/autodl-tmp/PUZZLE/hf_cache/models
```

### 6.3 运行输出

```json
{
  "selected_batch_size": 1,
  "total_kl_score": 0.22088745397232934,
  "total_memory_bytes": 216566272.0,
  "total_runtime_proxy": 0.013693779968000001,
  "throughput_proxy": 9347.309530247594,
  "selected": [
    "L0:parent",
    "L1:parent",
    "L2:parent",
    "L3:parent",
    "L4:parent",
    "L5:skip_attn",
    "L6:skip_attn",
    "L7:skip_attn"
  ]
}
```

脚本输出路径：

```text
wrote=/root/autodl-tmp/PUZZLE/outputs/qwen3_0_6b_layer_skip_search.json
wrote_pth=/root/autodl-tmp/PUZZLE/checkpoints/qwen3_0_6b_layer_skip_search.pth
```

## 7. 产物验证

### 7.1 验证命令

```bash
ssh -tt -p 33052 root@connect.nma1.seetacloud.com \
  'cd /root/autodl-tmp/PUZZLE && \
   echo "== project files ==" && ls -lh outputs checkpoints && \
   echo "== dirs ==" && du -sh hf_cache vendor checkpoints outputs 2>/dev/null && \
   echo "== default cache qwen leftovers ==" && \
   find /root/.cache/huggingface -maxdepth 4 -iname "*Qwen3-0.6B*" -print 2>/dev/null | head -20'
```

输出：

```text
== project files ==
outputs:
total 12K
-rw-r--r-- 1 root root 9.3K Jun  9 10:15 qwen3_0_6b_layer_skip_search.json

checkpoints:
total 4.0K
-rw-r--r-- 1 root root 3.8K Jun  9 10:15 qwen3_0_6b_layer_skip_search.pth

== dirs ==
1.5G    hf_cache
343M    vendor
4.0K    checkpoints
12K     outputs

== default cache qwen leftovers ==
# 无输出，表示默认 /root/.cache/huggingface 下没有 Qwen3-0.6B 残留
```

### 7.2 JSON 头部摘要

```json
{
  "model_id": "Qwen/Qwen3-0.6B",
  "device": "cuda",
  "dtype": "torch.bfloat16",
  "root": "/root/autodl-tmp/PUZZLE",
  "cache_dir": "/root/autodl-tmp/PUZZLE/hf_cache/models",
  "vendor_dir": "/root/autodl-tmp/PUZZLE/vendor/python",
  "num_model_layers": 28,
  "searched_layers": [0, 1, 2, 3, 4, 5, 6, 7],
  "num_prompts": 2,
  "seq_len": 128,
  "constraints": {
    "memory_max": 220064890.88,
    "latency_max": 0.01385510207488,
    "target_param_fraction": 0.86,
    "target_runtime_fraction": 0.86
  },
  "solution": {
    "selected_batch_size": 1,
    "total_kl_score": 0.22088745397232934,
    "total_memory_bytes": 216566272.0,
    "total_runtime_proxy": 0.013693779968000001,
    "throughput_proxy": 9347.309530247594,
    "selected": [
      "L0:parent",
      "L1:parent",
      "L2:parent",
      "L3:parent",
      "L4:parent",
      "L5:skip_attn",
      "L6:skip_attn",
      "L7:skip_attn"
    ]
  }
}
```

## 8. 最终目录结构摘要

```text
/root/autodl-tmp/PUZZLE
├── examples/run_qwen3_attention_search.py
├── puzzle_nas/
├── vendor/python/                         # 项目内 Python 依赖库
├── hf_cache/models/                       # Qwen3-0.6B 权重和 tokenizer 缓存
├── outputs/qwen3_0_6b_layer_skip_search.json    # JSON 运行结果
└── checkpoints/qwen3_0_6b_layer_skip_search.pth         # torch.save 后的运行结果
```

## 9. 注意事项

- 当前 `.pth` 保存的是 NAS 搜索结果、分数、约束和架构选择，不是重新训练后的 Qwen 模型权重。
- Qwen3-0.6B 原始权重以 HuggingFace cache/safetensors 形式保存在 `hf_cache/models`。
- 本次示例是 no-op-only NAS 搜索，未执行完整 BLD/GKD 训练。
- `HF_ENDPOINT=https://hf-mirror.com` 用于解决服务器直接访问 HuggingFace 超时的问题。

## 10. 增加 Flash Linear Attention 候选

后续已将 `fla-org/flash-linear-attention` 中的 `LinearAttention` 作为额外候选加入，不再只有 no-op 类候选。

### 10.1 新候选

候选集合变为：

```text
parent
skip_attn
skip_mlp
skip_both
fla_linear_attn
```

`fla_linear_attn` 的行为：

- 使用 `flash-linear-attention` 的 `LinearAttention` 替换 Qwen decoder layer 中的 `self_attn`。
- 保留原 layer 的 MLP。
- 根据 Qwen attention 的 q/k/v/o 投影形状设置 `expand_k` 和 `expand_v`。
- 用原始 Qwen attention 的 q/k/v/o 权重初始化 FLA 的 q/k/v/o 权重。
- 将 KV-cache memory 估计为 0，因为该候选不使用标准 Transformer KV cache。

### 10.2 FLA 相关项目内路径

```text
/root/autodl-tmp/PUZZLE/vendor/flash-linear-attention
/root/autodl-tmp/PUZZLE/vendor/flash-linear-attention-v0.4.2
/root/autodl-tmp/PUZZLE/vendor/python
```

说明：

- 最新版 FLA 官方要求 `torch>=2.7.0` 和 `triton>=3.3`。
- 当前服务器环境是 `torch==2.3.0+cu121`、Python 3.12，因此使用 `flash-linear-attention` 的 `v0.4.2` 源码，并在脚本中加入了单卡推理所需的兼容 shim。
- 这些 shim 只用于绕开旧 PyTorch 下的 `torch.compile` 和分布式张量类型导入问题，不启用分布式 tensor parallel。

### 10.3 FLA smoke test 命令

```bash
ssh -tt -p 33052 root@connect.nma1.seetacloud.com \
  'cd /root/autodl-tmp/PUZZLE && \
   HF_ENDPOINT=https://hf-mirror.com \
   HF_HUB_DISABLE_XET=1 \
   PYTHONDONTWRITEBYTECODE=1 \
   /root/miniconda3/bin/python examples/run_qwen3_attention_search.py \
     --model-id Qwen/Qwen3-0.6B \
     --device gpu \
     --max-layers 1 \
     --max-prompts 1 \
     --seq-len 64 \
     --variants parent,skip_attn,fla_linear_attn \
     --cache-dir /root/autodl-tmp/PUZZLE/hf_cache/models \
     --output /root/autodl-tmp/PUZZLE/outputs/qwen3_0_6b_fla_smoke.json \
     --pth-output /root/autodl-tmp/PUZZLE/checkpoints/qwen3_0_6b_fla_smoke.pth'
```

输出：

```json
{
  "selected_batch_size": 1,
  "total_kl_score": 14.706918716430664,
  "total_memory_bytes": 18878464.0,
  "total_runtime_proxy": 0.000604110848,
  "throughput_proxy": 105940.82230418746,
  "selected": [
    "L0:skip_attn"
  ]
}
```

smoke test 的 JSON 中确认 `fla_linear_attn` 已参与打分：

```json
{
  "layer_idx": 0,
  "variant": "fla_linear_attn",
  "kl": 9.951766967773438,
  "effective_param_memory_bytes": 31461376,
  "kv_cache_memory_bytes": 0,
  "runtime_proxy": 0.001006764032,
  "measured_seconds": 0.764012610539794
}
```

### 10.4 8 层 FLA 候选运行命令

```bash
ssh -tt -p 33052 root@connect.nma1.seetacloud.com \
  'cd /root/autodl-tmp/PUZZLE && \
   HF_ENDPOINT=https://hf-mirror.com \
   HF_HUB_DISABLE_XET=1 \
   PYTHONDONTWRITEBYTECODE=1 \
   /root/miniconda3/bin/python examples/run_qwen3_attention_search.py \
     --model-id Qwen/Qwen3-0.6B \
     --device gpu \
     --max-layers 8 \
     --max-prompts 2 \
     --seq-len 128 \
     --variants parent,skip_attn,skip_mlp,skip_both,fla_linear_attn \
     --cache-dir /root/autodl-tmp/PUZZLE/hf_cache/models \
     --output /root/autodl-tmp/PUZZLE/outputs/qwen3_0_6b_fla_attention_search.json \
     --pth-output /root/autodl-tmp/PUZZLE/checkpoints/qwen3_0_6b_fla_attention_search.pth'
```

输出：

```json
{
  "selected_batch_size": 1,
  "total_kl_score": 0.22088745397232934,
  "total_memory_bytes": 216566272.0,
  "total_runtime_proxy": 0.013693779968000001,
  "throughput_proxy": 9347.309530247594,
  "selected": [
    "L0:parent",
    "L1:parent",
    "L2:parent",
    "L3:parent",
    "L4:parent",
    "L5:skip_attn",
    "L6:skip_attn",
    "L7:skip_attn"
  ]
}
```

结果文件：

```text
/root/autodl-tmp/PUZZLE/outputs/qwen3_0_6b_fla_attention_search.json
/root/autodl-tmp/PUZZLE/checkpoints/qwen3_0_6b_fla_attention_search.pth
```

结果确认：

```text
variants ['parent', 'skip_attn', 'skip_mlp', 'skip_both', 'fla_linear_attn']
fla {'mode': 'chunk', 'feature_map': 'elu'}
fla_score_count 8
first_fla {
  'layer_idx': 0,
  'variant': 'fla_linear_attn',
  'kl': 8.903446674346924,
  'effective_param_memory_bytes': 31461376,
  'kv_cache_memory_bytes': 0,
  'runtime_proxy': 0.002013528064,
  'measured_seconds': 0.7403501532971859
}
```
