# Qwen3-0.6B 加入 Flash Linear Attention 候选运行记录

记录日期：2026-06-09  
远端工作目录：`/root/autodl-tmp/PUZZLE`  
远端服务器：`ssh -p 33052 root@connect.nma1.seetacloud.com`  
目标：将 [`fla-org/flash-linear-attention`](https://github.com/fla-org/flash-linear-attention) 中的 `LinearAttention` 作为 NAS 搜索候选项，不再只有 no-op 类候选。

> 说明：本文不记录 SSH 密码。所有模型权重、Python 依赖、FLA 源码、JSON 输出和 `.pth` 输出均放在 `/root/autodl-tmp/PUZZLE` 下。

## 1. 最终结论

已新增候选：

```text
fla_linear_attn
```

最终候选集合：

```text
parent
skip_attn
skip_mlp
skip_both
fla_linear_attn
```

远端已完成两次验证运行：

```text
/root/autodl-tmp/PUZZLE/outputs/qwen3_0_6b_fla_smoke.json
/root/autodl-tmp/PUZZLE/checkpoints/qwen3_0_6b_fla_smoke.pth

/root/autodl-tmp/PUZZLE/outputs/qwen3_0_6b_fla_attention_search.json
/root/autodl-tmp/PUZZLE/checkpoints/qwen3_0_6b_fla_attention_search.pth
```

8 层正式运行确认：

```text
variants ['parent', 'skip_attn', 'skip_mlp', 'skip_both', 'fla_linear_attn']
fla_count 8
```

这说明 8 个搜索层都实际计算了 `fla_linear_attn` 的 replace-1-layer KL 分数。

## 2. 代码改动说明

主脚本：

```text
examples/run_qwen3_attention_search.py
```

新增逻辑：

- 优先从项目内 FLA 源码目录导入：

```text
/root/autodl-tmp/PUZZLE/vendor/flash-linear-attention-v0.4.2
/root/autodl-tmp/PUZZLE/vendor/flash-linear-attention
```

- 新增 `QwenCandidateLayer`，支持以下候选：

```text
parent
skip_attn
skip_mlp
skip_both
fla_linear_attn
```

- `fla_linear_attn` 使用 `fla.layers.LinearAttention` 替换 Qwen decoder layer 中的 `self_attn`。
- 保留该层原始 MLP。
- 用原 Qwen attention 的 `q_proj/k_proj/v_proj/o_proj` 权重初始化 FLA 的对应投影。
- 根据 Qwen3-0.6B 实际投影形状动态设置：

```text
expand_k = parent_attn.q_proj.out_features / hidden_size
expand_v = parent_attn.o_proj.in_features / hidden_size
```

- 将 `fla_linear_attn` 的标准 Transformer KV-cache memory 估计为 0。
- 对当前远端环境加入单卡兼容 shim：
  - `torch.compile` 不可用时改为 identity。
  - 补齐旧 PyTorch 缺失的 `DeviceMesh`、`Replicate`、`Shard`、`Placement`、`DTensor`、`distribute_module` 符号。

## 3. 远端环境

已有环境：

```text
torch 2.3.0+cu121
cuda True
gpu NVIDIA A800 80GB PCIe
python /root/miniconda3/bin/python
```

已有模型缓存：

```text
/root/autodl-tmp/PUZZLE/hf_cache/models
```

已有项目内依赖：

```text
/root/autodl-tmp/PUZZLE/vendor/python
```

## 4. 本地检查命令

### 4.1 Python 语法检查

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m py_compile examples/run_qwen3_attention_search.py
```

输出：

```text
# 无输出，表示通过
```

### 4.2 单元测试

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest
```

输出：

```text
.....
----------------------------------------------------------------------
Ran 5 tests in 0.617s

OK
```

## 5. 远端依赖探测

命令：

```bash
ssh -tt -p 33052 root@connect.nma1.seetacloud.com \
  'cd /root/autodl-tmp/PUZZLE && /root/miniconda3/bin/python - <<'"'"'PY'"'"'
import importlib
for name in ["triton", "einops", "ninja", "packaging", "torch"]:
    try:
        mod = importlib.import_module(name)
        print(name, "OK", getattr(mod, "__version__", ""))
    except Exception as e:
        print(name, "NO", type(e).__name__, str(e)[:80])
PY'
```

输出：

```text
triton NO ModuleNotFoundError No module named 'triton'
einops NO ModuleNotFoundError No module named 'einops'
ninja NO ModuleNotFoundError No module named 'ninja'
packaging OK 26.2
torch OK 2.3.0+cu121
```

结论：远端缺少 FLA 所需依赖，需要安装到项目内 `vendor/python`。

## 6. 安装 FLA 依赖到项目目录

### 6.1 安装 Triton 2.3.0、einops、ninja 和 PyPI FLA

命令：

```bash
ssh -tt -p 33052 root@connect.nma1.seetacloud.com \
  'cd /root/autodl-tmp/PUZZLE && \
   /root/miniconda3/bin/pip install -q \
     --target vendor/python \
     --upgrade \
     --no-deps "triton==2.3.0" einops ninja && \
   /root/miniconda3/bin/pip install -q \
     --target vendor/python \
     --upgrade \
     --no-deps flash-linear-attention'
```

输出：

```text
WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager...
```

### 6.2 验证 PyPI FLA 失败

命令：

```bash
ssh -tt -p 33052 root@connect.nma1.seetacloud.com \
  'cd /root/autodl-tmp/PUZZLE && \
   PYTHONDONTWRITEBYTECODE=1 /root/miniconda3/bin/python - <<'"'"'PY'"'"'
import sys
from pathlib import Path
root = Path("/root/autodl-tmp/PUZZLE")
sys.path.insert(0, str(root / "vendor/python"))
import torch, triton, einops
from fla.layers import LinearAttention
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("triton", triton.__version__, triton.__file__)
print("LinearAttention", LinearAttention)
PY'
```

输出：

```text
ModuleNotFoundError: No module named 'fla.ops'
```

结论：PyPI 包缺少 `fla.ops`，需要使用 GitHub 源码。

## 7. 获取 FLA 源码

### 7.1 远端直接 clone 失败

命令：

```bash
ssh -tt -p 33052 root@connect.nma1.seetacloud.com \
  'cd /root/autodl-tmp/PUZZLE && \
   mkdir -p vendor && \
   git clone --depth 1 https://github.com/fla-org/flash-linear-attention.git vendor/flash-linear-attention'
```

输出：

```text
Cloning into 'vendor/flash-linear-attention'...
fatal: unable to access 'https://github.com/fla-org/flash-linear-attention.git/': GnuTLS recv error (-110): The TLS connection was non-properly terminated.
```

结论：远端 GitHub 连接不稳定，改为本地 clone 后 rsync 到远端。

### 7.2 本地 clone 最新源码

命令：

```bash
rm -rf /tmp/flash-linear-attention && \
git clone --depth 1 https://github.com/fla-org/flash-linear-attention.git /tmp/flash-linear-attention && \
find /tmp/flash-linear-attention/fla -maxdepth 2 -type d | head -30
```

输出摘要：

```text
Cloning into '/tmp/flash-linear-attention'...
/tmp/flash-linear-attention/fla
/tmp/flash-linear-attention/fla/layers
/tmp/flash-linear-attention/fla/utils
/tmp/flash-linear-attention/fla/models
...
/tmp/flash-linear-attention/fla/models/linear_attn
```

### 7.3 同步最新源码到远端

命令：

```bash
rsync -az --delete \
  --exclude '.git' \
  --exclude '__pycache__' \
  /tmp/flash-linear-attention/ \
  root@connect.nma1.seetacloud.com:/root/autodl-tmp/PUZZLE/vendor/flash-linear-attention/ \
  -e 'ssh -p 33052 -o StrictHostKeyChecking=no'
```

输出：

```text
# rsync 正常完成，无错误输出
```

### 7.4 最新 FLA 源码验证失败

命令：

```bash
ssh -tt -p 33052 root@connect.nma1.seetacloud.com \
  'cd /root/autodl-tmp/PUZZLE && \
   PYTHONDONTWRITEBYTECODE=1 /root/miniconda3/bin/python - <<'"'"'PY'"'"'
import sys
from pathlib import Path
root = Path("/root/autodl-tmp/PUZZLE")
sys.path.insert(0, str(root / "vendor/flash-linear-attention"))
sys.path.insert(1, str(root / "vendor/python"))
import torch, triton
from fla.layers import LinearAttention
PY'
```

输出摘要：

```text
Current Triton version 2.3.0 is below the recommended 3.3.0 version.
AssertionError: Only cuda device is supported for PyTorch version < 2.4.0.
```

结论：最新版 FLA 官方 `pyproject.toml` 要求：

```text
torch>=2.7.0
triton>=3.3
```

当前服务器为 `torch==2.3.0+cu121`，因此改用 FLA v0.4.2 源码。

## 8. 获取 FLA v0.4.2 源码

### 8.1 查看可用 tag

命令：

```bash
git ls-remote --tags https://github.com/fla-org/flash-linear-attention.git | tail -50
```

输出摘要：

```text
refs/tags/v0.1.0
refs/tags/v0.1.1
refs/tags/v0.1.2
refs/tags/v0.2.0
refs/tags/v0.2.1
refs/tags/v0.2.2
refs/tags/v0.3.0
refs/tags/v0.3.1
refs/tags/v0.3.2
refs/tags/v0.4.0
refs/tags/v0.4.1
refs/tags/v0.4.2
refs/tags/v0.5.0
```

### 8.2 本地 clone v0.4.2

命令：

```bash
rm -rf /tmp/flash-linear-attention-v0.4.2 && \
git clone --depth 1 --branch v0.4.2 https://github.com/fla-org/flash-linear-attention.git /tmp/flash-linear-attention-v0.4.2 && \
find /tmp/flash-linear-attention-v0.4.2/fla/ops -maxdepth 1 -type d | head
```

输出摘要：

```text
Cloning into '/tmp/flash-linear-attention-v0.4.2'...
/tmp/flash-linear-attention-v0.4.2/fla/ops
/tmp/flash-linear-attention-v0.4.2/fla/ops/kda
/tmp/flash-linear-attention-v0.4.2/fla/ops/abc
/tmp/flash-linear-attention-v0.4.2/fla/ops/delta_rule
...
```

### 8.3 同步 v0.4.2 到远端

命令：

```bash
rsync -az --delete \
  --exclude '.git' \
  --exclude '__pycache__' \
  /tmp/flash-linear-attention-v0.4.2/ \
  root@connect.nma1.seetacloud.com:/root/autodl-tmp/PUZZLE/vendor/flash-linear-attention-v0.4.2/ \
  -e 'ssh -p 33052 -o StrictHostKeyChecking=no'
```

输出：

```text
# rsync 正常完成，无错误输出
```

## 9. FLA v0.4.2 兼容问题和修复

### 9.1 设备检测失败

验证命令：

```bash
ssh -tt -p 33052 root@connect.nma1.seetacloud.com \
  'cd /root/autodl-tmp/PUZZLE && \
   PYTHONDONTWRITEBYTECODE=1 /root/miniconda3/bin/python - <<'"'"'PY'"'"'
import sys
from pathlib import Path
root = Path("/root/autodl-tmp/PUZZLE")
sys.path.insert(0, str(root / "vendor/flash-linear-attention-v0.4.2"))
sys.path.insert(1, str(root / "vendor/python"))
import torch, triton
from fla.layers import LinearAttention
PY'
```

输出：

```text
Current Triton version 2.3.0 is below the recommended 3.2.0 version.
AssertionError: Only cuda device is supported for PyTorch version < 2.4.0.
```

修复：升级项目内 Triton。

```bash
ssh -tt -p 33052 root@connect.nma1.seetacloud.com \
  'cd /root/autodl-tmp/PUZZLE && \
   rm -rf vendor/python/triton vendor/python/triton-* vendor/fla-0.4.2/triton vendor/fla-0.4.2/triton-* && \
   /root/miniconda3/bin/pip install -q \
     --target vendor/python \
     --upgrade \
     --no-deps "triton==3.2.0"'
```

输出：

```text
WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager...
```

### 9.2 `torch.compile` 不支持 Python 3.12

输出：

```text
RuntimeError: Dynamo is not supported on Python 3.12+
```

脚本修复：

```python
try:
    torch.compile(lambda x: x)
except Exception:
    def identity_compile(fn=None, *args, **kwargs):
        if fn is None:
            return lambda wrapped: wrapped
        return fn

    torch.compile = identity_compile
```

### 9.3 旧 PyTorch 缺少分布式 tensor 符号

依次遇到：

```text
ImportError: cannot import name 'DeviceMesh' from 'torch.distributed'
ImportError: cannot import name 'Replicate' from 'torch.distributed.tensor'
ImportError: cannot import name 'Placement' from 'torch.distributed.tensor'
ImportError: cannot import name 'DTensor' from 'torch.distributed.tensor'
```

脚本修复：

```python
if hasattr(torch, "distributed") and not hasattr(torch.distributed, "DeviceMesh"):
    torch.distributed.DeviceMesh = object
try:
    import torch.distributed.tensor as distributed_tensor
except Exception:
    distributed_tensor = None
if distributed_tensor is not None:
    if not hasattr(distributed_tensor, "Replicate"):
        distributed_tensor.Replicate = object
    if not hasattr(distributed_tensor, "Shard"):
        distributed_tensor.Shard = object
    if not hasattr(distributed_tensor, "Placement"):
        distributed_tensor.Placement = object
    if not hasattr(distributed_tensor, "DTensor"):
        distributed_tensor.DTensor = torch.Tensor
    if not hasattr(distributed_tensor, "distribute_module"):
        def distribute_module(module, *args, **kwargs):
            return module

        distributed_tensor.distribute_module = distribute_module
```

说明：这些 shim 只用于单卡推理候选打分，不启用分布式 tensor parallel。

### 9.4 Qwen3 投影维度不等于 hidden size

错误：

```text
ValueError: cannot initialize FLA weight: target torch.Size([1024, 1024]) != source torch.Size([2048, 1024])
```

原因：Qwen3-0.6B 的 `q_proj` 和 `o_proj` 形状不是简单的 `hidden_size -> hidden_size`。

脚本修复：

```python
expand_k = parent_attn.q_proj.out_features / hidden_size
expand_v = parent_attn.o_proj.in_features / hidden_size
```

### 9.5 FLA v0.4.2 fused recurrent normalize bug

错误：

```text
TypeError: unsupported operand type(s) for *: 'Tensor' and 'NoneType'
```

触发点：`do_feature_map_norm=True` 时 `scale=None`。

脚本修复：

```python
do_feature_map_norm=False
```

## 10. 同步脚本到远端

同步命令：

```bash
rsync -az \
  --exclude '__pycache__' \
  --exclude '.git' \
  --exclude 'vendor/python' \
  --exclude 'vendor/flash-linear-attention' \
  --exclude 'vendor/flash-linear-attention-v0.4.2' \
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

## 11. Smoke Test：1 层 + FLA 候选

### 11.1 执行命令

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

### 11.2 输出

```text
/root/autodl-tmp/PUZZLE/vendor/python/transformers/utils/hub.py:111: FutureWarning: Using `TRANSFORMERS_CACHE` is deprecated...
PyTorch < 2.4 detected - computations may be slower due to lack of optimizations
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
wrote=/root/autodl-tmp/PUZZLE/outputs/qwen3_0_6b_fla_smoke.json
wrote_pth=/root/autodl-tmp/PUZZLE/checkpoints/qwen3_0_6b_fla_smoke.pth
```

### 11.3 Smoke Test JSON 摘要

```json
{
  "model_id": "Qwen/Qwen3-0.6B",
  "device": "cuda",
  "dtype": "torch.bfloat16",
  "root": "/root/autodl-tmp/PUZZLE",
  "cache_dir": "/root/autodl-tmp/PUZZLE/hf_cache/models",
  "vendor_dir": "/root/autodl-tmp/PUZZLE/vendor/python",
  "fla_repo_dir": "/root/autodl-tmp/PUZZLE/vendor/flash-linear-attention-v0.4.2",
  "searched_layers": [0],
  "variants": ["parent", "skip_attn", "fla_linear_attn"],
  "fla": {
    "mode": "chunk",
    "feature_map": "elu"
  }
}
```

确认 `fla_linear_attn` 已参与打分：

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

说明：

- MIP 最终选了 `L0:skip_attn`。
- 但 `fla_linear_attn` 确实被 forward、打 KL 分、进入 MIP 候选集合。

## 12. 正式运行：8 层 + 完整候选集合

### 12.1 执行命令

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

### 12.2 输出

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

脚本写入：

```text
wrote=/root/autodl-tmp/PUZZLE/outputs/qwen3_0_6b_fla_attention_search.json
wrote_pth=/root/autodl-tmp/PUZZLE/checkpoints/qwen3_0_6b_fla_attention_search.pth
```

说明：

- `fla_linear_attn` 被纳入 8 个搜索层的候选并完成 KL 打分。
- 当前约束和分数下，MIP 最终没有选择 FLA，而是选择了 `parent` 和 `skip_attn` 的组合。

## 13. 最终产物验证

### 13.1 验证命令

```bash
ssh -tt -p 33052 root@connect.nma1.seetacloud.com \
  'cd /root/autodl-tmp/PUZZLE && \
   echo "== ls outputs checkpoints ==" && \
   ls -lh outputs/qwen3_0_6b_fla_smoke.json outputs/qwen3_0_6b_fla_attention_search.json \
          checkpoints/qwen3_0_6b_fla_smoke.pth checkpoints/qwen3_0_6b_fla_attention_search.pth && \
   echo "== du ==" && \
   du -sh vendor/flash-linear-attention-v0.4.2 vendor/python hf_cache outputs checkpoints && \
   echo "== json summary ==" && \
   /root/miniconda3/bin/python - <<'"'"'PY'"'"'
import json
for path in ["outputs/qwen3_0_6b_fla_smoke.json", "outputs/qwen3_0_6b_fla_attention_search.json"]:
    data = json.load(open(path))
    print(path)
    print("  variants", data["variants"])
    print("  selected", data["solution"]["selected"])
    print("  total_kl", data["solution"]["total_kl_score"])
    print("  fla_count", sum(1 for s in data["scores"] if s["variant"] == "fla_linear_attn"))
    first = next(s for s in data["scores"] if s["variant"] == "fla_linear_attn")
    print("  first_fla", first)
PY'
```

### 13.2 验证输出

```text
== ls outputs checkpoints ==
-rw-r--r-- 1 root root  12K Jun  9 11:03 outputs/qwen3_0_6b_fla_attention_search.json
-rw-r--r-- 1 root root 1.8K Jun  9 11:03 outputs/qwen3_0_6b_fla_smoke.json
-rw-r--r-- 1 root root 4.4K Jun  9 11:03 checkpoints/qwen3_0_6b_fla_attention_search.pth
-rw-r--r-- 1 root root 2.2K Jun  9 11:03 checkpoints/qwen3_0_6b_fla_smoke.pth

== du ==
5.3M    vendor/flash-linear-attention-v0.4.2
1.1G    vendor/python
1.5G    hf_cache
28K     outputs
16K     checkpoints

== json summary ==
outputs/qwen3_0_6b_fla_smoke.json
  variants ['parent', 'skip_attn', 'fla_linear_attn']
  selected ['L0:skip_attn']
  total_kl 14.706918716430664
  fla_count 1
  first_fla {'layer_idx': 0, 'variant': 'fla_linear_attn', 'kl': 9.951766967773438, 'effective_param_memory_bytes': 31461376, 'kv_cache_memory_bytes': 0, 'runtime_proxy': 0.001006764032, 'measured_seconds': 0.764012610539794}

outputs/qwen3_0_6b_fla_attention_search.json
  variants ['parent', 'skip_attn', 'skip_mlp', 'skip_both', 'fla_linear_attn']
  selected ['L0:parent', 'L1:parent', 'L2:parent', 'L3:parent', 'L4:parent', 'L5:skip_attn', 'L6:skip_attn', 'L7:skip_attn']
  total_kl 0.22088745397232934
  fla_count 8
  first_fla {'layer_idx': 0, 'variant': 'fla_linear_attn', 'kl': 8.903446674346924, 'effective_param_memory_bytes': 31461376, 'kv_cache_memory_bytes': 0, 'runtime_proxy': 0.002013528064, 'measured_seconds': 0.7403501532971859}
```

## 14. 产物说明

### 14.1 FLA 源码

```text
/root/autodl-tmp/PUZZLE/vendor/flash-linear-attention-v0.4.2
```

说明：

- 这是 `fla-org/flash-linear-attention` 的 `v0.4.2` 源码。
- 当前脚本优先从该目录导入 FLA。
- 使用 v0.4.2 是为了适配远端 `torch==2.3.0+cu121`，因为最新版 FLA 要求更高版本 PyTorch/Triton。

### 14.2 Python 依赖

```text
/root/autodl-tmp/PUZZLE/vendor/python
```

说明：

- 项目内 Python 依赖目录。
- 包含 `transformers`、`tokenizers`、`huggingface_hub`、`triton`、`einops` 等。
- 不依赖系统 site-packages 中的 Transformers。

### 14.3 模型权重缓存

```text
/root/autodl-tmp/PUZZLE/hf_cache/models
```

说明：

- Qwen3-0.6B 原始 HuggingFace 权重和 tokenizer 缓存。
- 不是蒸馏后的权重。

### 14.4 Smoke Test 输出

```text
/root/autodl-tmp/PUZZLE/outputs/qwen3_0_6b_fla_smoke.json
/root/autodl-tmp/PUZZLE/checkpoints/qwen3_0_6b_fla_smoke.pth
```

说明：

- 只搜索第 0 层。
- 候选集合为 `parent,skip_attn,fla_linear_attn`。
- 用于确认 FLA 能正常 forward、打分、进入 MIP。

### 14.5 8 层正式运行输出

```text
/root/autodl-tmp/PUZZLE/outputs/qwen3_0_6b_fla_attention_search.json
/root/autodl-tmp/PUZZLE/checkpoints/qwen3_0_6b_fla_attention_search.pth
```

说明：

- 搜索第 0 到第 7 层。
- 候选集合为 `parent,skip_attn,skip_mlp,skip_both,fla_linear_attn`。
- `.json` 是可读结果。
- `.pth` 是 `torch.save(result, pth_output)` 后的同一份结果对象。
- `.pth` 不是模型权重，也不是 BLD/GKD 后的蒸馏权重。

## 15. 当前限制

- 本次仍是 replace-1-layer scoring + MIP 搜索，没有执行 BLD/GKD 训练。
- FLA 候选已参与搜索，但当前约束下未被最终 MIP 选中。
- FLA v0.4.2 在当前环境下通过兼容 shim 运行；更干净的长期方案是使用官方推荐环境：

```text
torch>=2.7.0
triton>=3.3
```

