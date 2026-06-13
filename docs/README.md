# OmniDistill-NAS

中文说明见 [README.zh-CN.md](README.zh-CN.md)。

This repository contains **OmniDistill-NAS**, a compact, runnable implementation of the main ideas
from `distillation_nas_paper.pdf`: **Distillation-Based NAS for
Inference-Optimized LLMs**.

The implementation keeps the same stages as the paper and exposes two
backends: a tiny causal Transformer for fast local checks, and a Qwen-style
LLM/VLM/VLA backend for real staged BLD, NAS scoring, MIP, assembly, and
GKD/OPD runs:

1. Build a block library from attention and FFN alternatives.
2. Initialize alternatives with training-free transformations.
3. Apply blockwise local distillation (BLD) with normalized MSE.
4. Score candidates with replace-1-block KL or LM loss.
5. Estimate memory/runtime costs for deployment constraints.
6. Solve the grouped knapsack architecture search with MILP.
7. Optionally run global knowledge distillation (GKD) with hidden cosine plus
   logits KL loss, and optionally add on-policy distillation (OPD).

The paper uses large Llama/Nemotron models and target-hardware measurements.
This repo provides a tiny reference path plus real Qwen-style model wrappers.

## Quick run

```bash
python3 tools/run_tiny_nas.py --quick
```

The example automatically uses CUDA or MPS when available. Expected output
includes the device, number of generated block candidates, a selected
architecture, total score, memory, runtime, and throughput estimate.

For the staged shell workflow, see [scripts.md](scripts.md).

Real Qwen-style staged workflow:

```bash
WORKFLOW_BACKEND=qwen MODEL_ID=Qwen/Qwen3-0.6B DEVICE=gpu \
bash scripts/run_all.sh
```

## Verify

```bash
python3 -m compileall distill_nas_core scripts tools tests
python3 -m unittest discover -s tests
```

For the project-level CI check used by contributors:

```bash
python3 tools/check_project.py
```

## Platform Workflow

The shell workflow remains available, but experiments can also be described in
a JSON/YAML spec and run with resume semantics:

```bash
omnidistill run --config configs/toy_experiment.json
omnidistill run --config configs/toy_experiment.json --from-stage evaluate
omnidistill status --workflow-dir outputs/distill_nas_workflow
```

Benchmark and result-zoo commands are first-class CLI paths:

```bash
omnidistill validate configs/toy_experiment.json
omnidistill benchmark --suite benchmarks/suites/toy_smoke.json --dry-run
omnidistill report --results-dir results
```

`run_experiment.py` skips completed artifact-producing stages unless
`--force` is set. The default toy spec runs the paper stages and then adds:

- `evaluate`: writes `09_evaluation/metrics.json`.
- `profile`: writes `10_profiling/profile.json` with measured latency.
- `export`: writes `11_export/manifest.json` plus portable artifact files.
- `report`: writes `report.md`.

Standalone tools are available for each post-processing stage:

```bash
python3 tools/evaluate_artifact.py --artifact-pth outputs/distill_nas_workflow/08_global_knowledge_distillation/gkd_model.pth --output-json outputs/distill_nas_workflow/09_evaluation/metrics.json
python3 tools/profile_artifact.py --artifact-pth outputs/distill_nas_workflow/08_global_knowledge_distillation/gkd_model.pth --output-json outputs/distill_nas_workflow/10_profiling/profile.json
python3 tools/export_artifact.py --artifact-pth outputs/distill_nas_workflow/08_global_knowledge_distillation/gkd_model.pth --export-dir outputs/distill_nas_workflow/11_export
python3 tools/generate_report.py --workflow-dir outputs/distill_nas_workflow
```

## Main modules

- `distill_nas_core.blocks`: causal attention, FFN, no-op/linear subblocks, and
  Transformer blocks.
- `distill_nas_core.search_space`: NAS search specs and paper-style
  initialization for MHA, MQA, GQA, MFA, MLA, MKA, linear attention, pruned
  FFNs, linear FFNs, and no-ops.
- `distill_nas_core.library`: coupled and decoupled BLD block-library builders.
- `distill_nas_core.distill`: BLD, GKD, and optional OPD losses/training loops.
- `distill_nas_core.scoring`: replace-1-block KL and LM-loss scoring.
- `distill_nas_core.resources`: parameter memory, KV-cache memory, and simple
  runtime profiling/estimation.
- `distill_nas_core.mip`: mixed-integer architecture search with optional diversity
  constraints and an exhaustive fallback for tiny cases.
- `distill_nas_core.toy`: a small causal language model used by the demo.
- `distill_nas_core.experiment`: config-driven workflow planning, resume, and
  stage execution.
- `distill_nas_core.evaluation`, `profiler`, `export`, and `reporting`:
  evaluation, measured profiling, portable exports, and generated run reports.
- `distill_nas_core.benchmarks`, `schema`, `result_zoo`, `tracking`, and
  `plugins`: benchmark manifests, validation, result indexing, event logging,
  and extension registration.
- `distill_nas_core.data_adapters`, `quantization`, `distributed`, and `vla`:
  extension points for datasets, calibration, device plans, and VLA rollouts.

Search-space plugins can register attention candidates at runtime:

```python
from distill_nas_core.search_space import register_attention_variant

register_attention_variant("my_attn", spec_factory, module_factory)
```

## GKD With Optional OPD

`global_knowledge_distillation` keeps the original offline GKD objective by
default. To add token OPD, pass a positive `opd_weight` and
`opd_max_new_tokens`; each batch must provide `input_ids`.
The student samples continuations from its current policy; the teacher then
scores those sampled tokens, and the extra loss is the sampled reverse-KL term
`log p_student(token) - log p_teacher(token)` on generated tokens.

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

VLA action-space OPD is also supported: action logits use sampled reverse KL,
and continuous action predictions use action MSE as an action-space
distillation approximation. Continuous-action support does not perform an
environment rollout. These paths follow the on-policy distillation idea
described by Thinking Machines:
https://thinkingmachines.ai/blog/on-policy-distillation/

## Attention Candidates

The toy distillation NAS pipeline now includes layer-level candidates `parent`,
`skip_attn`, `skip_mlp`, and `skip_both`, plus attention candidates
`parent_attn`, `mha_attn`, `quant_mha_attn`, `mqa_attn`, `gqa_kv*`,
`mfa_attn`/`mfa_kv*`, `mla_attn`/`mla_kv*`, `mka_attn`, `linear_attn`,
`noop_attn`, and the FLA-named candidates `fla_linear_attn`,
`fla_gated_linear_attn`, `fla_based_linear_attn`, `fla_rebased_linear_attn`,
`fla_deltanet_attn`, `fla_gated_deltanet_attn`, `fla_kimi_delta_attn`,
`fla_multiscale_retention_attn`, `fla_mla_attn`, `fla_native_sparse_attn`,
and `fla_moba_attn`. The generic toy pipeline uses lightweight PyTorch
equivalents for the FLA names, so they forward, distill, score, and enter MIP.
`all_linear_attn` expands to `linear_attn` plus the FLA linear/delta family,
and `all_core_attn` places those linear candidates at the same level as
MHA/MQA/GQA/MFA/MLA/MKA.
The toy demo defaults to `--attention-variants all_attention` and
`--layer-variants parent,skip_attn,skip_mlp,skip_both`; use `all_qwen_attn`,
`all_linear_attn`, `all_core_attn`, `all_fla`, or comma-separated names to
narrow the search.

## Multi-Objective Search

MIP defaults to the original objective: minimize candidate score under hard
memory/runtime constraints. To make memory and latency part of the objective,
use weighted mode:

```bash
python3 tools/run_staged_toy_pipeline.py mip \
  --objective-mode weighted \
  --score-weight 1.0 \
  --memory-weight 0.25 \
  --runtime-weight 0.25
```

The same flags are available in `tools/run_staged_model_pipeline.py mip`.
Through shell workflow variables:

```bash
OBJECTIVE_MODE=weighted SCORE_WEIGHT=1.0 MEMORY_WEIGHT=0.25 RUNTIME_WEIGHT=0.25 \
bash scripts/06_mip_topk_configs.sh
```

Objectives are normalized by default so score, bytes, and seconds can be mixed
without one unit dominating by scale.

For a broader view of the trade space, run the weight sweep/Pareto report:

```bash
python3 tools/run_multi_objective_search.py \
  --scores-json outputs/distill_nas_workflow/05_nas_layer_scoring/layer_importance.json
```

This writes `multi_objective_search.json`, per-Pareto architecture configs,
`multi_objective_report.md`, and `pareto_front.svg`. For small search spaces it
enumerates the exact feasible Pareto frontier; for larger spaces it uses the
weighted sweep solutions as an approximate frontier.

## Qwen3-0.6B Example

The recommended staged path is:

```bash
WORKFLOW_BACKEND=qwen \
MODEL_ID=Qwen/Qwen3-0.6B \
DEVICE=gpu \
MODEL_VARIANTS=parent,skip_attn,skip_mlp,skip_both,all_core_attn,all_fla \
MAX_LAYERS=2 \
MAX_PROMPTS=2 \
bash scripts/run_all.sh
```

Stages 04-08 write `block_library.pth`, `layer_importance.json`, top-K configs,
`assembled_model.pth`, and `gkd_model.pth`. Assembly and GKD save delta
checkpoints by default: model ID, selected architecture config, and replacement
layer weights. Checkpoint restore is strict by default; set
`ALLOW_PARTIAL_CHECKPOINT_LOAD=1` only when intentionally loading across changed
candidate definitions. Set `SAVE_FULL_STATE_DICT=1` only when you intentionally
want a full model state dict.

`tools/run_qwen3_attention_search.py` runs a real-model NAS candidate
search on `Qwen/Qwen3-0.6B`:

```bash
python3 tools/run_qwen3_attention_search.py \
  --model-id Qwen/Qwen3-0.6B \
  --device gpu \
  --max-layers 8 \
  --max-prompts 2 \
  --seq-len 128
```

It computes replace-1-layer KL scores on GPU and solves MIP over real Qwen
attention candidates, skip candidates, and FLA candidates. `all_qwen_attn`
expands to `parent_attn`, `mha_attn`, `quant_mha_attn`, `mqa_attn`, `gqa_kv2`,
`mfa_kv2`, `mla_kv2`, `mka_attn`, `linear_attn`, and `noop_attn`.
`all_linear_attn` expands to `linear_attn`, `fla_linear_attn`,
`fla_gated_linear_attn`, `fla_based_linear_attn`, `fla_rebased_linear_attn`,
`fla_deltanet_attn`, `fla_gated_deltanet_attn`, and
`fla_kimi_delta_attn`. `all_fla` expands to
`fla_linear_attn`, `fla_gated_linear_attn`, `fla_based_linear_attn`,
`fla_rebased_linear_attn`, `fla_deltanet_attn`, `fla_gated_deltanet_attn`,
`fla_kimi_delta_attn`, `fla_multiscale_retention_attn`, `fla_mla_attn`,
`fla_native_sparse_attn`, and `fla_moba_attn`. `scripts/01_prepare_environment.sh`
clones `fla-org/flash-linear-attention` under the workspace-local
`vendor/flash-linear-attention` by default on Linux GPU hosts; set
`INSTALL_FLA=0` to skip this, or
`REQUIRE_FLA=1` during validation to fail if `fla.layers` is unavailable.
Missing FLA classes or CUDA deps are skipped by default during search; pass
`--no-skip-unavailable-fla` to fail instead.
Results are written under the active workspace to
`outputs/qwen3_0_6b_layer_skip_search.json` and `checkpoints/qwen3_0_6b_layer_skip_search.pth`.
Model weights are cached under `hf_cache/models`, and a local `transformers`
install can live in `vendor/python`.

To smoke test on a few real MMLU samples, install the optional `datasets`
package and run:

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

## Vision-Language Models

The same real-model script can search Qwen-style language decoder layers inside
VLM wrappers. Use `--model-kind vlm` to load with `AutoProcessor` and
`AutoModelForImageTextToText`; `--model-kind auto` also detects common Qwen-VL
ids. Built-in smoke prompts may use a blank RGB image when `--image-path` is
omitted, but dataset prompts require a real image field unless
`--allow-blank-image` is set explicitly. Qwen3-VL checkpoints need a
transformers build that recognizes `qwen3_vl`; `scripts/01_prepare_environment.sh`
installs `transformers>=4.57,<5` into `vendor/python` when the active version is older.

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

For a local image, add `--image-path path/to/image.jpg`. If a smaller
`Qwen3-VL-0.6B` checkpoint is available in your environment, pass that model id
with the same command. The search only swaps the language decoder layers; the
vision encoder and multimodal projection stay unchanged.

## VLA And Common Datasets

`tools/run_qwen3_attention_search.py` also accepts `--model-kind vla` for
vision-language-action models such as OpenVLA-style policies. VLA support uses
the same decoder-layer search as VLM: the vision encoder/action projection stay
unchanged, while Qwen/Llama-style language decoder layers are swapped and
scored against teacher outputs. If the model exposes `action_logits`, those are
scored with KL; continuous `action_mean` outputs are scored with MSE, and
`actions`/`predicted_actions` remain supported as deterministic action values.
For GKD/OPD, continuous Gaussian policies with `action_mean` plus
`action_log_std` use sampled reverse KL; deterministic action tensors use MSE
unless `STRICT_ACTION_OPD=1` is set. Plain `logits` remain the fallback for
language-only VLA wrappers. Local VLA datasets preserve numeric `action` fields
as `actions` tensors and `state`/`proprio` fields as `proprio` tensors in the
batch.

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

Common dataset aliases are built in:

- LLM: `mmlu`, `mmlu_pro`, `hellaswag`, `arc_challenge`, `arc_easy`, `gsm8k`,
  `boolq`, `winogrande`, `truthfulqa`.
- VLM: `vqav2`, `okvqa`, `gqa`, `textvqa`, `scienceqa`, `vizwiz`,
  `coco_caption`.
- VLA: `libero`, `lerobot_libero`, `lerobot_pusht`, `aloha_transfer_cube`,
  `aloha_insertion`, plus dataset-family aliases `bridge_v2`, `rt1`,
  `open_x_embodiment`, and `droid` that require `--dataset-name` or
  `--dataset-path` because public hosting varies.

Generic HuggingFace or local JSON/JSONL/CSV/Parquet data is also supported:

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

For local VLA data, include fields such as `instruction`, `image_path`, and
`action`, then pass `--dataset-path data.jsonl --dataset-task vla` and optionally
`--dataset-image-root /path/to/images`. Nested image fields and byte-encoded
images are supported; missing dataset images raise an error by default.
