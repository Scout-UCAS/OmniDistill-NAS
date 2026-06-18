# OmniDistill-NAS

<p align="center">
  <strong>Distillation-guided neural architecture search for efficient LLM, VLM, and VLA inference.</strong>
</p>

<p align="center">
  <a href="https://github.com/Scout-UCAS/OmniDistill-NAS/actions/workflows/ci.yml">
    <img src="https://github.com/Scout-UCAS/OmniDistill-NAS/actions/workflows/ci.yml/badge.svg" alt="CI">
  </a>
  <a href="https://www.python.org/">
    <img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Python">
  </a>
  <a href="https://pytorch.org/">
    <img src="https://img.shields.io/badge/PyTorch-supported-ee4c2c.svg" alt="PyTorch">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License">
  </a>
</p>

<p align="center">
  <a href="docs/README.md">Documentation</a> |
  <a href="docs/README.zh-CN.md">中文文档</a> |
  <a href="docs/tutorials/toy_workflow.md">Toy Tutorial</a> |
  <a href="docs/tutorials/qwen3_text.md">Qwen Tutorial</a> |
  <a href="docs/tutorials/vlm_vla.md">VLM/VLA Tutorial</a> |
  <a href="distillation_nas_paper.pdf">Report PDF</a>
</p>

OmniDistill-NAS is an end-to-end research framework for turning large
Transformer teachers into smaller, faster, and more deployable student models.
It brings together block-level distillation, replace-one-layer scoring,
constrained mixed-integer architecture search, multi-objective Pareto
exploration, global knowledge distillation, artifact evaluation, profiling,
export, benchmark manifests, and result tracking in one reproducible workflow.

The project is designed to be useful at two very different scales:

- **Local and CI scale**: a tiny packaged workflow validates the complete
  platform without downloading model weights.
- **GPU research scale**: Qwen text, Qwen-VL, and OpenVLA-style smoke suites
  exercise real decoder-layer replacement paths on LLM, VLM, and VLA models.

<p align="center">
  <img src="docs/assets/omnidistill-nas-overview.svg" alt="OmniDistill-NAS system overview" width="92%">
</p>

```bash
git clone git@github.com:Scout-UCAS/OmniDistill-NAS.git
cd OmniDistill-NAS
python3 -m pip install -e ".[dev,docs]"

omnidistill benchmark --dry-run
omnidistill run --config configs/toy_experiment.json
```

## News And Highlights

- **2026-06-13**: `v0.1.0` research release metadata added in
  [CITATION.cff](CITATION.cff).
- **Packaged toy benchmark**: `omnidistill benchmark --dry-run` works from an
  installed package, not only from a source checkout.
- **Config-driven workflow**: JSON/YAML experiment specs can run, resume,
  validate, profile, export, and report full compression workflows.
- **Real model smoke paths**: Qwen3 text, Qwen3-VL, and OpenVLA-style VLA
  workflows cover model loading, multimodal batching, decoder replacement,
  scoring, and search.
- **Multi-objective search**: weighted objectives and Pareto reports help
  compare quality, memory, latency, throughput, and diversity.
- **Extension registries**: attention variants, dataset adapters, quantization
  plans, evaluators, exporters, tracking sinks, and VLA rollout adapters can be
  registered without rewriting the core workflow.

## Table Of Contents

- [Overview](#overview)
- [Why OmniDistill-NAS](#why-omnidistill-nas)
- [What You Get](#what-you-get)
- [Method At A Glance](#method-at-a-glance)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [CLI Reference](#cli-reference)
- [Workflow Stages](#workflow-stages)
- [Experiment Specs](#experiment-specs)
- [Search And Optimization](#search-and-optimization)
- [Distillation Objectives](#distillation-objectives)
- [Attention And Layer Candidates](#attention-and-layer-candidates)
- [Real Model Smoke Suites](#real-model-smoke-suites)
- [Qwen3 Text Example](#qwen3-text-example)
- [Vision-Language Models](#vision-language-models)
- [VLA And Common Datasets](#vla-and-common-datasets)
- [Artifacts And Output Contracts](#artifacts-and-output-contracts)
- [Benchmarking And Result Zoo](#benchmarking-and-result-zoo)
- [Python API](#python-api)
- [Extension Surface](#extension-surface)
- [Repository Layout](#repository-layout)
- [Quality Bar](#quality-bar)
- [Troubleshooting](#troubleshooting)
- [Documentation](#documentation)
- [Project Status](#project-status)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [Citation](#citation)
- [License](#license)

## Overview

OmniDistill-NAS implements the main ideas from
[distillation_nas_paper.pdf](distillation_nas_paper.pdf), which studies
**Distillation-Based NAS for Inference-Optimized LLMs**. The paper frames model
compression as a structured search problem: build many local replacement
blocks, measure how well each replacement preserves teacher behavior, solve for
an architecture under deployment constraints, then uptrain the selected student
with global distillation.

This repository keeps that paper-aligned structure while making the workflow
directly runnable:

1. Build candidate attention and FFN replacements.
2. Initialize alternatives with training-free transformations when possible.
3. Locally distill candidate blocks against the teacher.
4. Score each candidate by replacing one layer at a time.
5. Estimate memory, KV-cache, latency, runtime, and throughput costs.
6. Solve grouped selection problems with MIP or exhaustive fallback.
7. Explore Pareto trade-offs across score, memory, and runtime.
8. Assemble a selected architecture into a student artifact.
9. Refine the student with global knowledge distillation and optional OPD.
10. Evaluate, profile, export, report, and index the result.

The implementation exposes both a tiny internal model and real model wrappers.
The tiny model is fast enough for CI and local sanity checks. The Qwen-style
wrappers let the same platform exercise text-only, vision-language, and
vision-language-action decoder replacement paths.

## Why OmniDistill-NAS

Practical model compression is rarely a single pruning ratio or a single
quantization switch. An inference system has to decide:

- which layers can be changed,
- which attention variants are acceptable,
- which FFN replacements are cheap enough,
- how much memory is available at different batch sizes,
- which latency or throughput budget matters,
- whether the student still matches teacher behavior,
- which artifact can be exported and reproduced later.

OmniDistill-NAS turns that loop into a staged, inspectable experiment. Every
major boundary is represented as a file: JSON configs, benchmark suites, PTH
artifacts, metric reports, profiler outputs, export manifests, result-zoo
manifests, and Markdown summaries. Runs are therefore easier to inspect, resume,
compare across hardware, and share with collaborators.

## What You Get

OmniDistill-NAS is not just a search script. It is a structured compression
platform with auditable inputs and outputs.

| Capability | What it provides |
| --- | --- |
| Search spaces | Attention variants, FFN replacements, layer skips, quantized candidates, and FLA-style alternatives |
| Distillation-aware scoring | Teacher-student logits, hidden states, action outputs, OPD-style targets, and replacement-local probes |
| Constrained architecture selection | MIP objectives over quality, memory, runtime, throughput, latency, batch size, and diversity |
| Multi-objective exploration | Pareto configs, Markdown reports, SVG frontier plots, and weighted objective sweeps |
| End-to-end workflow | Preparation, validation, smoke testing, library building, scoring, search, assembly, distillation, evaluation, profiling, export, and reporting |
| Real model smoke paths | Qwen3 text, Qwen3-VL, and OpenVLA-style action prediction |
| Packaged assets | Default benchmark, configs, schemas, and toy result manifests are usable outside a source checkout |
| Extension registries | Attention, dataset, quantization, device, evaluator, exporter, tracker, plugin, and VLA rollout extension points |

The project is intentionally small at the core. The default toy workflow gives a
deterministic way to verify the machinery, while optional real-model tooling
lets researchers plug in larger teachers, datasets, and hardware-specific
profilers.

## Method At A Glance

The workflow follows three paper-level phases and then adds reproducibility
infrastructure around them.

| Paper phase | Repository components | Description |
| --- | --- | --- |
| Crafting puzzle pieces | `distill_nas_core.library`, `distill_nas_core.search_space`, `distill_nas_core.distill.local_distill_block` | Build and locally distill candidate replacement blocks for each teacher layer |
| Assembling puzzle architecture | `distill_nas_core.scoring`, `distill_nas_core.resources`, `distill_nas_core.mip`, `distill_nas_core.multi_objective` | Score candidates, estimate costs, and solve constrained per-layer selection |
| Uptraining | `distill_nas_core.distill.global_knowledge_distillation` | Refine the assembled student with GKD and optional on-policy distillation |
| Evaluation and release | `evaluation`, `profiler`, `export`, `reporting`, `result_zoo`, `benchmarks` | Measure, package, report, validate, and compare produced artifacts |

The default stage sequence is:

```text
prepare -> validate -> smoke -> bld -> score -> mip -> multi_objective
        -> assemble -> gkd -> evaluate -> profile -> export -> report
```

You can run the entire sequence, resume from a later stage, or execute one
stage in isolation.

## Installation

OmniDistill-NAS targets Python 3.10+ and PyTorch 2.x.

Minimal install:

```bash
python3 -m pip install -e .
```

Development, docs, and tests:

```bash
python3 -m pip install -e ".[dev,docs]"
```

Real-model tooling:

```bash
python3 -m pip install -e ".[tools]"
```

Conda-style environment:

```bash
conda env create -f environment.yml
conda activate omnidistill-nas
python3 -m pip install -e ".[dev,docs]"
```

The optional `tools` extra includes packages used by real-model smoke suites,
including `accelerate`, `datasets`, `transformers`, `safetensors`,
`sentencepiece`, `einops`, `pillow`, and `timm`.

The OpenVLA path currently needs a `timm` version compatible with its remote
model code, so the project pins the optional tool range to:

```text
timm>=0.9.10,<1
```

Generated runtime files are intentionally ignored by Git:

```text
outputs/       workflow artifacts, reports, profiles, exports
checkpoints/   generated model checkpoints
hf_cache/      Hugging Face model and dataset caches
vendor/        optional workspace-local dependencies
```

## Quick Start

### 1. Validate The Packaged Benchmark

The default benchmark suite is bundled with the Python package, so this command
works even when OmniDistill-NAS is installed outside the source tree:

```bash
omnidistill benchmark --dry-run
```

To see the resolved benchmark plan:

```bash
omnidistill benchmark --print-plan
```

### 2. Run A Tiny NAS Demo

```bash
python3 tools/run_tiny_nas.py --quick
```

The example automatically uses CUDA or MPS when available. Expected output
includes the device, number of generated block candidates, selected batch size,
total score, memory estimate, runtime proxy, throughput estimate, and final
candidate choice per layer.

Example output shape:

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

### 3. Run The Full Toy Workflow

```bash
omnidistill run --config configs/toy_experiment.json
```

Inspect the run:

```bash
omnidistill status --workflow-dir outputs/distill_nas_workflow
omnidistill report --workflow-dir outputs/distill_nas_workflow
sed -n '1,120p' outputs/distill_nas_workflow/report.md
```

Resume from a later stage:

```bash
omnidistill run --config configs/toy_experiment.json --from-stage evaluate
```

Run a single stage:

```bash
omnidistill run --config configs/toy_experiment.json --stage profile
```

Force stage execution even when outputs already exist:

```bash
omnidistill run --config configs/toy_experiment.json --force
```

### 4. Create Your Own Experiment

```bash
omnidistill init --output configs/my_experiment.json
omnidistill validate configs/my_experiment.json
omnidistill plan --config configs/my_experiment.json
```

JSON and YAML experiment specs are both supported. See
[schemas/experiment.schema.json](schemas/experiment.schema.json) for the public
schema.

### 5. Run The Shell Workflow Directly

The CLI wraps the same stage scripts used by the shell workflow:

```bash
bash scripts/run_all.sh
```

For every stage-level environment variable, see
[docs/scripts.md](docs/scripts.md).

## CLI Reference

The package exposes two console entry points:

```text
omnidistill
omnidistill-nas
```

Both commands resolve to `distill_nas_core.cli:main`.

Main commands:

```text
omnidistill init        Create a starter experiment config
omnidistill run         Run a staged experiment with resume support
omnidistill plan        Print the resolved stage plan
omnidistill validate    Validate experiment, benchmark, or result files
omnidistill benchmark   Run or dry-run a benchmark suite
omnidistill report      Generate workflow or result-zoo reports
omnidistill plugins     List registered extension plugins
omnidistill status      Show workflow artifact status
```

Common examples:

```bash
omnidistill validate configs/toy_experiment.json
omnidistill validate benchmarks/suites/toy_smoke.json --kind benchmark
omnidistill validate results/toy_smoke/manifest.json --kind result

omnidistill benchmark --suite benchmarks/suites/toy_smoke.json --dry-run
omnidistill benchmark --suite benchmarks/suites/qwen3_text_smoke.json --dry-run
omnidistill report --results-dir results
omnidistill plugins
omnidistill status --workflow-dir outputs/distill_nas_workflow --json
```

Use `--workdir /path/to/workspace` when you want outputs, caches, and optional
local dependencies outside the source checkout:

```bash
omnidistill run \
  --config configs/toy_experiment.json \
  --workdir /path/to/omnidistill-runs/toy
```

## Workflow Stages

The default workflow is intentionally split into small, inspectable stages.

| Stage | Script | Purpose | Main output |
| --- | --- | --- | --- |
| Prepare | `scripts/01_prepare_environment.sh` | Create runtime directories, check optional dependencies, configure cache paths | workspace dirs and environment report |
| Validate | `scripts/02_validate_project.sh` | Compile modules, run tests, validate CLI entry points | validation log |
| Smoke | `scripts/03_smoke_tiny_nas.sh` | Run a fast end-to-end NAS sanity check | quick NAS log |
| BLD | `scripts/04_bld_block_library.sh` | Train or initialize candidate replacement blocks | `block_library.pth`, `summary.json` |
| Score | `scripts/05_nas_layer_importance.sh` | Score replacement candidates per layer | `layer_importance.json` |
| MIP | `scripts/06_mip_topk_configs.sh` | Solve constrained grouped selection problems | `topk_architecture_configs.json` |
| Pareto | `scripts/12_multi_objective_search.sh` | Sweep multi-objective trade-offs | Pareto JSON, configs, Markdown, SVG |
| Assemble | `scripts/07_assemble_model_from_config.sh` | Build the selected student | `assembled_model.pth` |
| Distill | `scripts/08_gkd_distill.sh` | Refine the student with GKD/OPD | `gkd_model.pth`, loss summary |
| Evaluate | `scripts/09_evaluate_artifact.sh` | Measure quality and task metrics | `metrics.json` |
| Profile | `scripts/10_profile_artifact.sh` | Measure runtime, throughput, memory | `profile.json` |
| Export | `scripts/11_export_and_report.sh` | Export portable artifact and final report | export manifest, `report.md` |

Run stages manually:

```bash
bash scripts/01_prepare_environment.sh
bash scripts/02_validate_project.sh
bash scripts/03_smoke_tiny_nas.sh
bash scripts/04_bld_block_library.sh
bash scripts/05_nas_layer_importance.sh
bash scripts/06_mip_topk_configs.sh
bash scripts/12_multi_objective_search.sh
bash scripts/07_assemble_model_from_config.sh
bash scripts/08_gkd_distill.sh
bash scripts/09_evaluate_artifact.sh
bash scripts/10_profile_artifact.sh
bash scripts/11_export_and_report.sh
```

The shell workflow and CLI share the same artifact layout, so a run can be
inspected with the same `status`, `report`, and validation commands.

## Experiment Specs

An experiment spec controls backend, stages, model options, search settings,
distillation settings, devices, distributed options, and environment overrides.

Minimal example:

```json
{
  "name": "toy-platform-smoke",
  "backend": "toy",
  "output_dir": "outputs/distill_nas_workflow",
  "stages": ["prepare", "validate", "bld", "score", "mip", "assemble", "gkd", "evaluate"],
  "model": {
    "seq_len": 16,
    "batch_size": 2,
    "num_batches": 4
  },
  "search": {
    "attention_variants": "parent_attn,mha_attn,quant_mha_attn,mqa_attn,linear_attn,noop_attn",
    "batch_sizes": "1,2,4",
    "top_k": 3
  },
  "distillation": {
    "gkd_steps": 2
  },
  "devices": {
    "device": "auto"
  }
}
```

The default packaged spec is [configs/toy_experiment.json](configs/toy_experiment.json).
It runs:

```text
prepare, validate, smoke, bld, score, mip, multi_objective,
assemble, gkd, evaluate, profile, export, report
```

Validate before running:

```bash
omnidistill validate configs/toy_experiment.json
omnidistill plan --config configs/toy_experiment.json
```

Write run results to a machine-readable JSON file:

```bash
omnidistill run \
  --config configs/toy_experiment.json \
  --results-json outputs/toy_run_results.json
```

`run_experiment.py` skips completed artifact-producing stages unless `--force`
is set. This makes iterative development fast while keeping the workflow
restartable.

## Search And Optimization

The solver works over per-layer candidate groups. Each candidate carries a
quality score, parameter memory estimate, KV-cache memory estimate, runtime
proxy, and optional payload. The MIP selects one candidate per layer while
satisfying constraints.

Single-objective score minimization:

```bash
OBJECTIVE_MODE=score bash scripts/06_mip_topk_configs.sh
```

Weighted score-memory-runtime objective:

```bash
OBJECTIVE_MODE=weighted \
SCORE_WEIGHT=1.0 \
MEMORY_WEIGHT=0.25 \
RUNTIME_WEIGHT=0.25 \
bash scripts/06_mip_topk_configs.sh
```

The same flags are available in `tools/run_staged_toy_pipeline.py mip` and
`tools/run_staged_model_pipeline.py mip`:

```bash
python3 tools/run_staged_toy_pipeline.py mip \
  --objective-mode weighted \
  --score-weight 1.0 \
  --memory-weight 0.25 \
  --runtime-weight 0.25
```

Objectives are normalized by default so score, bytes, and seconds can be mixed
without one unit dominating by scale.

Batch-specific memory and latency constraints are supported:

```json
{
  "search": {
    "batch_sizes": "1,2,4",
    "memory_max_by_batch": {"1": 8000000000, "2": 10000000000},
    "latency_max_by_batch": {"1": 0.05, "2": 0.08},
    "throughput_min": 1024
  }
}
```

Generate a Pareto frontier:

```bash
GRID_RESOLUTION=5 PARETO_MODE=auto bash scripts/12_multi_objective_search.sh
```

Or run the Python utility directly:

```bash
python3 tools/run_multi_objective_search.py \
  --scores-json outputs/distill_nas_workflow/05_nas_layer_scoring/layer_importance.json
```

This writes:

```text
outputs/distill_nas_workflow/06_mip_topk_architecture_configs/multi_objective_search.json
outputs/distill_nas_workflow/06_mip_topk_architecture_configs/multi_objective_report.md
outputs/distill_nas_workflow/06_mip_topk_architecture_configs/pareto_front.svg
```

For small search spaces the tool enumerates the exact feasible Pareto frontier.
For larger spaces it uses weighted sweep solutions as an approximate frontier.

## Distillation Objectives

OmniDistill-NAS uses distillation at three levels.

### Blockwise Local Distillation

BLD trains or calibrates candidate replacements against teacher block outputs.
The toy implementation supports coupled and decoupled block-library builders.
The real-model path uses model-specific wrappers to expose decoder layers and
candidate replacement modules.

Relevant modules:

- `distill_nas_core.library`
- `distill_nas_core.search_space`
- `distill_nas_core.distill.local_distill_block`

### Replace-One-Layer Scoring

Candidate scoring measures how much behavior changes when a single teacher
layer is replaced. Supported scoring signals include logits KL, LM loss, hidden
state differences, and action output differences for VLA-style models.

Relevant modules:

- `distill_nas_core.scoring`
- `distill_nas_core.resources`
- `tools/run_qwen3_attention_search.py`

### Global Knowledge Distillation

`global_knowledge_distillation` keeps the original offline GKD objective by
default. The main loss combines hidden cosine loss and logits KL loss on the
same prompt batches.

```python
from distill_nas_core.distill import global_knowledge_distillation

losses = global_knowledge_distillation(
    teacher,
    student,
    prompt_batches,
    steps=100,
    lr=1e-4,
)
```

### GKD With Optional OPD

To add token OPD, pass a positive `opd_weight` and `opd_max_new_tokens`; each
batch must provide `input_ids`.

The student samples continuations from its current policy. The teacher then
scores those sampled tokens, and the extra loss is the sampled reverse-KL term:

```text
log p_student(token) - log p_teacher(token)
```

Example:

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

VLA action-space OPD is also supported. Action logits use sampled reverse KL,
and continuous action predictions use action MSE as an action-space
distillation approximation. Continuous-action support does not perform an
environment rollout.

## Attention And Layer Candidates

The toy distillation NAS pipeline includes layer-level candidates:

```text
parent
skip_attn
skip_mlp
skip_both
```

Attention candidates include:

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
```

FLA-named candidates include:

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

The generic toy pipeline uses lightweight PyTorch equivalents for the FLA names,
so they forward, distill, score, and enter MIP without requiring the full FLA
runtime.

Useful variant aliases:

| Alias | Expands to |
| --- | --- |
| `all_qwen_attn` | Parent, MHA, quantized MHA, MQA, GQA, MFA, MLA, MKA, linear, and no-op attention |
| `all_linear_attn` | `linear_attn` plus the FLA linear and delta family |
| `all_core_attn` | Linear/FLA candidates at the same level as MHA/MQA/GQA/MFA/MLA/MKA |
| `all_fla` | All registered FLA-named candidates |
| `all_attention` | Broad toy search default used by the demo path |

The toy demo defaults to:

```text
--attention-variants all_attention
--layer-variants parent,skip_attn,skip_mlp,skip_both
```

Use aliases or comma-separated names to narrow the search.

Search-space plugins can register attention candidates at runtime:

```python
from distill_nas_core.search_space import register_attention_variant

register_attention_variant(
    "my_attention",
    spec_factory=my_spec_factory,
    module_factory=my_module_factory,
    aliases=["my_attn"],
)
```

## Real Model Smoke Suites

The toy backend is the default platform check. GPU smoke suites cover real
model-loading, multimodal batching, decoder replacement, scoring, and search.

Dry-run command generation:

```bash
omnidistill benchmark --suite benchmarks/suites/qwen3_text_smoke.json --dry-run
omnidistill benchmark --suite benchmarks/suites/qwen_vlm_smoke.json --dry-run
omnidistill benchmark --suite benchmarks/suites/openvla_smoke.json --dry-run
```

Execute a real smoke benchmark:

```bash
HF_ENDPOINT=https://hf-mirror.com \
HF_HUB_DISABLE_XET=1 \
HF_HOME=/path/to/hf_cache \
omnidistill benchmark \
  --suite benchmarks/suites/qwen3_text_smoke.json \
  --result-dir benchmark_runs/qwen3 \
  --allow-commands
```

Available source checkout suites:

| Suite | Model path | What it exercises |
| --- | --- | --- |
| `toy_smoke.json` | Tiny internal model | Full staged platform workflow |
| `qwen3_text_smoke.json` | `Qwen/Qwen3-0.6B` | Text decoder candidate scoring |
| `qwen_vlm_smoke.json` | `Qwen/Qwen3-VL-2B-Instruct` | Vision-language prompt encoding and decoder replacement |
| `openvla_smoke.json` | `openvla/openvla-7b` | VLA image/action path and OpenVLA-compatible loading |

External-command benchmark suites are dry-run safe by default. They execute only
when `--allow-commands` is provided.

### Runtime Validation Snapshot

The following smoke paths were exercised on a CUDA host with an NVIDIA A800 80GB
GPU during the current repository validation pass:

| Path | Status | Approx runtime |
| --- | --- | --- |
| Full toy workflow | completed | 43.3 seconds |
| Qwen3 text smoke | completed | 13.8 seconds |
| Qwen-VL smoke | completed | 14.6 seconds |
| OpenVLA smoke | completed | 24.5 seconds |

These numbers are validation evidence, not universal performance claims. Actual
runtime depends on cache state, model access, GPU type, driver stack, and
dataset availability.

## Qwen3 Text Example

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

Stages 04-08 write:

```text
block_library.pth
layer_importance.json
topk_architecture_configs.json
assembled_model.pth
gkd_model.pth
```

Assembly and GKD save delta checkpoints by default: model ID, selected
architecture config, and replacement layer weights. Checkpoint restore is strict
by default. Set `ALLOW_PARTIAL_CHECKPOINT_LOAD=1` only when intentionally
loading across changed candidate definitions. Set `SAVE_FULL_STATE_DICT=1` only
when you intentionally want a full model state dict.

`tools/run_qwen3_attention_search.py` runs a real-model NAS candidate search on
`Qwen/Qwen3-0.6B`:

```bash
python3 tools/run_qwen3_attention_search.py \
  --model-id Qwen/Qwen3-0.6B \
  --device gpu \
  --max-layers 8 \
  --max-prompts 2 \
  --seq-len 128
```

It computes replace-1-layer KL scores on GPU and solves MIP over real Qwen
attention candidates, skip candidates, and FLA candidates.

Results are written under the active workspace to:

```text
outputs/qwen3_0_6b_layer_skip_search.json
checkpoints/qwen3_0_6b_layer_skip_search.pth
```

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
ids.

Built-in smoke prompts may use a blank RGB image when `--image-path` is omitted,
but dataset prompts require a real image field unless `--allow-blank-image` is
set explicitly. Qwen3-VL checkpoints need a transformers build that recognizes
`qwen3_vl`; `scripts/01_prepare_environment.sh` installs
`transformers>=4.57,<5` into `vendor/python` when the active version is older.

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

For a local image, add:

```bash
--image-path path/to/image.jpg
```

If a smaller `Qwen3-VL-0.6B` checkpoint is available in your environment, pass
that model id with the same command. The search only swaps language decoder
layers; the vision encoder and multimodal projection stay unchanged.

## VLA And Common Datasets

`tools/run_qwen3_attention_search.py` also accepts `--model-kind vla` for
vision-language-action models such as OpenVLA-style policies. VLA support uses
the same decoder-layer search as VLM: the vision encoder and action projection
stay unchanged, while Qwen/Llama-style language decoder layers are swapped and
scored against teacher outputs.

If the model exposes `action_logits`, those are scored with KL. Continuous
`action_mean` outputs are scored with MSE, and `actions` or `predicted_actions`
remain supported as deterministic action values. For GKD/OPD, continuous
Gaussian policies with `action_mean` plus `action_log_std` use sampled reverse
KL; deterministic action tensors use MSE unless `STRICT_ACTION_OPD=1` is set.
Plain `logits` remain the fallback for language-only VLA wrappers.

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

| Task family | Aliases |
| --- | --- |
| LLM | `mmlu`, `mmlu_pro`, `hellaswag`, `arc_challenge`, `arc_easy`, `gsm8k`, `boolq`, `winogrande`, `truthfulqa` |
| VLM | `vqav2`, `okvqa`, `gqa`, `textvqa`, `scienceqa`, `vizwiz`, `coco_caption` |
| VLA | `libero`, `lerobot_libero`, `lerobot_pusht`, `aloha_transfer_cube`, `aloha_insertion`, `bridge_v2`, `rt1`, `open_x_embodiment`, `droid` |

Generic Hugging Face or local JSON/JSONL/CSV/Parquet data is also supported:

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
`action`, then pass:

```bash
--dataset-path data.jsonl --dataset-task vla --dataset-image-root /path/to/images
```

Nested image fields and byte-encoded images are supported. Missing dataset
images raise an error by default.

## Artifacts And Output Contracts

OmniDistill-NAS favors structured artifacts over implicit state.

Typical workflow output:

```text
outputs/distill_nas_workflow/
  04_bld_block_library/
    block_library.pth
    summary.json
  05_nas_layer_scoring/
    layer_importance.json
  06_mip_topk_architecture_configs/
    topk_architecture_configs.json
    multi_objective_search.json
    multi_objective_report.md
    pareto_front.svg
  07_model_assembly/
    assembled_model.pth
    summary.json
  08_global_knowledge_distillation/
    gkd_model.pth
    summary.json
  09_evaluation/
    metrics.json
  10_profiling/
    profile.json
  11_export/
    manifest.json
  report.md
```

Result manifests are validated with:

```bash
omnidistill validate results/toy_smoke/manifest.json --kind result
```

The public result schema lives in
[schemas/result-manifest.schema.json](schemas/result-manifest.schema.json).

Exported artifact manifests record source artifact paths, export format,
metadata, and files needed for downstream use. Reports are written in Markdown
so they can be opened directly in GitHub or converted by downstream tooling.

## Benchmarking And Result Zoo

Benchmark suites are JSON manifests under [benchmarks/suites](benchmarks/suites).

```bash
omnidistill benchmark --suite benchmarks/suites/toy_smoke.json --dry-run
omnidistill benchmark --suite benchmarks/suites/toy_smoke.json --result-dir benchmark_runs/toy
```

Each suite lists reproducible jobs, expected metrics, and either an experiment
config or an explicit command. Heavy model suites are dry-run safe by default;
external commands only run when `--allow-commands` is passed.

Result manifests belong under:

```text
results/<run-id>/manifest.json
```

Generate a result-zoo index:

```bash
omnidistill report --results-dir results --output-md results/index.md
```

See:

- [docs/benchmarking.md](docs/benchmarking.md)
- [docs/result_zoo.md](docs/result_zoo.md)

## Python API

Solve a constrained NAS MIP:

```python
from distill_nas_core.mip import SearchConstraints, solve_nas_mip

solution = solve_nas_mip(
    candidates_by_layer,
    SearchConstraints(
        seq_len=128,
        batch_sizes=[1, 2, 4],
        memory_max_by_batch={1: 8e9, 2: 10e9, 4: 14e9},
        latency_max_by_batch={1: 0.05, 2: 0.08, 4: 0.14},
        objective_mode="weighted",
        score_weight=1.0,
        memory_weight=0.25,
        runtime_weight=0.25,
    ),
)

print(solution.selected_names)
```

Register an attention variant:

```python
from distill_nas_core.search_space import register_attention_variant

register_attention_variant(
    "my_attention",
    spec_factory=my_spec_factory,
    module_factory=my_module_factory,
    aliases=["my_attn"],
)
```

Register an evaluator:

```python
from distill_nas_core.evaluation import register_evaluator

def evaluate_my_backend(path, **kwargs):
    return {"accuracy": 0.0, "artifact": str(path)}

register_evaluator("my-backend", evaluate_my_backend)
```

Register an exporter:

```python
from distill_nas_core.export import register_exporter

def export_my_format(source_path, export_dir, **kwargs):
    return {"format": "my-format", "files": [str(source_path)], "export_dir": str(export_dir)}

register_exporter("my-format", export_my_format)
```

Register a tracking sink:

```python
from distill_nas_core.tracking import register_tracking_provider

def tracking_provider(event, payload):
    print(event, payload or {})
    return {"status": "ok"}

register_tracking_provider("stdout", tracking_provider)
```

## Extension Surface

OmniDistill-NAS is intentionally small at the core and open at the boundaries.

| Extension point | Module | Typical use |
| --- | --- | --- |
| Attention variants | `distill_nas_core.search_space` | Add a custom attention module or alias group |
| Dataset adapters | `distill_nas_core.data_adapters` | Load prompts, images, actions, or task labels |
| Quantization plans | `distill_nas_core.quantization` | Compare dense and quantized block decisions |
| Device policies | `distill_nas_core.distributed` | Route teacher/student devices and accumulation |
| Evaluators | `distill_nas_core.evaluation` | Add backend-specific metric computation |
| Exporters | `distill_nas_core.export` | Write custom model packaging formats |
| Tracking providers | `distill_nas_core.tracking` | Emit events to JSONL or external services |
| VLA rollout adapters | `distill_nas_core.vla` | Connect action policies to environments |
| Plugin manifests | `distill_nas_core.plugins` | Activate entry points from structured manifests |

List registered plugins:

```bash
omnidistill plugins
omnidistill plugins --category attention
```

See [docs/plugins.md](docs/plugins.md).

## Repository Layout

Core package:

```text
distill_nas_core/
  __init__.py
  artifacts.py          workflow artifact contracts
  benchmarks.py         benchmark suite runner
  blocks.py             candidate attention and FFN blocks
  cli.py                command line interface
  data_adapters.py      dataset and prompt loading extension points
  distill.py            BLD, GKD, and OPD training utilities
  distributed.py        device and distributed helpers
  evaluation.py         artifact evaluation
  experiment.py         config-driven workflow planner and runner
  export.py             portable artifact export
  library.py            block-library construction
  mip.py                constrained NAS solver
  multi_objective.py    Pareto and weighted objective search
  package_assets.py     packaged benchmark/config/schema helpers
  plugins.py            extension registry
  profiler.py           latency, throughput, memory profiling
  quantization.py       quantization plan helpers
  reporting.py          workflow and result reports
  resources.py          memory/runtime estimates
  result_zoo.py         result manifest indexing
  schema.py             public config/result validation
  scoring.py            layer replacement scoring utilities
  search_space.py       variant registry and search-space construction
  toy.py                tiny causal language model
  toy_assembly.py       toy model assembly helpers
  tracking.py           event tracking providers
  vla.py                VLA adapters and action helpers
```

Project directories:

```text
scripts/       shell workflow stages
tools/         Python entry-point utilities
configs/       experiment specs
benchmarks/    benchmark suite manifests
results/       reproducible result manifests
schemas/       public JSON schema files
docs/          English and Chinese documentation
notebooks/     notebook and Colab demos
website/       static project homepage
tests/         unit and integration tests
```

Important top-level files:

```text
README.md                  project entry point
pyproject.toml             package metadata and dependency groups
environment.yml            conda-compatible environment
Dockerfile                 container build entry point
distillation_nas_paper.pdf project report
CITATION.cff               citation metadata
ROADMAP.md                 planned work
CONTRIBUTING.md            contribution guide
SECURITY.md                security policy
LICENSE                    MIT license
```

## Quality Bar

Recommended local checks:

```bash
python3 -m pytest -q
python3 -m ruff check distill_nas_core tools tests scripts
python3 -m mypy distill_nas_core tools tests
python3 -m build
mkdocs build --strict
```

Project-level check:

```bash
python3 tools/check_project.py
python3 tools/check_project.py --workflow
```

Lower-level verification:

```bash
python3 -m compileall distill_nas_core scripts tools tests
python3 -m unittest discover -s tests
```

If you do not want to generate `__pycache__` files during a quick local check:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s tests
PYTHONDONTWRITEBYTECODE=1 python3 tools/run_tiny_nas.py --quick
```

The current suite covers:

- MIP search and weighted objectives
- Pareto extraction and report generation
- artifact evaluation, profiling, export, and reports
- packaged benchmark assets outside the source tree
- YAML and JSON schema validation
- Qwen text, VLM, and VLA helper behavior
- plugin and extension registries
- toy staged workflow planning and execution

## Troubleshooting

### `python3` is not on the server PATH

Benchmark command entries that start with `python` or `python3` are resolved to
the active interpreter at runtime. You can still run explicitly with:

```bash
/path/to/python -m distill_nas_core.cli benchmark --suite benchmarks/suites/toy_smoke.json
```

### Hugging Face downloads time out

Use your preferred mirror or cache directory:

```bash
HF_ENDPOINT=https://hf-mirror.com \
HF_HUB_DISABLE_XET=1 \
HF_HOME=/path/to/hf_cache \
omnidistill benchmark --suite benchmarks/suites/qwen3_text_smoke.json --allow-commands
```

### OpenVLA model loading fails with SDPA or TIMM errors

The loader falls back to eager attention for known OpenVLA SDPA gaps. Make sure
the optional dependencies use the project-supported TIMM range:

```bash
python3 -m pip install -e ".[tools]"
python3 - <<'PY'
import timm
print(timm.__version__)
PY
```

Expected range:

```text
>=0.9.10,<1
```

### FLA candidates are unavailable

`scripts/01_prepare_environment.sh` clones `fla-org/flash-linear-attention`
under the workspace-local `vendor/flash-linear-attention` by default on Linux
GPU hosts. Set:

```bash
INSTALL_FLA=0
```

to skip this, or:

```bash
REQUIRE_FLA=1
```

during validation to fail if `fla.layers` is unavailable. Missing FLA classes or
CUDA dependencies are skipped by default during search; pass
`--no-skip-unavailable-fla` to fail instead.

### Generated files are large

Checkpoints and model caches can be large. They are ignored by Git and should be
stored as run artifacts rather than committed:

```text
outputs/
checkpoints/
hf_cache/
vendor/
```

### A workflow stage appears skipped

Artifact-producing stages are skipped when their expected outputs already
exist. Use `--force` with the CLI, or remove the relevant output directory when
you intentionally want to recompute a stage:

```bash
omnidistill run --config configs/toy_experiment.json --force
```

### A result manifest does not validate

Validate the file directly and inspect the schema:

```bash
omnidistill validate results/toy_smoke/manifest.json --kind result
sed -n '1,200p' schemas/result-manifest.schema.json
```

## Documentation

- [Full documentation](docs/README.md)
- [中文文档](docs/README.zh-CN.md)
- [Workflow scripts](docs/scripts.md)
- [Benchmarking](docs/benchmarking.md)
- [Result zoo](docs/result_zoo.md)
- [Tracking](docs/tracking.md)
- [Plugin guide](docs/plugins.md)
- [API reference](docs/api.md)
- [Design notes](docs/design.md)
- [Toy workflow tutorial](docs/tutorials/toy_workflow.md)
- [Qwen text tutorial](docs/tutorials/qwen3_text.md)
- [VLM and VLA tutorial](docs/tutorials/vlm_vla.md)
- [Qwen3 run records](docs/run_records/QWEN3_0_6B_RUN_RECORD.md)
- [Qwen3 FLA run record](docs/run_records/QWEN3_0_6B_FLA_RUN_RECORD.md)
- [Qwen3 linear-attention run record](docs/run_records/QWEN3_0_6B_LINEAR_ATTENTION_LEVEL_RUN_RECORD.md)
- [Qwen3 MMLU smoke run record](docs/run_records/QWEN3_0_6B_MMLU_SMOKE_RUN_RECORD.md)

## Project Status

OmniDistill-NAS is research software. The toy workflow is intended to be
CI-friendly and fast on local machines. Real-model workflows depend on model
availability, device memory, optional FLA support, dataset access, cache state,
and hardware profiling conditions.

The project is alpha but end-to-end:

- APIs are usable.
- Tests and schema validation are in place.
- Toy, Qwen text, Qwen-VL, and OpenVLA smoke paths have been exercised.
- Result manifests and benchmark suites are structured and versionable.
- CLI, shell scripts, Python utilities, docs, and packaged assets are aligned
  around the same workflow contracts.

Large-model result manifests should be treated as reproducibility targets, not
universal performance claims.

## Roadmap

See [ROADMAP.md](ROADMAP.md) for the living roadmap.

Near term:

- Keep the toy workflow fully CI-covered.
- Add and maintain benchmark manifests for Qwen text, VLM, and VLA runs.
- Improve schema validation and diagnostics.
- Expand result manifests with reproducible hardware and dependency metadata.

Mid term:

- Add first-class W&B and MLflow exporters.
- Grow plugin examples for datasets, objectives, and evaluators.
- Publish a hosted documentation site.
- Add richer Pareto dashboards and benchmark comparisons.

Long term:

- Maintain a curated model/result zoo.
- Support distributed benchmark campaigns.
- Publish paper-aligned reproducibility bundles.

## Contributing

Contributions are welcome across search spaces, model adapters, dataset
formatters, benchmark manifests, documentation, CI, and real-model run records.

Start with:

- [CONTRIBUTING.md](CONTRIBUTING.md)
- [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)
- [SECURITY.md](SECURITY.md)

When contributing benchmark or performance changes, include a result manifest or
the exact command needed to reproduce the claim.

Useful contribution targets:

- add an attention or FFN replacement candidate,
- add a dataset adapter for an LLM, VLM, or VLA benchmark,
- improve a real-model smoke suite,
- add an evaluator or exporter,
- expand result manifests and run records,
- improve docs for a hardware or dependency setup.

## Citation

If OmniDistill-NAS helps your research, please cite the repository using
[CITATION.cff](CITATION.cff).

```yaml
cff-version: 1.2.0
title: "OmniDistill-NAS"
authors:
  - name: "OmniDistill-NAS contributors"
date-released: "2026-06-13"
version: "0.1.0"
license: "MIT"
repository-code: "https://github.com/Scout-UCAS/OmniDistill-NAS"
```

A formal paper citation can be added when the accompanying report is released.

## License

OmniDistill-NAS is released under the [MIT License](LICENSE).
