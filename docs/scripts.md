# Workflow Steps

Run every command from the project root. The workflow scripts are organized by
the paper pipeline stages rather than by a specific dataset.

## Script Layout

| Script | Purpose | Main Output |
| --- | --- | --- |
| `00_common.sh` | Shared environment, cache paths, and helper functions. | Helper only |
| `01_prepare_environment.sh` | Create workspace-local directories and install missing optional packages into `vendor/python`. | Environment check |
| `02_validate_project.sh` | Compile source files, run unit tests, and check CLIs. | Validation log |
| `03_smoke_tiny_nas.sh` | Quick end-to-end smoke run for the tiny NAS pipeline. | `outputs/tiny_nas_quick.log` |
| `04_bld_block_library.sh` | Run BLD and save the trained block library. | `outputs/distill_nas_workflow/04_bld_block_library/` |
| `05_nas_layer_importance.sh` | Score candidates and write per-layer importance. | `outputs/distill_nas_workflow/05_nas_layer_scoring/` |
| `06_mip_topk_configs.sh` | Solve MIP and export the best top-K architecture configs. | `outputs/distill_nas_workflow/06_mip_topk_architecture_configs/` |
| `07_assemble_model_from_config.sh` | Assemble a model from a selected config. | `outputs/distill_nas_workflow/07_model_assembly/` |
| `08_gkd_distill.sh` | Run GKD, optionally with OPD, on the assembled model. | `outputs/distill_nas_workflow/08_global_knowledge_distillation/` |
| `09_evaluate_artifact.sh` | Evaluate the distilled artifact. | `outputs/distill_nas_workflow/09_evaluation/` |
| `10_profile_artifact.sh` | Measure artifact latency/throughput. | `outputs/distill_nas_workflow/10_profiling/` |
| `11_export_and_report.sh` | Export artifact manifest/files and generate a report. | `outputs/distill_nas_workflow/11_export/`, `report.md` |
| `12_multi_objective_search.sh` | Run weight sweep, Pareto extraction, and visualization. | `outputs/distill_nas_workflow/06_mip_topk_architecture_configs/multi_objective_*`, `pareto_front.svg` |
| `run_all.sh` | Runs the default workflow, with multi-objective search before the final report. | Full staged smoke workflow |
| `tools/run_experiment.py` | Config-driven workflow runner with resume/cache. | Same workflow outputs |

## Default Workflow

```bash
bash scripts/run_all.sh
```

Equivalent manual sequence:

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

## Stage Files

The staged workflow writes artifacts under clearly named stage directories:

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
    configs/
      config_rank_00.json
      config_rank_01.json
    pareto_configs/
      pareto_rank_00.json
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

Use `WORKFLOW_OUTPUT_DIR=outputs/<name>` or `omnidistill run --workdir <dir>` to move the workflow tree.
`STAGE_DIR=...` remains accepted as a backward-compatible alias.

## Config-Driven Runs

`configs/toy_experiment.json` describes the default toy workflow plus
evaluation, profiling, export, and report generation:

```bash
omnidistill run --config configs/toy_experiment.json
omnidistill run --config configs/toy_experiment.json --dry-run
omnidistill run --config configs/toy_experiment.json --from-stage evaluate
omnidistill status --workflow-dir outputs/distill_nas_workflow
```

Artifact-producing stages are skipped when their outputs already exist unless
`--force` is set. This gives the workflow a simple resume/cache behavior.

## Backends

The scripts support two backends:

- `WORKFLOW_BACKEND=toy` runs the tiny reference model for fast local checks.
- `WORKFLOW_BACKEND=qwen` runs the real Qwen-style LLM/VLM/VLA pipeline with
  `tools/run_staged_model_pipeline.py`.

Real Qwen smoke run:

```bash
WORKFLOW_BACKEND=qwen \
MODEL_ID=Qwen/Qwen3-0.6B \
DEVICE=gpu \
MODEL_VARIANTS=parent,skip_attn,skip_mlp,skip_both,all_core_attn \
MAX_LAYERS=2 \
MAX_PROMPTS=2 \
bash scripts/run_all.sh
```

VLM/VLA runs use the same stages. Set `MODEL_KIND=vlm` or `MODEL_KIND=vla`,
pass a matching `MODEL_ID`, and use `PROMPT_SOURCE`, `DATASET_*`, `IMAGE_PATH`,
or `ALLOW_BLANK_IMAGE=1` as needed.

On Linux GPU hosts, `01_prepare_environment.sh` prepares
`vendor/flash-linear-attention` automatically unless `INSTALL_FLA=0` is set.
Use `REQUIRE_FLA=1` during validation when FLA candidates must be importable.

## Common Overrides

BLD candidate set and training budget:

```bash
BLD_STEPS=4 \
ATTENTION_VARIANTS=parent_attn,mha_attn,quant_mha_attn,mqa_attn,gqa_kv2,linear_attn,noop_attn \
LAYER_VARIANTS=parent,skip_attn,skip_mlp,skip_both \
bash scripts/04_bld_block_library.sh
```

MIP top-K search:

```bash
TOP_K=5 MEMORY_FRACTION=0.80 RUNTIME_FRACTION=0.80 \
bash scripts/06_mip_topk_configs.sh
```

Weighted multi-objective MIP search:

```bash
OBJECTIVE_MODE=weighted SCORE_WEIGHT=1.0 MEMORY_WEIGHT=0.25 RUNTIME_WEIGHT=0.25 \
bash scripts/06_mip_topk_configs.sh
```

The weighted mode keeps the existing hard memory/runtime caps, but changes the
MILP objective from pure candidate score to a normalized weighted sum of score,
memory, and runtime. Set `NO_NORMALIZE_OBJECTIVES=1` only when the metrics are
already on comparable scales.

Pareto frontier sweep:

```bash
GRID_RESOLUTION=5 PARETO_MODE=auto bash scripts/12_multi_objective_search.sh
```

Set `WEIGHT_GRID='1,0,0;1,0.25,0.25;0,1,1'` for explicit sweep points.

Assemble a different rank:

```bash
CONFIG_RANK=1 bash scripts/07_assemble_model_from_config.sh
```

Run GKD with OPD:

```bash
GKD_STEPS=20 OPD_WEIGHT=0.25 OPD_MAX_NEW_TOKENS=8 \
bash scripts/08_gkd_distill.sh
```

For larger real models, split teacher and student placement:

```bash
WORKFLOW_BACKEND=qwen TEACHER_DEVICE=cpu STUDENT_DEVICE=gpu \
bash scripts/08_gkd_distill.sh
```

By default, assembly and GKD save delta `.pth` artifacts: model ID, selected
architecture config, and replacement-layer weights. Checkpoint restore is strict
by default; set `ALLOW_PARTIAL_CHECKPOINT_LOAD=1` only for intentional partial
loads. Set `SAVE_FULL_STATE_DICT=1` only when you intentionally want a full
model state-dict checkpoint. For VLA continuous-action OPD, set
`STRICT_ACTION_OPD=1` to require action logits or `action_mean` plus
`action_log_std` rather than falling back to deterministic action MSE.
