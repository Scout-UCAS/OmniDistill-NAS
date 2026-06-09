# Workflow Steps

Run every command from the project root. The workflow scripts are organized by
the paper pipeline stages rather than by a specific dataset.

## Script Layout

| Script | Purpose | Main Output |
| --- | --- | --- |
| `00_common.sh` | Shared environment, cache paths, and helper functions. | Helper only |
| `01_prepare_environment.sh` | Create local directories and install missing optional packages into `vendor/python`. | Environment check |
| `02_validate_project.sh` | Compile source files, run unit tests, and check CLIs. | Validation log |
| `03_smoke_tiny_nas.sh` | Quick end-to-end smoke run for the tiny NAS pipeline. | `outputs/tiny_nas_quick.log` |
| `04_bld_block_library.sh` | Run BLD and save the trained block library. | `outputs/distill_nas_workflow/04_bld_block_library/` |
| `05_nas_layer_importance.sh` | Score candidates and write per-layer importance. | `outputs/distill_nas_workflow/05_nas_layer_scoring/` |
| `06_mip_topk_configs.sh` | Solve MIP and export the best top-K architecture configs. | `outputs/distill_nas_workflow/06_mip_topk_architecture_configs/` |
| `07_assemble_model_from_config.sh` | Assemble a model from a selected config. | `outputs/distill_nas_workflow/07_model_assembly/` |
| `08_gkd_distill.sh` | Run GKD, optionally with OPD, on the assembled model. | `outputs/distill_nas_workflow/08_global_knowledge_distillation/` |
| `run_all.sh` | Runs steps 01-08 in order. | Full staged smoke workflow |

## Default Workflow

```bash
bash workflow_steps/run_all.sh
```

Equivalent manual sequence:

```bash
bash workflow_steps/01_prepare_environment.sh
bash workflow_steps/02_validate_project.sh
bash workflow_steps/03_smoke_tiny_nas.sh
bash workflow_steps/04_bld_block_library.sh
bash workflow_steps/05_nas_layer_importance.sh
bash workflow_steps/06_mip_topk_configs.sh
bash workflow_steps/07_assemble_model_from_config.sh
bash workflow_steps/08_gkd_distill.sh
```

## Stage Files

The staged reference workflow writes artifacts under clearly named stage
directories:

```text
outputs/distill_nas_workflow/
  04_bld_block_library/
    block_library.pth
    summary.json
  05_nas_layer_scoring/
    layer_importance.json
  06_mip_topk_architecture_configs/
    topk_architecture_configs.json
    configs/
      config_rank_00.json
      config_rank_01.json
  07_model_assembly/
    assembled_model.pth
    summary.json
  08_global_knowledge_distillation/
    gkd_model.pth
    summary.json
```

Use `WORKFLOW_OUTPUT_DIR=outputs/<name>` to move the whole workflow tree.
`STAGE_DIR=...` remains accepted as a backward-compatible alias.

## Common Overrides

BLD candidate set and training budget:

```bash
BLD_STEPS=4 \
ATTENTION_VARIANTS=parent_attn,mha_attn,quant_mha_attn,mqa_attn,gqa_kv2,linear_attn,noop_attn \
LAYER_VARIANTS=parent,skip_attn,skip_mlp,skip_both \
bash workflow_steps/04_bld_block_library.sh
```

MIP top-K search:

```bash
TOP_K=5 MEMORY_FRACTION=0.80 RUNTIME_FRACTION=0.80 \
bash workflow_steps/06_mip_topk_configs.sh
```

Assemble a different rank:

```bash
CONFIG_RANK=1 bash workflow_steps/07_assemble_model_from_config.sh
```

Run GKD with OPD:

```bash
GKD_STEPS=20 OPD_WEIGHT=0.25 OPD_MAX_NEW_TOKENS=8 \
bash workflow_steps/08_gkd_distill.sh
```

The staged scripts use the tiny reference model so every stage can run quickly
and produce concrete artifacts. Real Qwen/VLM/VLA runs still use
`scripts/run_qwen3_attention_search.py`; the stage artifacts here mirror the
same BLD, NAS scoring, MIP, assembly, and GKD concepts in a compact executable
workflow.
