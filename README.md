# OmniDistill-NAS

[![CI](https://github.com/Scout-UCAS/OmniDistill-NAS/actions/workflows/ci.yml/badge.svg)](https://github.com/Scout-UCAS/OmniDistill-NAS/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-supported-ee4c2c.svg)](https://pytorch.org/)

**Distillation-guided neural architecture search for efficient LLM, VLM, and
VLA inference.**

OmniDistill-NAS turns architecture compression into a reproducible workflow:
build a candidate block library, score layer replacements, solve constrained
NAS problems, assemble the selected student, distill it globally, and export a
profiled artifact. It is designed for research workflows where quality,
latency, memory, and deployment constraints have to be explored together.

```bash
python3 tools/run_tiny_nas.py --quick
bash scripts/run_all.sh
```

## Why OmniDistill-NAS

Modern model compression is rarely a single knob. A useful student model may
combine attention variants, FFN pruning, quantized projections, layer skips,
resource budgets, and post-search distillation. OmniDistill-NAS provides one
place to run that loop end to end.

- **Distillation-first search**: candidates are scored by replacement quality,
  then refined through global knowledge distillation.
- **Constrained architecture selection**: MIP search supports score, memory,
  runtime, throughput, diversity, and weighted multi-objective objectives.
- **Pareto exploration**: sweep objective weights and export Pareto-front
  configs, reports, and SVG visualizations.
- **LLM, VLM, and VLA hooks**: Qwen-style text, vision-language, and
  vision-language-action workflows share the same staged interface.
- **Operational workflow**: evaluation, profiling, export manifests, status
  checks, local CI, and resumable config-driven experiments are included.
- **Open extension points**: add datasets, attention variants, quantization
  checks, distributed plans, or rollout evaluators without rewriting the
  pipeline.

## Install

For local development:

```bash
git clone git@github.com:Scout-UCAS/OmniDistill-NAS.git
cd OmniDistill-NAS
python3 -m pip install -e ".[dev]"
```

The workflow scripts can also prepare workspace-local optional dependencies
under `vendor/python`:

```bash
bash scripts/01_prepare_environment.sh
```

Generated artifacts are written under `outputs/`, model checkpoints under
`checkpoints/`, and Hugging Face caches under `hf_cache/` in the active
workspace. These directories are ignored by Git.

## Quick Start

Run the tiny end-to-end smoke test:

```bash
python3 tools/run_tiny_nas.py --quick
```

Run the complete staged workflow:

```bash
bash scripts/run_all.sh
```

Inspect generated artifacts:

```bash
omnidistill status --workflow-dir outputs/distill_nas_workflow
sed -n '1,120p' outputs/distill_nas_workflow/report.md
```

Use the config-driven runner with resume/cache behavior:

```bash
omnidistill run --config configs/toy_experiment.json
omnidistill run --config configs/toy_experiment.json --from-stage evaluate
```

Use `--workdir /path/to/workspace` to keep outputs, checkpoints, caches, and
vendored optional dependencies outside the source tree.

## Workflow

The default workflow is split into small, inspectable scripts:

| Stage | Script | Output |
| --- | --- | --- |
| Prepare | `scripts/01_prepare_environment.sh` | local environment and caches |
| Validate | `scripts/02_validate_project.sh` | compile, unit tests, CLI checks |
| Smoke | `scripts/03_smoke_tiny_nas.sh` | quick NAS run |
| BLD | `scripts/04_bld_block_library.sh` | block library artifact |
| Score | `scripts/05_nas_layer_importance.sh` | layer replacement scores |
| Search | `scripts/06_mip_topk_configs.sh` | top-K architecture configs |
| Pareto | `scripts/12_multi_objective_search.sh` | Pareto configs and plot |
| Assemble | `scripts/07_assemble_model_from_config.sh` | selected student model |
| Distill | `scripts/08_gkd_distill.sh` | distilled model artifact |
| Evaluate | `scripts/09_evaluate_artifact.sh` | quality metrics |
| Profile | `scripts/10_profile_artifact.sh` | latency and throughput profile |
| Export | `scripts/11_export_and_report.sh` | portable export and report |

More detail: [docs/scripts.md](docs/scripts.md)

## Multi-Objective Search

Run a weighted score/memory/runtime MIP:

```bash
OBJECTIVE_MODE=weighted \
SCORE_WEIGHT=1.0 \
MEMORY_WEIGHT=0.25 \
RUNTIME_WEIGHT=0.25 \
bash scripts/06_mip_topk_configs.sh
```

Generate a Pareto frontier report:

```bash
GRID_RESOLUTION=5 PARETO_MODE=auto bash scripts/12_multi_objective_search.sh
```

Key outputs:

- `outputs/distill_nas_workflow/06_mip_topk_architecture_configs/topk_architecture_configs.json`
- `outputs/distill_nas_workflow/06_mip_topk_architecture_configs/multi_objective_search.json`
- `outputs/distill_nas_workflow/06_mip_topk_architecture_configs/multi_objective_report.md`
- `outputs/distill_nas_workflow/06_mip_topk_architecture_configs/pareto_front.svg`

## Real Model Workflows

Run the staged workflow on a Qwen-style model:

```bash
WORKFLOW_BACKEND=qwen \
MODEL_ID=Qwen/Qwen3-0.6B \
DEVICE=gpu \
bash scripts/run_all.sh
```

Useful switches:

- `MODEL_KIND=auto|text|vlm|vla`
- `INSTALL_FLA=0|1|auto`
- `REQUIRE_FLA=1`
- `TEACHER_DEVICE=cpu`
- `STUDENT_DEVICE=gpu`
- `STRICT_ACTION_OPD=1`
- `ALLOW_PARTIAL_CHECKPOINT_LOAD=1`

For dataset-backed prompts, VLM examples, VLA action scoring, and FLA variants,
see [docs/README.md](docs/README.md) and [docs/README.zh-CN.md](docs/README.zh-CN.md).

## Python API

```python
from distill_nas_core.mip import SearchCandidate, SearchConstraints, solve_nas_mip

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

## Project Layout

```text
distill_nas_core/      Core library: blocks, scoring, MIP, distillation, export
scripts/               Shell workflow steps
tools/                 Python CLI tools
tests/                 Unit and integration tests
configs/               Experiment specs
docs/                  Extended English and Chinese documentation
outputs/               Generated workflow artifacts, ignored by Git
checkpoints/           Generated model checkpoints, ignored by Git
```

## Validation

Run the local CI suite:

```bash
python3 tools/check_project.py
```

Run the full workflow as CI does:

```bash
python3 tools/check_project.py --workflow
```

The current suite covers MIP search, weighted objectives, Pareto extraction,
artifact evaluation/profiling/export, Qwen/VLM helpers, workflow planning, and
the staged toy pipeline.

## Documentation

- [Full English documentation](docs/README.md)
- [中文说明](docs/README.zh-CN.md)
- [Workflow script reference](docs/scripts.md)

## Status

OmniDistill-NAS is a research-oriented framework. The tiny workflow is designed
to run quickly on a local machine, while real-model workflows depend on model
availability, device memory, optional FLA support, and dataset access.

## Citation

If this project helps your research, please cite the repository for now. A
formal citation entry can be added when the accompanying paper or technical
report is released.
