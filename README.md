# OmniDistill-NAS

[![CI](https://github.com/Scout-UCAS/OmniDistill-NAS/actions/workflows/ci.yml/badge.svg)](https://github.com/Scout-UCAS/OmniDistill-NAS/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-supported-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Distillation-guided neural architecture search for efficient LLM, VLM, and VLA inference.**

OmniDistill-NAS is an end-to-end research framework for building smaller,
faster, and more deployable Transformer students. It combines block-level
distillation, layer-wise replacement scoring, constrained MIP search,
multi-objective Pareto exploration, global knowledge distillation, evaluation,
profiling, export, benchmark manifests, and result tracking in one reproducible
workflow.

```bash
git clone git@github.com:Scout-UCAS/OmniDistill-NAS.git
cd OmniDistill-NAS
python3 -m pip install -e ".[dev,docs]"

omnidistill benchmark --dry-run
omnidistill run --config configs/toy_experiment.json
```

## Why It Exists

Model compression is no longer a single pruning ratio or quantization flag.
Practical inference systems have to balance quality, memory, latency,
throughput, hardware limits, attention variants, FFN changes, layer skips,
distillation losses, and export constraints at the same time.

OmniDistill-NAS turns that messy search loop into a structured experiment:

1. Build a library of candidate replacement blocks.
2. Measure replacement quality with distillation-aware scores.
3. Solve constrained architecture search problems with MIP.
4. Sweep score, memory, and runtime trade-offs to expose Pareto candidates.
5. Assemble the selected student and refine it with GKD/OPD.
6. Evaluate, profile, export, and record the run with auditable manifests.

The result is not just a search script. It is a workflow you can inspect,
resume, benchmark, package, and extend.

## Highlights

- **Distillation-first NAS**: candidate blocks are scored by teacher-student
  behavior, then refined through global knowledge distillation.
- **Constrained MIP search**: select architectures under memory, latency,
  runtime, throughput, and diversity constraints.
- **Multi-objective Pareto search**: sweep score, memory, and runtime weights;
  export Pareto configs, Markdown reports, and SVG frontier plots.
- **LLM, VLM, and VLA paths**: Qwen-style text, vision-language, and
  vision-language-action workflows share the same staged interface.
- **Runnable by default**: a packaged toy benchmark works immediately after
  installation, including outside the source tree.
- **Reproducible artifacts**: every major stage writes structured artifacts,
  summaries, profiles, export manifests, and reports.
- **Operational CLI**: initialize configs, run staged experiments, resume from a
  stage, inspect status, validate schemas, run benchmark suites, and generate
  reports.
- **Extension surface**: register attention variants, dataset adapters,
  evaluators, exporters, tracking providers, quantization plans, device
  policies, and VLA rollout adapters without rewriting the core pipeline.

## Install

OmniDistill-NAS targets Python 3.10+ and PyTorch.

```bash
git clone git@github.com:Scout-UCAS/OmniDistill-NAS.git
cd OmniDistill-NAS
python3 -m pip install -e .
```

For development, docs, tests, and optional real-model tooling:

```bash
python3 -m pip install -e ".[dev,docs]"
```

The workflow can also prepare workspace-local optional dependencies under
`vendor/python`:

```bash
bash scripts/01_prepare_environment.sh
```

Generated files are intentionally kept out of Git:

```text
outputs/       workflow artifacts, reports, profiles, exports
checkpoints/   generated model checkpoints
hf_cache/      Hugging Face model and dataset caches
vendor/        optional local dependencies
```

## Quick Start

Run the packaged smoke benchmark:

```bash
omnidistill benchmark --dry-run
```

Run the full toy workflow:

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

Create your own experiment spec:

```bash
omnidistill init --output configs/my_experiment.json
omnidistill validate configs/my_experiment.json
```

JSON and YAML experiment specs are both supported.

## CLI

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

Useful examples:

```bash
omnidistill plan --config configs/toy_experiment.json
omnidistill run --config configs/toy_experiment.json --dry-run
omnidistill benchmark --suite benchmarks/suites/qwen3_text_smoke.json --dry-run
omnidistill report --results-dir results
```

Use `--workdir /path/to/workspace` to keep outputs, caches, and local
dependencies outside the source tree.

## Workflow

The default pipeline is intentionally split into small, inspectable stages.

| Stage | Script | Main output |
| --- | --- | --- |
| Prepare | `scripts/01_prepare_environment.sh` | local env, caches, optional deps |
| Validate | `scripts/02_validate_project.sh` | compile checks, tests, CLI checks |
| Smoke | `scripts/03_smoke_tiny_nas.sh` | quick NAS sanity run |
| BLD | `scripts/04_bld_block_library.sh` | candidate block library |
| Score | `scripts/05_nas_layer_importance.sh` | layer replacement scores |
| Search | `scripts/06_mip_topk_configs.sh` | top-K architecture configs |
| Pareto | `scripts/12_multi_objective_search.sh` | Pareto configs, report, SVG plot |
| Assemble | `scripts/07_assemble_model_from_config.sh` | selected student model |
| Distill | `scripts/08_gkd_distill.sh` | GKD/OPD model artifact |
| Evaluate | `scripts/09_evaluate_artifact.sh` | quality metrics |
| Profile | `scripts/10_profile_artifact.sh` | latency, throughput, memory profile |
| Export | `scripts/11_export_and_report.sh` | portable export and final report |

Run the shell workflow directly:

```bash
bash scripts/run_all.sh
```

See [docs/scripts.md](docs/scripts.md) for every environment variable and
stage-level command.

## Multi-Objective Search

The search layer supports single-objective score optimization and weighted
multi-objective optimization over quality, memory, and runtime proxies.

```bash
OBJECTIVE_MODE=weighted \
SCORE_WEIGHT=1.0 \
MEMORY_WEIGHT=0.25 \
RUNTIME_WEIGHT=0.25 \
bash scripts/06_mip_topk_configs.sh
```

Generate a Pareto frontier:

```bash
GRID_RESOLUTION=5 PARETO_MODE=auto bash scripts/12_multi_objective_search.sh
```

Key outputs:

```text
outputs/distill_nas_workflow/06_mip_topk_architecture_configs/topk_architecture_configs.json
outputs/distill_nas_workflow/06_mip_topk_architecture_configs/multi_objective_search.json
outputs/distill_nas_workflow/06_mip_topk_architecture_configs/multi_objective_report.md
outputs/distill_nas_workflow/06_mip_topk_architecture_configs/pareto_front.svg
```

## Benchmarks And Result Zoo

Benchmark suites are JSON manifests under [benchmarks/suites](benchmarks/suites).
The default CLI benchmark is packaged with the Python package, so this works
even outside a source checkout:

```bash
omnidistill benchmark --dry-run
```

Source checkout suites include:

```bash
omnidistill benchmark --suite benchmarks/suites/toy_smoke.json --dry-run
omnidistill benchmark --suite benchmarks/suites/qwen3_text_smoke.json --dry-run
omnidistill benchmark --suite benchmarks/suites/qwen_vlm_smoke.json --dry-run
omnidistill benchmark --suite benchmarks/suites/openvla_smoke.json --dry-run
```

External-command benchmarks are dry-run safe and require `--allow-commands`
before they execute real model jobs.

Validate and index result manifests:

```bash
omnidistill validate results/toy_smoke/manifest.json --kind result
omnidistill report --results-dir results
```

See [docs/benchmarking.md](docs/benchmarking.md) and
[docs/result_zoo.md](docs/result_zoo.md).

## Real Model Workflows

The toy backend is designed for fast local verification. Real-model workflows
target Qwen-style text, VLM, and VLA models with the same stage structure.

```bash
WORKFLOW_BACKEND=qwen \
MODEL_ID=Qwen/Qwen3-0.6B \
MODEL_KIND=auto \
DEVICE=gpu \
bash scripts/run_all.sh
```

Common switches:

```text
MODEL_KIND=auto|text|vlm|vla
INSTALL_FLA=0|1|auto
REQUIRE_FLA=1
TEACHER_DEVICE=cpu
STUDENT_DEVICE=gpu
STRICT_ACTION_OPD=1
ALLOW_PARTIAL_CHECKPOINT_LOAD=1
```

The real-model runner supports dataset-backed prompts, VLM image examples, VLA
action fields, FLA candidates when available, and Qwen-style decoder-layer
replacement. See [docs/README.md](docs/README.md),
[docs/tutorials/qwen3_text.md](docs/tutorials/qwen3_text.md), and
[docs/tutorials/vlm_vla.md](docs/tutorials/vlm_vla.md).

## Python API

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

## Architecture

```text
distill_nas_core/
  blocks.py             candidate attention and FFN blocks
  search_space.py       variant registry and search-space construction
  scoring.py            layer replacement scoring utilities
  mip.py                constrained NAS solver
  multi_objective.py    Pareto and weighted objective search
  distill.py            GKD and OPD training utilities
  evaluation.py         artifact evaluation
  profiler.py           latency, throughput, memory profiling
  export.py             portable artifact export
  benchmarks.py         benchmark suite runner
  schema.py             public config/result validation
  plugins.py            extension registry
```

Repository layout:

```text
scripts/       shell workflow stages
tools/         Python entry-point utilities
configs/       experiment specs
benchmarks/    benchmark suite manifests
results/       reproducible result manifests
schemas/       public JSON schema files
docs/          full English and Chinese documentation
notebooks/     notebook and Colab demos
website/       static project homepage
tests/         unit and integration tests
```

## Quality Bar

The project includes unit tests, workflow tests, schema validation, linting,
type checks for core platform modules, package builds, and docs builds.

```bash
python3 -m pytest -q
python3 -m ruff check distill_nas_core tools tests
python3 tools/check_project.py
python3 tools/check_project.py --workflow
```

The current suite covers MIP search, weighted objectives, Pareto extraction,
artifact evaluation/profiling/export, Qwen/VLM helpers, workflow planning,
packaged benchmark assets, YAML validation, and the staged toy pipeline.

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

## Project Status

OmniDistill-NAS is research software. The toy workflow is CI-friendly and
intended to run quickly on local machines. Real-model workflows depend on model
availability, device memory, optional FLA support, dataset access, and hardware
profiling conditions.

The project is in alpha: APIs are usable, tests are in place, and the workflow
is end-to-end, but large-model result manifests should be treated as
reproducibility targets rather than universal performance claims.

See [ROADMAP.md](ROADMAP.md) for planned work.

## Contributing

Contributions are welcome across search spaces, model adapters, dataset
formatters, benchmark manifests, documentation, CI, and real-model run records.

Start with:

- [CONTRIBUTING.md](CONTRIBUTING.md)
- [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)
- [SECURITY.md](SECURITY.md)

## Citation

If OmniDistill-NAS helps your research, please cite the repository using
[CITATION.cff](CITATION.cff). A formal paper citation can be added when the
accompanying report is released.

## License

OmniDistill-NAS is released under the [MIT License](LICENSE).
