# OmniDistill-NAS

OmniDistill-NAS is a distillation-based architecture search framework for
LLM/VLM/VLA inference experiments. It includes BLD, NAS scoring, MIP-based
architecture selection, model assembly, and GKD/OPD workflow stages.

- English docs: [docs/README.md](docs/README.md)
- 中文说明: [docs/README.zh-CN.md](docs/README.zh-CN.md)
- Workflow steps: [docs/scripts.md](docs/scripts.md)

Quick smoke test:

```bash
python3 tools/run_tiny_nas.py --quick
```

Staged workflow:

```bash
bash scripts/run_all.sh
```

Config-driven workflow with resume/cache, evaluation, profiling, export, and
report generation:

```bash
python3 tools/run_experiment.py --config configs/toy_experiment.json
python3 tools/workflow_status.py --workflow-dir outputs/distill_nas_workflow
```

Weighted multi-objective MIP:

```bash
OBJECTIVE_MODE=weighted SCORE_WEIGHT=1.0 MEMORY_WEIGHT=0.25 RUNTIME_WEIGHT=0.25 \
bash scripts/06_mip_topk_configs.sh
```

Pareto frontier sweep and report:

```bash
bash scripts/12_multi_objective_search.sh
```

Real Qwen/VLM/VLA staged workflow:

```bash
WORKFLOW_BACKEND=qwen MODEL_ID=Qwen/Qwen3-0.6B DEVICE=gpu \
bash scripts/run_all.sh
```

Useful real-model switches include `INSTALL_FLA=0/1/auto`, `REQUIRE_FLA=1`,
`TEACHER_DEVICE=cpu`, `STUDENT_DEVICE=gpu`, `STRICT_ACTION_OPD=1`, and
`ALLOW_PARTIAL_CHECKPOINT_LOAD=1`.

Local CI check:

```bash
python3 tools/check_project.py
```
