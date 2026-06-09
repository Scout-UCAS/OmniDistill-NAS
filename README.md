# OmniDistill-NAS

OmniDistill-NAS is a distillation-based architecture search framework for
LLM/VLM/VLA inference experiments. It includes BLD, NAS scoring, MIP-based
architecture selection, model assembly, and GKD/OPD workflow stages.

- English docs: [docs/README.md](docs/README.md)
- 中文说明: [docs/README.zh-CN.md](docs/README.zh-CN.md)
- Workflow steps: [docs/workflow_steps.md](docs/workflow_steps.md)

Quick smoke test:

```bash
python3 scripts/run_tiny_nas.py --quick
```

Staged workflow:

```bash
bash workflow_steps/run_all.sh
```

Real Qwen/VLM/VLA staged workflow:

```bash
WORKFLOW_BACKEND=qwen MODEL_ID=Qwen/Qwen3-0.6B DEVICE=gpu \
bash workflow_steps/run_all.sh
```
