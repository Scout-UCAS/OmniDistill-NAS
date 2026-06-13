# Tutorial: Qwen Text Workflow

This path targets Qwen-style text models on GPU hosts.

```bash
bash scripts/01_prepare_environment.sh
omnidistill benchmark --suite benchmarks/suites/qwen3_text_smoke.json --dry-run
omnidistill benchmark --suite benchmarks/suites/qwen3_text_smoke.json --allow-commands
```

For controlled experiments, record:

- exact model ID and revision
- prompt source and dataset split
- maximum layers and candidate variants
- GPU name, driver, CUDA, PyTorch, transformers, and FLA versions
- generated result manifest under `results/<run-id>/manifest.json`

