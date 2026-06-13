# Benchmarking

OmniDistill-NAS benchmark suites are JSON files under `benchmarks/suites/`.
They make heavy experiments explicit without forcing every developer machine to
download large checkpoints.

```bash
omnidistill benchmark --suite benchmarks/suites/toy_smoke.json --dry-run
omnidistill benchmark --suite benchmarks/suites/toy_smoke.json
```

Real-model suites are command based:

- `benchmarks/suites/qwen3_text_smoke.json`
- `benchmarks/suites/qwen_vlm_smoke.json`
- `benchmarks/suites/openvla_smoke.json`

External command benchmarks are skipped unless `--allow-commands` is passed.
After a run, add a manifest under `results/<run-id>/manifest.json` instead of
committing model checkpoints.

Recommended metrics:

- quality: loss, perplexity, task accuracy, action MSE/KL
- resources: parameter memory, KV-cache memory, peak memory
- speed: latency, throughput, batch-size profile
- search: number of candidates, Pareto-front size, objective weights

