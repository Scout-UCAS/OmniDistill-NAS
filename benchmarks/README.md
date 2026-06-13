# Benchmarks

Benchmark suites are JSON manifests consumed by:

```bash
omnidistill benchmark --suite benchmarks/suites/toy_smoke.json --dry-run
omnidistill benchmark --suite benchmarks/suites/toy_smoke.json
```

Each suite lists reproducible jobs, expected metrics, and either an experiment
config or an explicit command. Heavy model suites are dry-run safe by default;
external commands only run when `--allow-commands` is passed.

Result manifests belong under `results/<run-id>/manifest.json`.
