# Result Zoo

The result zoo is a manifest index, not a checkpoint store. Each result should
include:

- the exact config or command
- hardware and dependency notes
- quality, memory, latency, and throughput metrics
- links to generated reports, Pareto plots, and exported artifacts

Validate a manifest:

```bash
omnidistill validate results/toy_smoke/manifest.json --kind result
```

Regenerate the index:

```bash
omnidistill report --results-dir results
```

The toy result is intentionally small and CI-friendly. Larger Qwen, VLM, or VLA
results should be added as separate manifests after reproducible runs.

