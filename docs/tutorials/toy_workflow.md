# Tutorial: Toy Workflow

Run the full toy workflow:

```bash
python -m pip install -e ".[dev,docs]"
omnidistill run --config configs/toy_experiment.json
omnidistill status --workflow-dir outputs/distill_nas_workflow
omnidistill report --workflow-dir outputs/distill_nas_workflow
```

Run only the benchmark harness:

```bash
omnidistill benchmark --suite benchmarks/suites/toy_smoke.json --dry-run
omnidistill benchmark --suite benchmarks/suites/toy_smoke.json
```

The toy workflow is designed for fast local validation. It exercises candidate
generation, MIP, Pareto extraction, assembly, GKD, evaluation, profiling, export,
and reporting without downloading a large model.

