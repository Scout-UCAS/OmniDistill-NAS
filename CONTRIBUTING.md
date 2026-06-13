# Contributing

Thanks for improving OmniDistill-NAS. The project values reproducible research,
small reviewable changes, and clear benchmark evidence.

## Development Setup

```bash
python -m pip install -e ".[dev,docs]"
python tools/check_project.py
```

## Pull Request Checklist

- Add or update tests for code changes.
- Validate configs with `omnidistill validate`.
- For benchmark claims, add a result manifest under `results/<run-id>/`.
- Do not commit generated checkpoints, model caches, or workflow outputs.
- Keep new optional integrations behind explicit flags or extras.

## Benchmark Contributions

Benchmark PRs should include:

- suite or command used
- config file
- hardware details
- dependency versions
- metrics and generated report path
- result manifest

## Plugin Contributions

New model, dataset, evaluator, objective, or tracker integrations should use
the registry APIs documented in `docs/plugins.md`.

