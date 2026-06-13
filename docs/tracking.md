# Experiment Tracking

OmniDistill-NAS includes a lightweight JSONL tracking sink that works without
external services:

```bash
OMNIDISTILL_TRACKING=jsonl \
OMNIDISTILL_TRACKING_FILE=outputs/events.jsonl \
omnidistill benchmark --suite benchmarks/suites/toy_smoke.json --dry-run
```

Each line contains a timestamp, event name, and payload. This is enough for CI,
local debugging, and later ingestion into W&B, MLflow, or TensorBoard.

Environment variables:

- `OMNIDISTILL_TRACKING=none|jsonl|wandb|mlflow|tensorboard`
- `OMNIDISTILL_TRACKING_FILE=outputs/omnidistill_events.jsonl`

The `wandb`, `mlflow`, and `tensorboard` provider names are reserved extension
points. The default implementation reports that they need external setup rather
than importing optional packages implicitly.

