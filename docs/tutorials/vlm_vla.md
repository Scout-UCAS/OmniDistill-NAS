# Tutorial: VLM and VLA Workflows

Vision-language and vision-language-action models use the same staged search
interface, but dataset formatting matters more.

Dry-run the published suites:

```bash
omnidistill benchmark --suite benchmarks/suites/qwen_vlm_smoke.json --dry-run
omnidistill benchmark --suite benchmarks/suites/openvla_smoke.json --dry-run
```

For real runs, pass `--allow-commands` only on a machine with the right model
access, GPU memory, and dataset/image paths.

Important switches:

- `--model-kind vlm|vla`
- `--allow-blank-image` for smoke tests only
- `--dataset-image-root` for relative image paths
- `STRICT_ACTION_OPD=1` when action outputs are required

