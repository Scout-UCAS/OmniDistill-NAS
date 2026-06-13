# Plugin Architecture

The core extension surface is intentionally small:

- dataset adapters: `distill_nas_core.data_adapters.register_dataset_adapter`
- attention/search variants: `distill_nas_core.search_space.register_attention_variant`
- evaluators: `distill_nas_core.evaluation.register_evaluator`
- exporters: `distill_nas_core.export.register_exporter`
- tracking providers: `distill_nas_core.tracking.register_tracking_provider`
- quantization plans: `distill_nas_core.quantization.register_quantization_plan`
- device policies: `distill_nas_core.distributed.register_device_policy`
- VLA rollout adapters: `distill_nas_core.vla.register_rollout_adapter`
- project plugins: `distill_nas_core.plugins.register_plugin`

Plugin categories:

- `model`
- `dataset`
- `search_space`
- `objective`
- `evaluator`
- `exporter`
- `tracker`
- `quantization`
- `device_policy`
- `rollout_adapter`

Manifest entries with an `entrypoint` use `module:attribute` syntax. Loading a
manifest activates supported categories by registering the imported callable
into the matching runtime registry.

Example:

```python
from distill_nas_core.plugins import load_plugin_manifest

load_plugin_manifest({
    "plugins": [
        {
            "name": "my-vla-evaluator",
            "category": "evaluator",
            "description": "Evaluate VLA action outputs on a private robotics benchmark.",
            "entrypoint": "my_package.evaluators:evaluate_artifact",
        }
    ]
})
```

List registered plugins:

```bash
omnidistill plugins
```
