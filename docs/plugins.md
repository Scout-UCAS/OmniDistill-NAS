# Plugin Architecture

The core extension surface is intentionally small:

- dataset adapters: `distill_nas_core.data_adapters.register_dataset_adapter`
- attention/search variants: `distill_nas_core.search_space.register_attention_variant`
- project plugins: `distill_nas_core.plugins.register_plugin`

Plugin categories:

- `model`
- `dataset`
- `search_space`
- `objective`
- `evaluator`
- `exporter`
- `tracker`

Example:

```python
from distill_nas_core.plugins import register_plugin

register_plugin(
    name="my-vla-evaluator",
    category="evaluator",
    description="Evaluate VLA action outputs on a private robotics benchmark.",
    entrypoint="my_package.evaluators:make_evaluator",
)
```

List registered plugins:

```bash
omnidistill plugins
```

