from __future__ import annotations

import importlib
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class PluginSpec:
    name: str
    category: str
    description: str
    entrypoint: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


PLUGIN_REGISTRY: dict[str, PluginSpec] = {}
PLUGIN_CATEGORIES = {
    "model",
    "dataset",
    "search_space",
    "objective",
    "evaluator",
    "exporter",
    "tracker",
    "quantization",
    "device_policy",
    "rollout_adapter",
}


def register_plugin(
    name: str,
    category: str,
    description: str,
    entrypoint: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> PluginSpec:
    if not name:
        raise ValueError("plugin name must be non-empty")
    if category not in PLUGIN_CATEGORIES:
        raise ValueError(f"unsupported plugin category: {category}")
    spec = PluginSpec(name, category, description, entrypoint=entrypoint, metadata=metadata or {})
    PLUGIN_REGISTRY[name] = spec
    return spec


def list_plugins(category: str | None = None) -> list[dict[str, Any]]:
    specs: list[PluginSpec] = list(PLUGIN_REGISTRY.values())
    if category is not None:
        specs = [spec for spec in specs if spec.category == category]
    return [asdict(spec) for spec in sorted(specs, key=lambda item: (item.category, item.name))]


def _load_entrypoint(entrypoint: str) -> Any:
    if ":" not in entrypoint:
        raise ValueError("plugin entrypoint must use 'module:attribute' syntax")
    module_name, attribute = entrypoint.split(":", 1)
    module = importlib.import_module(module_name)
    target: Any = module
    for part in attribute.split("."):
        target = getattr(target, part)
    return target


def activate_plugin(spec: PluginSpec) -> Any | None:
    if spec.entrypoint is None:
        return None
    target = _load_entrypoint(spec.entrypoint)
    if spec.category == "dataset":
        from .data_adapters import register_dataset_adapter

        register_dataset_adapter(spec.name, target)
    elif spec.category == "evaluator":
        from .evaluation import register_evaluator

        register_evaluator(spec.name, target)
    elif spec.category == "exporter":
        from .export import register_exporter

        register_exporter(spec.name, target)
    elif spec.category == "tracker":
        from .tracking import register_tracking_provider

        register_tracking_provider(spec.name, target)
    elif spec.category == "quantization":
        from .quantization import register_quantization_plan

        register_quantization_plan(spec.name, target)
    elif spec.category == "device_policy":
        from .distributed import register_device_policy

        register_device_policy(spec.name, target)
    elif spec.category == "rollout_adapter":
        from .vla import register_rollout_adapter

        register_rollout_adapter(spec.name, target)
    elif callable(target):
        target(spec)
    return target


def load_plugin_manifest(manifest: dict[str, Any], activate: bool = True) -> list[PluginSpec]:
    plugins = manifest.get("plugins", [])
    if not isinstance(plugins, list):
        raise ValueError("plugin manifest must contain a plugins list")
    registered = []
    for item in plugins:
        if not isinstance(item, dict):
            raise ValueError("plugin manifest entries must be objects")
        spec = register_plugin(
            str(item["name"]),
            str(item["category"]),
            str(item.get("description", "")),
            entrypoint=item.get("entrypoint"),
            metadata=item.get("metadata") if isinstance(item.get("metadata"), dict) else None,
        )
        if activate:
            activate_plugin(spec)
        registered.append(spec)
    return registered
