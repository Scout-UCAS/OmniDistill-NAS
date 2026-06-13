from __future__ import annotations

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


def register_plugin(
    name: str,
    category: str,
    description: str,
    entrypoint: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> PluginSpec:
    if not name:
        raise ValueError("plugin name must be non-empty")
    if category not in {"model", "dataset", "search_space", "objective", "evaluator", "exporter", "tracker"}:
        raise ValueError(f"unsupported plugin category: {category}")
    spec = PluginSpec(name, category, description, entrypoint=entrypoint, metadata=metadata or {})
    PLUGIN_REGISTRY[name] = spec
    return spec


def list_plugins(category: str | None = None) -> list[dict[str, Any]]:
    specs: list[PluginSpec] = list(PLUGIN_REGISTRY.values())
    if category is not None:
        specs = [spec for spec in specs if spec.category == category]
    return [asdict(spec) for spec in sorted(specs, key=lambda item: (item.category, item.name))]


def load_plugin_manifest(manifest: dict[str, Any]) -> list[PluginSpec]:
    plugins = manifest.get("plugins", [])
    if not isinstance(plugins, list):
        raise ValueError("plugin manifest must contain a plugins list")
    registered = []
    for item in plugins:
        if not isinstance(item, dict):
            raise ValueError("plugin manifest entries must be objects")
        registered.append(
            register_plugin(
                str(item["name"]),
                str(item["category"]),
                str(item.get("description", "")),
                entrypoint=item.get("entrypoint"),
                metadata=item.get("metadata") if isinstance(item.get("metadata"), dict) else None,
            )
        )
    return registered
