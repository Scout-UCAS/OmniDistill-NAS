from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .artifacts import resolve_path


class SchemaError(ValueError):
    """Raised when a public config file does not match the project schema."""


EXPERIMENT_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "OmniDistill-NAS experiment spec",
    "type": "object",
    "required": ["backend", "stages"],
    "properties": {
        "name": {"type": "string"},
        "backend": {"type": "string"},
        "output_dir": {"type": "string"},
        "workdir": {"type": "string"},
        "stages": {
            "oneOf": [
                {"type": "string"},
                {"type": "array", "items": {"type": "string"}},
            ]
        },
        "model": {"type": "object"},
        "search": {"type": "object"},
        "distillation": {"type": "object"},
        "devices": {"type": "object"},
        "distributed": {"type": "object"},
        "evaluation": {"type": "object"},
        "profiling": {"type": "object"},
        "env": {"type": "object", "additionalProperties": {"type": ["string", "number", "boolean"]}},
    },
    "additionalProperties": True,
}

BENCHMARK_SUITE_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "OmniDistill-NAS benchmark suite",
    "type": "object",
    "required": ["schema_version", "name", "benchmarks"],
    "properties": {
        "schema_version": {"type": "integer", "minimum": 1},
        "name": {"type": "string"},
        "description": {"type": "string"},
        "benchmarks": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["id", "task", "backend"],
                "properties": {
                    "id": {"type": "string"},
                    "task": {"type": "string"},
                    "backend": {"type": "string"},
                    "config": {"type": "string"},
                    "command": {"type": "array", "items": {"type": "string"}},
                    "env": {"type": "object"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "expected_metrics": {"type": "array", "items": {"type": "string"}},
                },
                "additionalProperties": True,
            },
        },
    },
    "additionalProperties": True,
}

RESULT_MANIFEST_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "OmniDistill-NAS result manifest",
    "type": "object",
    "required": ["schema_version", "id", "backend", "task", "date", "metrics"],
    "properties": {
        "schema_version": {"type": "integer", "minimum": 1},
        "id": {"type": "string"},
        "title": {"type": "string"},
        "backend": {"type": "string"},
        "model": {"type": "string"},
        "task": {"type": "string"},
        "date": {"type": "string"},
        "hardware": {"type": "object"},
        "config": {"type": "string"},
        "command": {"type": "string"},
        "metrics": {"type": "object"},
        "artifacts": {"type": "object"},
        "notes": {"type": "string"},
    },
    "additionalProperties": True,
}


def _is_mapping(value: Any) -> bool:
    return isinstance(value, dict)


def _is_string_list(value: Any) -> bool:
    return isinstance(value, list) and all(isinstance(item, str) for item in value)


def _require_mapping(payload: Any, label: str, errors: list[str]) -> None:
    if not _is_mapping(payload):
        errors.append(f"{label} must be a JSON object")


def validate_experiment_spec(spec: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    _require_mapping(spec, "experiment spec", errors)
    if errors:
        return errors
    if "backend" not in spec:
        errors.append("experiment spec must include 'backend'")
    elif not isinstance(spec["backend"], str):
        errors.append("'backend' must be a string")
    stages = spec.get("stages")
    if stages is None:
        errors.append("experiment spec must include 'stages'")
    elif not isinstance(stages, str) and not _is_string_list(stages):
        errors.append("'stages' must be a comma-separated string or list of strings")
    for key in ("model", "search", "distillation", "devices", "distributed", "evaluation", "profiling", "env"):
        if key in spec and not _is_mapping(spec[key]):
            errors.append(f"'{key}' must be an object")
    for key in ("output_dir", "workdir", "name"):
        if key in spec and spec[key] is not None and not isinstance(spec[key], str):
            errors.append(f"'{key}' must be a string")
    return errors


def validate_benchmark_suite(suite: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    _require_mapping(suite, "benchmark suite", errors)
    if errors:
        return errors
    if not isinstance(suite.get("schema_version"), int):
        errors.append("'schema_version' must be an integer")
    if not isinstance(suite.get("name"), str):
        errors.append("'name' must be a string")
    benchmarks = suite.get("benchmarks")
    if not isinstance(benchmarks, list) or not benchmarks:
        errors.append("'benchmarks' must be a non-empty list")
        return errors
    seen: set[str] = set()
    for index, benchmark in enumerate(benchmarks):
        label = f"benchmarks[{index}]"
        if not _is_mapping(benchmark):
            errors.append(f"{label} must be an object")
            continue
        benchmark_id = benchmark.get("id")
        if not isinstance(benchmark_id, str) or not benchmark_id:
            errors.append(f"{label}.id must be a non-empty string")
        elif benchmark_id in seen:
            errors.append(f"{label}.id {benchmark_id!r} is duplicated")
        else:
            seen.add(benchmark_id)
        for key in ("task", "backend"):
            if not isinstance(benchmark.get(key), str):
                errors.append(f"{label}.{key} must be a string")
        if "config" not in benchmark and "command" not in benchmark:
            errors.append(f"{label} must define either 'config' or 'command'")
        if "command" in benchmark and not _is_string_list(benchmark["command"]):
            errors.append(f"{label}.command must be a list of strings")
    return errors


def validate_result_manifest(manifest: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    _require_mapping(manifest, "result manifest", errors)
    if errors:
        return errors
    for key in ("schema_version", "id", "backend", "task", "date", "metrics"):
        if key not in manifest:
            errors.append(f"result manifest must include {key!r}")
    if "schema_version" in manifest and not isinstance(manifest["schema_version"], int):
        errors.append("'schema_version' must be an integer")
    for key in ("id", "backend", "model", "task", "date", "title", "config", "command", "notes"):
        if key in manifest and manifest[key] is not None and not isinstance(manifest[key], str):
            errors.append(f"'{key}' must be a string")
    if "metrics" in manifest and not _is_mapping(manifest["metrics"]):
        errors.append("'metrics' must be an object")
    for key in ("hardware", "artifacts"):
        if key in manifest and not _is_mapping(manifest[key]):
            errors.append(f"'{key}' must be an object")
    return errors


def raise_if_errors(errors: list[str]) -> None:
    if errors:
        raise SchemaError("; ".join(errors))


def load_json_file(path: str | Path) -> dict[str, Any]:
    resolved = resolve_path(path)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SchemaError(f"expected JSON object at {resolved}")
    return payload


def validate_json_file(path: str | Path, kind: str) -> list[str]:
    payload = load_json_file(path)
    if kind == "experiment":
        return validate_experiment_spec(payload)
    if kind == "benchmark":
        return validate_benchmark_suite(payload)
    if kind == "result":
        return validate_result_manifest(payload)
    raise ValueError(f"unknown schema kind: {kind}")

