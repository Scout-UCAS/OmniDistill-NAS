from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .artifacts import resolve_path
from .schema import raise_if_errors, validate_result_manifest


def load_result_manifests(results_dir: str | Path = "results") -> list[dict[str, Any]]:
    root = resolve_path(results_dir)
    manifests: list[dict[str, Any]] = []
    if not root.exists():
        return manifests
    for path in sorted(root.glob("*/manifest.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            raise_if_errors(validate_result_manifest(payload))
            payload["_path"] = str(path)
            manifests.append(payload)
    return manifests


def result_metric(manifest: dict[str, Any], key: str) -> Any:
    metrics = manifest.get("metrics", {})
    return metrics.get(key) if isinstance(metrics, dict) else None


def result_table(manifests: list[dict[str, Any]]) -> str:
    lines = [
        "| Result | Backend | Task | Model | Perplexity | Memory bytes | Throughput tok/s | Config |",
        "| --- | --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for item in manifests:
        metrics = item.get("metrics", {}) if isinstance(item.get("metrics"), dict) else {}
        artifacts = item.get("artifacts", {}) if isinstance(item.get("artifacts"), dict) else {}
        throughput = metrics.get("throughput_tokens_per_second")
        if throughput is None and isinstance(metrics.get("profiles"), dict):
            first_profile: Any = next(iter(metrics["profiles"].values()), {})
            throughput = first_profile.get("throughput_tokens_per_second") if isinstance(first_profile, dict) else None
        lines.append(
            "| {title} | {backend} | {task} | {model} | {perplexity} | {memory} | {throughput} | {config} |".format(
                title=item.get("title") or item.get("id"),
                backend=item.get("backend", ""),
                task=item.get("task", ""),
                model=item.get("model", ""),
                perplexity=_format_metric(metrics.get("perplexity")),
                memory=_format_metric(metrics.get("parameter_memory_bytes")),
                throughput=_format_metric(throughput),
                config=artifacts.get("config") or item.get("config", ""),
            )
        )
    return "\n".join(lines)


def _format_metric(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def write_result_index(results_dir: str | Path = "results", output_md: str | Path | None = None) -> Path:
    root = resolve_path(results_dir)
    target = resolve_path(output_md) if output_md is not None else root / "README.md"
    manifests = load_result_manifests(root)
    lines = [
        "# OmniDistill-NAS Result Zoo",
        "",
        "This directory tracks reproducible benchmark manifests. Heavy artifacts stay outside Git; manifests point to configs, commands, and generated reports.",
        "",
        result_table(manifests) if manifests else "No result manifests found.",
        "",
    ]
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines), encoding="utf-8")
    return target
