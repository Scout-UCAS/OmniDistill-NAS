from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .artifacts import resolve_path
from .experiment import load_experiment_spec, run_experiment
from .schema import raise_if_errors, validate_benchmark_suite
from .tracking import emit_tracking_event


def resolve_benchmark_command(command: list[Any]) -> list[str]:
    resolved = [str(part) for part in command]
    if resolved and resolved[0] in {"python", "python3"}:
        resolved[0] = sys.executable
    return resolved


def load_benchmark_suite(path: str | Path) -> dict[str, Any]:
    resolved = resolve_path(path)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"benchmark suite must be a JSON object: {resolved}")
    raise_if_errors(validate_benchmark_suite(payload))
    payload["_suite_path"] = str(resolved)
    return payload


def benchmark_plan(suite: dict[str, Any], suite_root: str | Path | None = None) -> list[dict[str, Any]]:
    root = Path(suite_root) if suite_root is not None else Path(suite.get("_suite_path", ".")).resolve().parent
    plan: list[dict[str, Any]] = []
    for item in suite.get("benchmarks", []):
        record = {
            "id": item["id"],
            "task": item["task"],
            "backend": item["backend"],
            "tags": item.get("tags", []),
            "expected_metrics": item.get("expected_metrics", []),
            "kind": "experiment" if "config" in item else "command",
        }
        if "config" in item:
            record["config"] = str(resolve_path(item["config"], root=root))
        if "command" in item:
            record["command"] = resolve_benchmark_command(item["command"])
        plan.append(record)
    return plan


def run_benchmark_suite(
    suite_path: str | Path,
    result_dir: str | Path = "benchmark_runs",
    dry_run: bool = False,
    workdir: str | Path | None = None,
    allow_commands: bool = False,
) -> dict[str, Any]:
    suite = load_benchmark_suite(suite_path)
    suite_root = Path(suite["_suite_path"]).parent
    output_root = resolve_path(result_dir, root=workdir)
    output_root.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    emit_tracking_event("benchmark_suite_start", {"suite": suite["name"], "dry_run": dry_run})
    for item in suite["benchmarks"]:
        started = time.time()
        record: dict[str, Any] = {
            "id": item["id"],
            "task": item["task"],
            "backend": item["backend"],
            "status": "dry_run" if dry_run else "pending",
        }
        if "config" in item:
            config_path = resolve_path(item["config"], root=suite_root)
            spec = load_experiment_spec(config_path)
            workflow_dir = output_root / item["id"] / "workflow"
            configured_output_dir = spec.get("output_dir")
            spec["output_dir"] = str(workflow_dir)
            record["results"] = run_experiment(spec, dry_run=dry_run, workdir=workdir)
            record["status"] = "dry_run" if dry_run else "completed"
            record["config"] = str(config_path)
            record["workflow_dir"] = str(workflow_dir)
            if configured_output_dir is not None:
                record["configured_output_dir"] = str(configured_output_dir)
        elif "command" in item:
            command = resolve_benchmark_command(item["command"])
            record["command"] = command
            if dry_run:
                record["status"] = "dry_run"
            elif not allow_commands:
                record["status"] = "skipped_external_command"
                record["message"] = "Pass allow_commands=True or --allow-commands to execute benchmark commands."
            else:
                env = os.environ.copy()
                env.update({str(key): str(value) for key, value in item.get("env", {}).items()})
                subprocess.run(command, cwd=workdir, env=env, check=True)
                record["status"] = "completed"
        record["duration_seconds"] = time.time() - started
        results.append(record)
        emit_tracking_event("benchmark_complete", record)
    payload = {
        "suite": suite["name"],
        "description": suite.get("description", ""),
        "dry_run": dry_run,
        "result_dir": str(output_root),
        "benchmarks": results,
    }
    (output_root / "benchmark_results.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return payload
