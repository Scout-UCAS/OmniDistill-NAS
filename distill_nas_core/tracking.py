from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .artifacts import resolve_path


def tracking_event(event: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "time": datetime.now(timezone.utc).isoformat(),
        "event": event,
        "payload": payload or {},
    }


def write_jsonl_event(path: str | Path, event: str, payload: dict[str, Any] | None = None) -> Path:
    target = resolve_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(tracking_event(event, payload), default=str) + "\n")
    return target


def emit_tracking_event(event: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
    provider = os.environ.get("OMNIDISTILL_TRACKING", "none").strip().lower()
    if provider in {"", "0", "false", "none", "off"}:
        return {"provider": "none", "status": "disabled"}
    if provider == "jsonl":
        path = os.environ.get("OMNIDISTILL_TRACKING_FILE", "outputs/omnidistill_events.jsonl")
        return {"provider": "jsonl", "status": "written", "path": str(write_jsonl_event(path, event, payload))}
    if provider in {"wandb", "mlflow", "tensorboard"}:
        return {
            "provider": provider,
            "status": "not_configured",
            "message": f"Install and configure {provider} integration, or use OMNIDISTILL_TRACKING=jsonl.",
        }
    return {"provider": provider, "status": "unknown_provider"}

