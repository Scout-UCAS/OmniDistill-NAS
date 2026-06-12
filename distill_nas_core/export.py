from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from .artifacts import file_sha256, load_torch_artifact, metadata_view, resolve_path, stable_json_digest


EXPORT_FORMAT_VERSION = 1


def export_artifact(
    artifact_pth: str | Path,
    export_dir: str | Path,
    include_state_dict: bool = True,
) -> dict[str, Any]:
    source = resolve_path(artifact_pth)
    target_dir = resolve_path(export_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    payload = load_torch_artifact(source)
    artifact_copy = target_dir / source.name
    if source.resolve() != artifact_copy.resolve():
        shutil.copy2(source, artifact_copy)

    if "architecture_config" in payload:
        (target_dir / "architecture_config.json").write_text(
            json.dumps(payload["architecture_config"], indent=2, default=str),
            encoding="utf-8",
        )
    if "config" in payload:
        (target_dir / "model_config.json").write_text(
            json.dumps(payload["config"], indent=2, default=str),
            encoding="utf-8",
        )
    if include_state_dict and "model_state_dict" in payload:
        torch.save(payload["model_state_dict"], target_dir / "model_state_dict.pth")

    manifest = {
        "format": "omnidistill-nas-export",
        "format_version": EXPORT_FORMAT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_artifact": str(source),
        "artifact_file": artifact_copy.name,
        "artifact_sha256": file_sha256(source),
        "stage": payload.get("stage"),
        "backend": payload.get("backend", "toy"),
        "metadata_digest": stable_json_digest(metadata_view(payload)),
        "has_model_state_dict": "model_state_dict" in payload,
        "has_architecture_config": "architecture_config" in payload,
    }
    (target_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (target_dir / "README.md").write_text(_export_readme(manifest), encoding="utf-8")
    return manifest


def _export_readme(manifest: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# OmniDistill-NAS Export",
            "",
            f"- Stage: `{manifest.get('stage')}`",
            f"- Backend: `{manifest.get('backend')}`",
            f"- Artifact: `{manifest.get('artifact_file')}`",
            f"- SHA256: `{manifest.get('artifact_sha256')}`",
            "",
            "This directory contains a portable artifact manifest, the source artifact copy,",
            "and any extracted architecture/model config files available in the artifact.",
            "",
        ]
    )


def load_export_manifest(export_dir: str | Path) -> dict[str, Any]:
    path = resolve_path(export_dir) / "manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("format") != "omnidistill-nas-export":
        raise ValueError(f"not an OmniDistill-NAS export manifest: {path}")
    return payload
