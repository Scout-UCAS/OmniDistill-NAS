from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]


STAGE_REQUIRED_KEYS: dict[str, tuple[str, ...]] = {
    "bld": ("config",),
    "bld_block_library": ("records",),
    "nas_layer_importance": ("scores",),
    "mip_topk_configs": ("configs",),
    "assembled_model": ("architecture_config",),
    "gkd_model": ("architecture_config", "losses"),
}


def resolve_path(path: str | Path, root: str | Path = PROJECT_ROOT) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = Path(root) / resolved
    return resolved


def load_torch_artifact(path: str | Path) -> dict[str, Any]:
    resolved = resolve_path(path)
    try:
        payload = torch.load(resolved, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(resolved, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"expected dict artifact at {resolved}, got {type(payload)!r}")
    return payload


def load_json(path: str | Path) -> dict[str, Any]:
    resolved = resolve_path(path)
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object at {resolved}")
    return payload


def load_artifact(path: str | Path) -> dict[str, Any]:
    resolved = resolve_path(path)
    if resolved.suffix == ".json":
        return load_json(resolved)
    return load_torch_artifact(resolved)


def file_sha256(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with resolve_path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_digest(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def metadata_view(payload: dict[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    for key, value in payload.items():
        if key.endswith("state_dict"):
            metadata[key] = f"<state_dict:{len(value) if isinstance(value, dict) else 'unknown'}>"
        elif key in {"blocks", "records", "scores"} and isinstance(value, list):
            metadata[key] = f"<list:{len(value)}>"
        else:
            metadata[key] = value
    return metadata


def validate_artifact(payload: dict[str, Any], expected_stage: str | None = None) -> list[str]:
    errors: list[str] = []
    stage = str(payload.get("stage", ""))
    if expected_stage is not None and stage != expected_stage:
        errors.append(f"expected stage {expected_stage!r}, got {stage!r}")
    if not stage:
        errors.append("artifact is missing a non-empty 'stage' field")
    for key in STAGE_REQUIRED_KEYS.get(stage, ()):
        if key not in payload:
            errors.append(f"{stage} artifact is missing required key {key!r}")
    return errors


def assert_valid_artifact(payload: dict[str, Any], expected_stage: str | None = None) -> None:
    errors = validate_artifact(payload, expected_stage=expected_stage)
    if errors:
        raise ValueError("; ".join(errors))


def workflow_expected_outputs(output_dir: str | Path) -> dict[str, list[Path]]:
    root = resolve_path(output_dir)
    return {
        "bld": [root / "04_bld_block_library" / "block_library.pth"],
        "score": [root / "05_nas_layer_scoring" / "layer_importance.json"],
        "mip": [root / "06_mip_topk_architecture_configs" / "topk_architecture_configs.json"],
        "multi_objective": [root / "06_mip_topk_architecture_configs" / "multi_objective_search.json"],
        "assemble": [root / "07_model_assembly" / "assembled_model.pth"],
        "gkd": [root / "08_global_knowledge_distillation" / "gkd_model.pth"],
        "evaluate": [root / "09_evaluation" / "metrics.json"],
        "profile": [root / "10_profiling" / "profile.json"],
        "export": [root / "11_export" / "manifest.json"],
        "report": [root / "report.md"],
    }


def outputs_exist(paths: list[Path]) -> bool:
    return bool(paths) and all(path.exists() for path in paths)
