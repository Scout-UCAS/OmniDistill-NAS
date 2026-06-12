from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .artifacts import resolve_path


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def generate_workflow_report(workflow_dir: str | Path) -> str:
    root = resolve_path(workflow_dir)
    summaries = {
        "BLD": _read_json(root / "04_bld_block_library" / "summary.json"),
        "NAS Scoring": _read_json(root / "05_nas_layer_scoring" / "layer_importance.json"),
        "MIP": _read_json(root / "06_mip_topk_architecture_configs" / "topk_architecture_configs.json"),
        "Multi-Objective": _read_json(root / "06_mip_topk_architecture_configs" / "multi_objective_search.json"),
        "Assembly": _read_json(root / "07_model_assembly" / "summary.json"),
        "GKD": _read_json(root / "08_global_knowledge_distillation" / "summary.json"),
        "Evaluation": _read_json(root / "09_evaluation" / "metrics.json"),
        "Profiling": _read_json(root / "10_profiling" / "profile.json"),
        "Export": _read_json(root / "11_export" / "manifest.json"),
    }
    lines = ["# OmniDistill-NAS Run Report", "", f"Workflow directory: `{root}`", ""]
    for title, payload in summaries.items():
        lines.extend([f"## {title}", ""])
        if payload is None:
            lines.extend(["Not available.", ""])
            continue
        stage = payload.get("stage") or payload.get("format") or title
        lines.append(f"- Stage: `{stage}`")
        for key in (
            "pth",
            "num_records",
            "num_blocks",
            "num_losses",
            "first_loss",
            "last_loss",
            "lm_loss",
            "perplexity",
            "parameter_memory_bytes",
            "artifact_sha256",
            "num_sweep_solutions",
            "num_pareto_solutions",
            "pareto_source",
        ):
            if key in payload:
                lines.append(f"- {key}: `{payload[key]}`")
        if "configs" in payload and isinstance(payload["configs"], list):
            lines.append(f"- configs: `{len(payload['configs'])}`")
        if "selected" in payload and isinstance(payload["selected"], list):
            lines.append(f"- selected: `{', '.join(map(str, payload['selected'][:8]))}`")
        lines.append("")
    return "\n".join(lines)


def write_workflow_report(workflow_dir: str | Path, output_md: str | Path | None = None) -> Path:
    root = resolve_path(workflow_dir)
    target = resolve_path(output_md) if output_md is not None else root / "report.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(generate_workflow_report(root), encoding="utf-8")
    return target
