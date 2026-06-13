from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .artifacts import PROJECT_ROOT, outputs_exist, resolve_path, workflow_expected_outputs


DEFAULT_STAGES = (
    "prepare",
    "validate",
    "smoke",
    "bld",
    "score",
    "mip",
    "multi_objective",
    "assemble",
    "gkd",
    "evaluate",
    "profile",
    "export",
    "report",
)


@dataclass(frozen=True)
class StagePlan:
    name: str
    command: list[str]
    outputs: list[Path] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    skippable: bool = True


def load_experiment_spec(path: str | Path) -> dict[str, Any]:
    resolved = resolve_path(path)
    text = resolved.read_text(encoding="utf-8")
    if resolved.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore[import-untyped]
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise RuntimeError("YAML experiment specs require PyYAML") from exc
        payload = yaml.safe_load(text)
    else:
        payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("experiment spec must be a mapping/object")
    return payload


def write_experiment_spec(path: str | Path, payload: dict[str, Any]) -> Path:
    resolved = resolve_path(path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return resolved


def _bool_env(value: Any) -> str:
    return "1" if bool(value) else "0"


def experiment_workspace(spec: dict[str, Any], workdir: str | Path | None = None) -> Path:
    workspace = workdir if workdir is not None else spec.get("workdir", ".")
    return resolve_path(workspace)


def _resolve_spec_path(path: Any, workspace: Path) -> Path:
    return resolve_path(path, root=workspace)


def spec_env(spec: dict[str, Any], output_dir: str | Path | None = None, workdir: str | Path | None = None) -> dict[str, str]:
    env: dict[str, str] = {}
    env.update({str(key): str(value) for key, value in spec.get("env", {}).items()})
    workspace = experiment_workspace(spec, workdir=workdir)
    resolved_output_dir = resolve_path(output_dir or spec.get("output_dir", "outputs/distill_nas_workflow"), root=workspace)
    env.setdefault("WORKFLOW_WORKDIR", str(workspace))
    env.setdefault("WORKFLOW_OUTPUT_DIR", str(resolved_output_dir))
    env.setdefault("WORKFLOW_BACKEND", str(spec.get("backend", "toy")))
    for key, value in spec.get("model", {}).items():
        env.setdefault(str(key).upper().replace("-", "_"), str(value))
    for key, value in spec.get("search", {}).items():
        env.setdefault(str(key).upper().replace("-", "_"), str(value))
    for key, value in spec.get("distillation", {}).items():
        env.setdefault(str(key).upper().replace("-", "_"), str(value))
    devices = spec.get("devices", {})
    if isinstance(devices, dict):
        for key in ("device", "teacher_device", "student_device"):
            if key in devices:
                env.setdefault(key.upper(), str(devices[key]))
    distributed = spec.get("distributed", {})
    if isinstance(distributed, dict):
        if "gradient_accumulation_steps" in distributed:
            env.setdefault("GRADIENT_ACCUMULATION_STEPS", str(distributed["gradient_accumulation_steps"]))
        if "use_accelerate" in distributed:
            env.setdefault("USE_ACCELERATE", _bool_env(distributed["use_accelerate"]))
        if "accelerate_config" in distributed:
            env.setdefault("ACCELERATE_CONFIG", str(distributed["accelerate_config"]))
    return env


def _python_cmd(script: str, *args: str) -> list[str]:
    return [sys.executable, str(PROJECT_ROOT / script), *args]


def _script_cmd(script: str) -> list[str]:
    return ["bash", str(PROJECT_ROOT / "scripts" / script)]


def build_stage_plan(spec: dict[str, Any], workdir: str | Path | None = None) -> list[StagePlan]:
    workspace = experiment_workspace(spec, workdir=workdir)
    output_dir = resolve_path(spec.get("output_dir", "outputs/distill_nas_workflow"), root=workspace)
    expected = workflow_expected_outputs(output_dir)
    env = spec_env(spec, output_dir=output_dir, workdir=workspace)
    evaluation = spec.get("evaluation", {})
    profiling = spec.get("profiling", {})
    artifact = str(
        _resolve_spec_path(evaluation.get("artifact_pth"), workspace)
        if isinstance(evaluation, dict) and evaluation.get("artifact_pth")
        else output_dir / "08_global_knowledge_distillation" / "gkd_model.pth"
    )
    profile_artifact = str(
        _resolve_spec_path(profiling.get("artifact_pth"), workspace)
        if isinstance(profiling, dict) and profiling.get("artifact_pth")
        else artifact
    )

    stages: dict[str, StagePlan] = {
        "prepare": StagePlan("prepare", _script_cmd("01_prepare_environment.sh"), skippable=False, env=env),
        "validate": StagePlan("validate", _script_cmd("02_validate_project.sh"), skippable=False, env=env),
        "smoke": StagePlan("smoke", _script_cmd("03_smoke_tiny_nas.sh"), skippable=False, env=env),
        "bld": StagePlan("bld", _script_cmd("04_bld_block_library.sh"), expected["bld"], env=env),
        "score": StagePlan("score", _script_cmd("05_nas_layer_importance.sh"), expected["score"], env=env),
        "mip": StagePlan("mip", _script_cmd("06_mip_topk_configs.sh"), expected["mip"], env=env),
        "multi_objective": StagePlan(
            "multi_objective",
            _python_cmd(
                "tools/run_multi_objective_search.py",
                "--scores-json",
                str(output_dir / "05_nas_layer_scoring" / "layer_importance.json"),
                "--output-json",
                str(expected["multi_objective"][0]),
                "--config-dir",
                str(output_dir / "06_mip_topk_architecture_configs" / "pareto_configs"),
                "--report-md",
                str(output_dir / "06_mip_topk_architecture_configs" / "multi_objective_report.md"),
                "--plot-svg",
                str(output_dir / "06_mip_topk_architecture_configs" / "pareto_front.svg"),
            ),
            expected["multi_objective"],
            env=env,
        ),
        "assemble": StagePlan("assemble", _script_cmd("07_assemble_model_from_config.sh"), expected["assemble"], env=env),
        "gkd": StagePlan("gkd", _script_cmd("08_gkd_distill.sh"), expected["gkd"], env=env),
        "evaluate": StagePlan(
            "evaluate",
            _python_cmd(
                "tools/evaluate_artifact.py",
                "--artifact-pth",
                artifact,
                "--output-json",
                str(expected["evaluate"][0]),
            ),
            expected["evaluate"],
            env=env,
        ),
        "profile": StagePlan(
            "profile",
            _python_cmd(
                "tools/profile_artifact.py",
                "--artifact-pth",
                profile_artifact,
                "--output-json",
                str(expected["profile"][0]),
            ),
            expected["profile"],
            env=env,
        ),
        "export": StagePlan(
            "export",
            _python_cmd(
                "tools/export_artifact.py",
                "--artifact-pth",
                artifact,
                "--export-dir",
                str(output_dir / "11_export"),
            ),
            expected["export"],
            env=env,
        ),
        "report": StagePlan(
            "report",
            _python_cmd(
                "tools/generate_report.py",
                "--workflow-dir",
                str(output_dir),
                "--output-md",
                str(expected["report"][0]),
            ),
            expected["report"],
            env=env,
        ),
    }

    requested = spec.get("stages", DEFAULT_STAGES)
    if isinstance(requested, str):
        requested = [item.strip() for item in requested.split(",") if item.strip()]
    plan: list[StagePlan] = []
    for stage_name in requested:
        if stage_name not in stages:
            raise ValueError(f"unknown experiment stage: {stage_name}")
        plan.append(stages[stage_name])
    return plan


def filter_plan_from_stage(plan: list[StagePlan], from_stage: str | None, only_stage: str | None) -> list[StagePlan]:
    if only_stage is not None:
        return [stage for stage in plan if stage.name == only_stage]
    if from_stage is None:
        return plan
    seen = False
    filtered: list[StagePlan] = []
    for stage in plan:
        if stage.name == from_stage:
            seen = True
        if seen:
            filtered.append(stage)
    if not seen:
        raise ValueError(f"stage {from_stage!r} is not in the plan")
    return filtered


def run_stage(stage: StagePlan, force: bool = False, dry_run: bool = False, cwd: str | Path | None = None) -> str:
    if dry_run:
        return "dry_run"
    if stage.skippable and not force and outputs_exist(stage.outputs):
        return "skipped"
    env = os.environ.copy()
    env.update(stage.env)
    subprocess.run(stage.command, cwd=resolve_path(cwd or "."), env=env, check=True)
    return "completed"


def run_experiment(
    spec: dict[str, Any],
    force: bool = False,
    dry_run: bool = False,
    from_stage: str | None = None,
    only_stage: str | None = None,
    workdir: str | Path | None = None,
) -> list[dict[str, Any]]:
    workspace = experiment_workspace(spec, workdir=workdir)
    workspace.mkdir(parents=True, exist_ok=True)
    plan = filter_plan_from_stage(build_stage_plan(spec, workdir=workspace), from_stage=from_stage, only_stage=only_stage)
    results: list[dict[str, Any]] = []
    for stage in plan:
        status = run_stage(stage, force=force, dry_run=dry_run, cwd=workspace)
        results.append(
            {
                "stage": stage.name,
                "status": status,
                "command": stage.command,
                "outputs": [str(path) for path in stage.outputs],
            }
        )
    return results


def plan_to_dict(plan: list[StagePlan]) -> list[dict[str, Any]]:
    return [
        {
            **asdict(stage),
            "outputs": [str(path) for path in stage.outputs],
        }
        for stage in plan
    ]
