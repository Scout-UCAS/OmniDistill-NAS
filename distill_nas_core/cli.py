from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .artifacts import workflow_expected_outputs
from .experiment import build_stage_plan, load_experiment_spec, plan_to_dict, run_experiment


def _write_json(path: str | Path, payload: object) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return target


def add_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, help="JSON/YAML experiment spec.")
    parser.add_argument("--stage", default=None, help="Run only one named stage from the spec.")
    parser.add_argument("--from-stage", default=None, help="Resume from this stage onward.")
    parser.add_argument("--workdir", default=None, help="Workspace root for relative outputs and caches.")
    parser.add_argument("--force", action="store_true", help="Run stages even when their outputs already exist.")
    parser.add_argument("--dry-run", action="store_true", help="Print runnable actions without executing them.")
    parser.add_argument("--print-plan", action="store_true", help="Print the resolved stage plan as JSON.")
    parser.add_argument("--results-json", default=None, help="Optional path for run results JSON.")


def run_command(args: argparse.Namespace) -> int:
    spec = load_experiment_spec(args.config)
    if args.print_plan:
        print(json.dumps(plan_to_dict(build_stage_plan(spec, workdir=args.workdir)), indent=2, default=str))
        return 0
    results = run_experiment(
        spec,
        force=args.force,
        dry_run=args.dry_run,
        from_stage=args.from_stage,
        only_stage=args.stage,
        workdir=args.workdir,
    )
    print(json.dumps(results, indent=2, default=str))
    if args.results_json:
        _write_json(args.results_json, results)
    return 0


def status_command(args: argparse.Namespace) -> int:
    payload = {
        stage: {
            "complete": all(path.exists() for path in paths),
            "outputs": [str(path) for path in paths],
        }
        for stage, paths in workflow_expected_outputs(args.workflow_dir).items()
    }
    if args.json:
        print(json.dumps(payload, indent=2))
        return 0
    for stage, item in payload.items():
        marker = "ok" if item["complete"] else "missing"
        print(f"{stage:16s} {marker}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="omnidistill",
        description="OmniDistill-NAS command line interface.",
    )
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run an experiment spec with resume support.")
    add_run_arguments(run_parser)
    run_parser.set_defaults(func=run_command)

    plan_parser = subparsers.add_parser("plan", help="Print the resolved stage plan for an experiment spec.")
    add_run_arguments(plan_parser)
    plan_parser.set_defaults(print_plan=True, dry_run=True, func=run_command)

    status_parser = subparsers.add_parser("status", help="Show workflow artifact status.")
    status_parser.add_argument("--workflow-dir", default="outputs/distill_nas_workflow")
    status_parser.add_argument("--json", action="store_true")
    status_parser.set_defaults(func=status_command)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        parser.print_help()
        return 0
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
