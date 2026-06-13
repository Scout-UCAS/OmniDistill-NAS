from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .artifacts import workflow_expected_outputs
from .benchmarks import benchmark_plan, load_benchmark_suite, run_benchmark_suite
from .experiment import build_stage_plan, load_experiment_spec, plan_to_dict, run_experiment
from .plugins import list_plugins
from .reporting import write_workflow_report
from .result_zoo import write_result_index
from .schema import (
    raise_if_errors,
    validate_benchmark_suite,
    validate_experiment_spec,
    validate_json_file,
    validate_result_manifest,
)


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
    raise_if_errors(validate_experiment_spec(spec))
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


def init_command(args: argparse.Namespace) -> int:
    target = Path(args.output)
    if target.exists() and not args.force:
        raise FileExistsError(f"{target} already exists; pass --force to overwrite")
    spec = {
        "name": args.name,
        "backend": args.backend,
        "output_dir": args.output_dir,
        "stages": [
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
        ],
        "model": {"seq_len": 16, "batch_size": 2, "num_batches": 4},
        "search": {
            "attention_variants": "parent_attn,mha_attn,quant_mha_attn,mqa_attn,gqa_kv2,linear_attn,noop_attn",
            "layer_variants": "parent,skip_attn,skip_mlp,skip_both",
            "bld_steps": 1,
            "score_batches": 2,
            "batch_sizes": "1,2,4",
            "top_k": 3,
        },
        "distillation": {"gkd_steps": 2, "opd_weight": 0.0},
        "devices": {"device": "auto"},
        "distributed": {"gradient_accumulation_steps": 1, "use_accelerate": False},
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(spec, indent=2), encoding="utf-8")
    print(f"wrote_config={target}")
    return 0


def validate_command(args: argparse.Namespace) -> int:
    if args.kind == "auto":
        payload = json.loads(Path(args.path).read_text(encoding="utf-8"))
        if "benchmarks" in payload:
            errors = validate_benchmark_suite(payload)
            kind = "benchmark"
        elif "metrics" in payload:
            errors = validate_result_manifest(payload)
            kind = "result"
        else:
            errors = validate_experiment_spec(payload)
            kind = "experiment"
    else:
        errors = validate_json_file(args.path, args.kind)
        kind = args.kind
    raise_if_errors(errors)
    print(json.dumps({"path": args.path, "kind": kind, "valid": True}, indent=2))
    return 0


def benchmark_command(args: argparse.Namespace) -> int:
    if args.print_plan:
        suite = load_benchmark_suite(args.suite)
        print(json.dumps(benchmark_plan(suite), indent=2))
        return 0
    payload = run_benchmark_suite(
        args.suite,
        result_dir=args.result_dir,
        dry_run=args.dry_run,
        workdir=args.workdir,
        allow_commands=args.allow_commands,
    )
    print(json.dumps(payload, indent=2, default=str))
    return 0


def report_command(args: argparse.Namespace) -> int:
    if args.workflow_dir:
        path = write_workflow_report(args.workflow_dir, args.output_md)
    else:
        path = write_result_index(args.results_dir, args.output_md)
    print(f"wrote_report={path}")
    return 0


def plugins_command(args: argparse.Namespace) -> int:
    print(json.dumps(list_plugins(category=args.category), indent=2))
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

    init_parser = subparsers.add_parser("init", help="Create a starter experiment config.")
    init_parser.add_argument("--output", default="configs/experiment.json")
    init_parser.add_argument("--name", default="omnidistill-experiment")
    init_parser.add_argument("--backend", default="toy")
    init_parser.add_argument("--output-dir", default="outputs/distill_nas_workflow")
    init_parser.add_argument("--force", action="store_true")
    init_parser.set_defaults(func=init_command)

    run_parser = subparsers.add_parser("run", help="Run an experiment spec with resume support.")
    add_run_arguments(run_parser)
    run_parser.set_defaults(func=run_command)

    plan_parser = subparsers.add_parser("plan", help="Print the resolved stage plan for an experiment spec.")
    add_run_arguments(plan_parser)
    plan_parser.set_defaults(print_plan=True, dry_run=True, func=run_command)

    validate_parser = subparsers.add_parser("validate", help="Validate experiment, benchmark, or result JSON.")
    validate_parser.add_argument("path")
    validate_parser.add_argument("--kind", choices=["auto", "experiment", "benchmark", "result"], default="auto")
    validate_parser.set_defaults(func=validate_command)

    benchmark_parser = subparsers.add_parser("benchmark", help="Run or dry-run a benchmark suite.")
    benchmark_parser.add_argument("--suite", default="benchmarks/suites/toy_smoke.json")
    benchmark_parser.add_argument("--result-dir", default="benchmark_runs")
    benchmark_parser.add_argument("--workdir", default=None)
    benchmark_parser.add_argument("--dry-run", action="store_true")
    benchmark_parser.add_argument("--allow-commands", action="store_true")
    benchmark_parser.add_argument("--print-plan", action="store_true")
    benchmark_parser.set_defaults(func=benchmark_command)

    report_parser = subparsers.add_parser("report", help="Generate workflow or result-zoo reports.")
    report_parser.add_argument("--workflow-dir", default=None)
    report_parser.add_argument("--results-dir", default="results")
    report_parser.add_argument("--output-md", default=None)
    report_parser.set_defaults(func=report_command)

    plugins_parser = subparsers.add_parser("plugins", help="List registered extension plugins.")
    plugins_parser.add_argument("--category", default=None)
    plugins_parser.set_defaults(func=plugins_command)

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
