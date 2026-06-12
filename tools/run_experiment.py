from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from distill_nas_core.experiment import build_stage_plan, load_experiment_spec, plan_to_dict, run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an OmniDistill-NAS experiment spec with resume support.")
    parser.add_argument("--config", required=True, help="JSON/YAML experiment spec.")
    parser.add_argument("--stage", default=None, help="Run only one named stage from the spec.")
    parser.add_argument("--from-stage", default=None, help="Resume from this stage onward.")
    parser.add_argument("--force", action="store_true", help="Run stages even when their outputs already exist.")
    parser.add_argument("--dry-run", action="store_true", help="Print runnable actions without executing them.")
    parser.add_argument("--print-plan", action="store_true", help="Print the resolved stage plan as JSON.")
    parser.add_argument("--results-json", default=None, help="Optional path for run results JSON.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    spec = load_experiment_spec(args.config)
    if args.print_plan:
        print(json.dumps(plan_to_dict(build_stage_plan(spec)), indent=2))
        return
    results = run_experiment(
        spec,
        force=args.force,
        dry_run=args.dry_run,
        from_stage=args.from_stage,
        only_stage=args.stage,
    )
    print(json.dumps(results, indent=2))
    if args.results_json:
        path = Path(args.results_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
