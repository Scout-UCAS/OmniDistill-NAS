from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from distill_nas_core.artifacts import workflow_expected_outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Show workflow artifact status.")
    parser.add_argument("--workflow-dir", default="outputs/distill_nas_workflow")
    parser.add_argument("--json", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    payload = {
        stage: {
            "complete": all(path.exists() for path in paths),
            "outputs": [str(path) for path in paths],
        }
        for stage, paths in workflow_expected_outputs(args.workflow_dir).items()
    }
    if args.json:
        print(json.dumps(payload, indent=2))
        return
    for stage, item in payload.items():
        marker = "ok" if item["complete"] else "missing"
        print(f"{stage:10s} {marker}")


if __name__ == "__main__":
    main()
