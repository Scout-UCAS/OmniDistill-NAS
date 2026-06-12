from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from distill_nas_core.reporting import write_workflow_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate a Markdown report from workflow artifacts.")
    parser.add_argument("--workflow-dir", required=True)
    parser.add_argument("--output-md", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    output = write_workflow_report(args.workflow_dir, output_md=args.output_md)
    print(f"wrote_report_md={output}")


if __name__ == "__main__":
    main()
