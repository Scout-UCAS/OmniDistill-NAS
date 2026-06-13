from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from distill_nas_core.cli import add_run_arguments, run_command


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an OmniDistill-NAS experiment spec with resume support.")
    add_run_arguments(parser)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    raise SystemExit(run_command(args))


if __name__ == "__main__":
    main()
