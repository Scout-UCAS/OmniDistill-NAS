from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from distill_nas_core.export import export_artifact


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export an OmniDistill-NAS artifact directory.")
    parser.add_argument("--artifact-pth", required=True)
    parser.add_argument("--export-dir", required=True)
    parser.add_argument("--no-state-dict", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    manifest = export_artifact(args.artifact_pth, args.export_dir, include_state_dict=not args.no_state_dict)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
