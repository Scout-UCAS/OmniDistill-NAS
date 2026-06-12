from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from distill_nas_core.profiler import profile_artifact


def parse_batch_sizes(value: str) -> list[int]:
    return [int(item) for item in value.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Profile an OmniDistill-NAS artifact.")
    parser.add_argument("--artifact-pth", required=True)
    parser.add_argument("--backend", default="auto")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-sizes", default="1,2,4")
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--output-json", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    profile = profile_artifact(
        args.artifact_pth,
        backend=args.backend,
        device=args.device,
        batch_sizes=parse_batch_sizes(args.batch_sizes),
        seq_len=args.seq_len,
        warmup=args.warmup,
        steps=args.steps,
    )
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    print(f"wrote_profile_json={output}")


if __name__ == "__main__":
    main()
