from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from distill_nas_core.evaluation import evaluate_artifact


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate an OmniDistill-NAS artifact.")
    parser.add_argument("--artifact-pth", required=True)
    parser.add_argument("--backend", default="auto")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-batches", type=int, default=4)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--output-json", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    metrics = evaluate_artifact(
        args.artifact_pth,
        backend=args.backend,
        device=args.device,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        num_batches=args.num_batches,
        seed=args.seed,
    )
    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"wrote_metrics_json={output}")


if __name__ == "__main__":
    main()
