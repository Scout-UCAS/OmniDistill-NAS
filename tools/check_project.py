from __future__ import annotations

import argparse
import subprocess
import sys


def run(command: list[str]) -> None:
    print("+", " ".join(command))
    subprocess.run(command, check=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run local CI checks for OmniDistill-NAS.")
    parser.add_argument("--workflow", action="store_true", help="Also run the default staged toy workflow.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run([sys.executable, "-m", "compileall", "distill_nas_core", "scripts", "tools", "tests"])
    run([sys.executable, "-m", "pytest", "-q"])
    for script in [
        "tools/run_qwen3_attention_search.py",
        "tools/run_staged_model_pipeline.py",
        "tools/run_staged_toy_pipeline.py",
        "tools/run_experiment.py",
        "tools/run_multi_objective_search.py",
        "tools/evaluate_artifact.py",
        "tools/export_artifact.py",
        "tools/profile_artifact.py",
        "tools/generate_report.py",
        "tools/workflow_status.py",
    ]:
        run([sys.executable, script, "--help"])
    if args.workflow:
        run(["bash", "scripts/run_all.sh"])


if __name__ == "__main__":
    main()
