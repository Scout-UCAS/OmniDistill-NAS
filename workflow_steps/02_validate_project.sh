#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00_common.sh"

print_step "Step 2: validate project code and CLI"
run_python -m compileall distill_nas_core scripts test_suite
run_python -m unittest discover -s test_suite

print_step "Check Qwen runner CLI"
run_python scripts/run_qwen3_attention_search.py --help >/dev/null
echo "verify_code_ok=True"
