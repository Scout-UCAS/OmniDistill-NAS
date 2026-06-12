#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/00_common.sh"

print_step "Step 2: validate project code and CLI"
run_python -m compileall distill_nas_core scripts tools tests
run_python -m unittest discover -s tests

print_step "Check Qwen runner CLI"
run_python tools/run_qwen3_attention_search.py --help >/dev/null
run_python tools/run_staged_model_pipeline.py --help >/dev/null
if [[ "${REQUIRE_FLA:-0}" =~ ^(1|true|yes|on)$ ]]; then
  print_step "Check flash-linear-attention import"
  run_python - <<'PY'
import importlib

importlib.import_module("fla.layers")
print("fla_import_ok=True")
PY
fi
echo "verify_code_ok=True"
