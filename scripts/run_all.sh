#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"${SCRIPT_DIR}/01_prepare_environment.sh"
"${SCRIPT_DIR}/02_validate_project.sh"
"${SCRIPT_DIR}/03_smoke_tiny_nas.sh"
"${SCRIPT_DIR}/04_bld_block_library.sh"
"${SCRIPT_DIR}/05_nas_layer_importance.sh"
"${SCRIPT_DIR}/06_mip_topk_configs.sh"
"${SCRIPT_DIR}/12_multi_objective_search.sh"
"${SCRIPT_DIR}/07_assemble_model_from_config.sh"
"${SCRIPT_DIR}/08_gkd_distill.sh"
"${SCRIPT_DIR}/09_evaluate_artifact.sh"
"${SCRIPT_DIR}/10_profile_artifact.sh"
"${SCRIPT_DIR}/11_export_and_report.sh"
