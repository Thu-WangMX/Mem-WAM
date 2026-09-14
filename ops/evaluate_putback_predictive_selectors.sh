#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/mnt/vepfs02/output/kevin_wang/memorywam/code/fastwam_dynamic_multiframe_selection}"
PYTHON_BIN="${PYTHON_BIN:-/root/kevin_wang/envs/fastwam_guidemem/bin/python}"
COMPARISON_JSON="${COMPARISON_JSON:?set the immutable development comparison JSON}"
LOCKED_CANDIDATE="${LOCKED_CANDIDATE:?set a new locked-candidate JSON path}"

cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}"
"${PYTHON_BIN}" scripts/evaluate_putback_predictive_selectors.py \
  --comparison-json "${COMPARISON_JSON}" \
  --locked-candidate "${LOCKED_CANDIDATE}"

