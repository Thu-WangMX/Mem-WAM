#!/usr/bin/env bash
set -euo pipefail

repo_root="${REPO_ROOT:-/mnt/vepfs02/output/kevin_wang/memorywam/code/fastwam_memory_fullkv}"
python_bin="${PYTHON_BIN:-/root/kevin_wang/envs/fastwam_guidemem/bin/python}"
rmbench_root="${RMBENCH_ROOT:-/mnt/vepfs02/datasets/kevin_wang/code/projects/RMBench}"
output_root="${OUTPUT_ROOT:-/mnt/vepfs02/output/kevin_wang/memorywam/data/rmbench_lerobot_v21_rgb_v3}"

export PYTHONPATH="${repo_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUTF8=1
export LANG=C.UTF-8
export LC_ALL=C.UTF-8

mkdir -p "${output_root}"
cd "${repo_root}"
exec "${python_bin}" scripts/build_rmbench_putback_rgb_v3.py \
  --rmbench-root "${rmbench_root}" \
  --output-root "${output_root}" \
  "$@"
