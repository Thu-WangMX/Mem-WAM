#!/usr/bin/env bash
set -euo pipefail

repo_root="${REPO_ROOT:-/mnt/vepfs02/output/kevin_wang/memorywam/code/fastwam_dynamic_surprise_memory}"
export REPO_ROOT="${repo_root}"

exec bash "${repo_root}/ops/train_putback_dynamic_surprise_k8_40k.sh" "$@"
