#!/usr/bin/env bash
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export ABLATION=without_anchor
exec bash "${script_dir}/train_rearrange_memory_ablation_30k_common.sh"
