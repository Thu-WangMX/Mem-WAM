#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export PHYSICAL_TASK_ID=rearrange_blocks
export PHYSICAL_TASK_CONFIG=rmbench_rearrange_blocks_physical_settle_rate_debt_k8_scratch_joint_40k_lr2e4
exec bash "${script_dir}/train_physical_settle_rate_debt_t7c5_k8_scratch_joint40k_common.sh"
