#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/comparison_env.sh"
cd "${COMPARISON_ROOT}"
: "${CHECKPOINT:?Set CHECKPOINT to a checkpoint for the selected backend}"
: "${RMBENCH_EVAL_ROOT:?Set RMBENCH_EVAL_ROOT to your own writable RMBench simulator copy}"
resolved_root="$(realpath "${RMBENCH_EVAL_ROOT}")"
case "${resolved_root}" in
  /mnt/vepfs01/output/spidy.wang/*) ;;
  *) echo 'Evaluation must run in your own simulator copy; the harness writes files.' >&2; exit 1 ;;
esac
case "${MEMORY_BACKEND:-helios}" in
  helios) cfg=sim_robotwin_helios ;;
  memorywam) cfg=sim_robotwin_memorywam ;;
  fullkv) cfg=sim_robotwin_full_kv ;;
  *) echo 'Unknown MEMORY_BACKEND' >&2; exit 1 ;;
esac
exec python -B experiments/robotwin/eval_robotwin_single.py --config-name "${cfg}" \
  "ckpt=${CHECKPOINT}" "EVALUATION.robotwin_root=${resolved_root}" \
  "EVALUATION.task_name=${RMBENCH_TASK}" "EVALUATION.dataset_stats_path=${EVAL_STATS_PATH:-null}" \
  EVALUATION.task_config=demo_clean "EVALUATION.eval_num_episodes=${EVAL_EPISODES:-20}" "$@"
