#!/usr/bin/env bash
set -u
set -o pipefail

checkpoint="${CHECKPOINT:?Set CHECKPOINT to a Full Attention weight checkpoint}"
repo_root="${REPO_ROOT:-/mnt/vepfs02/output/kevin_wang/memorywam/code/fastwam_memory_fullkv}"
python_bin="${PYTHON_BIN:-/root/kevin_wang/envs/fastwam_guidemem/bin/python}"
run_tag="${RUN_TAG:-fullattention_putback_$(date +%Y%m%d_%H%M%S)}"
log_root="${LOG_ROOT:-/mnt/vepfs02/output/kevin_wang/memorywam/eval_runs/${run_tag}}"
episodes="${EVAL_NUM_EPISODES:-1}"

export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-/mnt/vepfs02/datasets/kevin_wang/code/projects/FastWAM/checkpoints}"
export DIFFSYNTH_SKIP_DOWNLOAD=true
export PYTHONUTF8=1
export LANG=C.UTF-8
export LC_ALL=C.UTF-8

if [[ ! -s "${checkpoint}" ]]; then
  echo "Missing checkpoint: ${checkpoint}" >&2
  exit 2
fi
if [[ ! "${episodes}" =~ ^[1-9][0-9]*$ ]]; then
  echo "EVAL_NUM_EPISODES must be a positive integer, got: ${episodes}" >&2
  exit 2
fi
if [[ -e "${log_root}" ]]; then
  echo "Refusing to overwrite evaluation directory: ${log_root}" >&2
  exit 2
fi

mkdir -p "${log_root}"
cd "${repo_root}" || exit 1

pids=()
for seed in 0 1 2 3 4 5 6 7; do
  output_dir="${log_root}/${run_tag}_seed${seed}"
  mkdir -p "${output_dir}"
  "${python_bin}" -u experiments/robotwin/eval_robotwin_single.py \
    --config-name sim_robotwin_full_kv \
    "ckpt=${checkpoint}" \
    "gpu_id=${seed}" \
    "seed=${seed}" \
    model.action_rope_spatial_mode=memorywam \
    EVALUATION.task_name=put_back_block \
    EVALUATION.task_config=demo_clean \
    "EVALUATION.eval_num_episodes=${episodes}" \
    "EVALUATION.output_dir=${output_dir}" \
    EVALUATION.num_inference_steps=50 \
    EVALUATION.action_delta_fraction=null \
    >"${log_root}/seed${seed}.launcher.log" 2>&1 &
  pids+=("$!")
  echo "launched seed=${seed} gpu=${seed} pid=$!"
done

failed=0
for seed in 0 1 2 3 4 5 6 7; do
  if wait "${pids[$seed]}"; then
    echo "seed=${seed} process_status=ok"
  else
    echo "seed=${seed} process_status=failed"
    failed=$((failed + 1))
  fi
done

summary="${log_root}/summary.txt"
{
  echo "checkpoint=${checkpoint}"
  echo "episodes_per_seed=${episodes}"
  for seed in 0 1 2 3 4 5 6 7; do
    result="$(grep -a 'Success rate:' "${log_root}/seed${seed}.launcher.log" | tail -n 1 || true)"
    echo "seed=${seed} ${result:-missing_result}"
  done
  echo "failed_processes=${failed}"
} | tee "${summary}"

exit "${failed}"
