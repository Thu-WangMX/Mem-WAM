#!/usr/bin/env bash
set -u
set -o pipefail

repo_root="${REPO_ROOT:-/mnt/vepfs02/output/kevin_wang/memorywam/code/fastwam_memory_fullkv}"
python_bin="${PYTHON_BIN:-/root/kevin_wang/envs/fastwam_guidemem/bin/python}"
model_base="${DIFFSYNTH_MODEL_BASE_PATH:-/mnt/vepfs02/output/kevin_wang/memorywam/model_assets/diffsynth_official}"
checkpoint="${CHECKPOINT:-/mnt/vepfs02/output/kevin_wang/memorywam/train/fastwam_memorywam_fullkv_putback_frame0_b1x8_2k_seed42_20260730_v2/checkpoints/weights/step_002000.pt}"
run_tag="${RUN_TAG:-memorywam_aligned_fullkv_$(date +%Y%m%d_%H%M%S)}"
log_root="${LOG_ROOT:-/mnt/vepfs02/output/kevin_wang/memorywam/eval_runs/${run_tag}}"
action_delta_fraction="${ACTION_DELTA_FRACTION:-null}"

export DIFFSYNTH_MODEL_BASE_PATH="${model_base}"
export DIFFSYNTH_SKIP_DOWNLOAD=true
export PYTHONUTF8=1
export LANG=C.UTF-8
export LC_ALL=C.UTF-8

mkdir -p "${log_root}"
cd "${repo_root}" || exit 1

pids=()
for seed in 0 1 2 3 4; do
  gpu="${seed}"
  # eval_robotwin_single uses only the output directory basename as its
  # result tag, so include the run tag to prevent collisions with older runs.
  output_dir="${log_root}/${run_tag}_seed${seed}"
  mkdir -p "${output_dir}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${python_bin}" -u \
    experiments/robotwin/eval_robotwin_single.py \
    --config-name sim_robotwin_full_kv \
    "ckpt=${checkpoint}" \
    "gpu_id=${gpu}" \
    "seed=${seed}" \
    "EVALUATION.task_name=put_back_block" \
    "EVALUATION.task_config=demo_clean" \
    "EVALUATION.eval_num_episodes=1" \
    "EVALUATION.output_dir=${output_dir}" \
    "EVALUATION.action_delta_fraction=${action_delta_fraction}" \
    >"${log_root}/seed${seed}.launcher.log" 2>&1 &
  pids+=("$!")
  echo "launched seed=${seed} gpu=${gpu} pid=$!"
done

failed=0
for index in 0 1 2 3 4; do
  if wait "${pids[$index]}"; then
    echo "seed=${index} process_status=ok"
  else
    echo "seed=${index} process_status=failed"
    failed=$((failed + 1))
  fi
done

echo "run_tag=${run_tag}"
echo "log_root=${log_root}"
for seed in 0 1 2 3 4; do
  echo "===== seed=${seed} ====="
  grep -E "success|Success|episode|Episode|Evaluation finished" \
    "${log_root}/seed${seed}.launcher.log" | tail -n 12 || true
done

exit "${failed}"
