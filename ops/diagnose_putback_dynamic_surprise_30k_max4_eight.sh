#!/usr/bin/env bash
set -euo pipefail

repo_root="${REPO_ROOT:-/mnt/vepfs02/output/kevin_wang/memorywam/code/fastwam_dynamic_surprise_memory}"
python_bin="${PYTHON_BIN:-/root/kevin_wang/envs/fastwam_guidemem/bin/python}"
checkpoint="${CHECKPOINT:-/mnt/vepfs02/output/kevin_wang/memorywam/train/dynamic_surprise_memory_putback_k8_25k_seed42/checkpoints/weights/step_030000.pt}"
run_tag="${RUN_TAG:-dynamic_surprise_30k_rmbench8_max4_diag_$(date +%Y%m%d_%H%M%S)}"
log_root="${LOG_ROOT:-/mnt/vepfs02/output/kevin_wang/memorywam/eval_runs/${run_tag}}"

scene_seeds=(100000 200000 300000 400000 1300000 1400000 1500000 1600000)
policy_seeds=(1000 1002 1004 1006 1008 1010 1012 1014)

export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-/mnt/vepfs02/output/kevin_wang/memorywam/model_assets/diffsynth_official}"
export DIFFSYNTH_SKIP_DOWNLOAD=true
export PYTHONPATH="${repo_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUTF8=1 LANG=C.UTF-8 LC_ALL=C.UTF-8

cd "${repo_root}"
mkdir -p "${log_root}"
pids=()
for gpu in {0..7}; do
  scene_seed="${scene_seeds[$gpu]}"
  policy_seed="${policy_seeds[$gpu]}"
  output_dir="${log_root}/scene${scene_seed}_policy${policy_seed}"
  mkdir -p "${output_dir}"
  "${python_bin}" -u experiments/robotwin/eval_robotwin_single.py \
    --config-name sim_robotwin_dynamic_surprise \
    "ckpt=${checkpoint}" \
    "gpu_id=${gpu}" \
    "seed=${policy_seed}" \
    "EVALUATION.start_scene_seed=${scene_seed}" \
    "EVALUATION.task_name=put_back_block" \
    "EVALUATION.task_config=demo_clean" \
    "EVALUATION.eval_num_episodes=1" \
    "EVALUATION.dynamic_surprise_max_segment=4" \
    "EVALUATION.output_dir=${output_dir}" \
    >"${log_root}/scene${scene_seed}_policy${policy_seed}.log" 2>&1 &
  pids+=("$!")
  echo "launched scene=${scene_seed} policy=${policy_seed} gpu=${gpu} pid=$!"
done

failed=0
for pid in "${pids[@]}"; do
  wait "${pid}" || failed=$((failed + 1))
done
echo "log_root=${log_root} failed=${failed}"
exit "${failed}"
