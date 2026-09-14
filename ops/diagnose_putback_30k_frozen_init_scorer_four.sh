#!/usr/bin/env bash
set -euo pipefail

repo_root="${REPO_ROOT:-/mnt/vepfs02/output/kevin_wang/memorywam/code/fastwam_dynamic_surprise_memory}"
python_bin="${PYTHON_BIN:-/root/kevin_wang/envs/fastwam_guidemem/bin/python}"
checkpoint="${CHECKPOINT:-/mnt/vepfs02/output/kevin_wang/memorywam/train/dynamic_surprise_memory_putback_k8_25k_seed42/checkpoints/weights/step_030000.pt}"
run_tag="${RUN_TAG:-dynamic_surprise_30k_frozen_init_scorer_rmbench4_$(date +%Y%m%d_%H%M%S)}"
log_root="${LOG_ROOT:-/mnt/vepfs02/output/kevin_wang/memorywam/eval_runs/${run_tag}}"
preflight_only="${PREFLIGHT_ONLY:-0}"
gpus_text="${GPUS:-4,5,6,7}"

scene_seeds=(100000 200000 300000 400000)
policy_seeds=(1000 1002 1004 1006)
read -r -a gpus <<<"${gpus_text//,/ }"

[[ -d "${repo_root}" ]] || { echo "repository not found: ${repo_root}" >&2; exit 1; }
[[ -f "${checkpoint}" ]] || { echo "checkpoint not found: ${checkpoint}" >&2; exit 1; }
[[ "${checkpoint}" == *step_030000.pt ]] || { echo "checkpoint must be step 30000" >&2; exit 1; }
[[ ${#gpus[@]} -eq 4 ]] || { echo "GPUS must contain exactly four ids" >&2; exit 1; }

echo "scorer_source=initialization"
echo "checkpoint_step=30000"
echo "pairs=100000:1000 200000:1002 300000:1004 400000:1006"
if [[ "${preflight_only}" == 1 ]]; then
  exit 0
fi
[[ "${preflight_only}" == 0 ]] || { echo "PREFLIGHT_ONLY must be 0 or 1" >&2; exit 1; }
[[ -x "${python_bin}" ]] || { echo "python not executable: ${python_bin}" >&2; exit 1; }

export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-/mnt/vepfs02/output/kevin_wang/memorywam/model_assets/diffsynth_official}"
export DIFFSYNTH_SKIP_DOWNLOAD=true
export PYTHONPATH="${repo_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUTF8=1 LANG=C.UTF-8 LC_ALL=C.UTF-8

cd "${repo_root}"
mkdir -p "${log_root}"
pids=()
for lane in {0..3}; do
  gpu="${gpus[$lane]}"
  scene_seed="${scene_seeds[$lane]}"
  policy_seed="${policy_seeds[$lane]}"
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
    "EVALUATION.dynamic_surprise_max_segment=8" \
    "EVALUATION.dynamic_surprise_scorer_source=initialization" \
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
