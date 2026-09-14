#!/usr/bin/env bash
set -uo pipefail

# Evaluate the SwapT physical-settle/rate-debt 28k checkpoint for exactly 150
# accepted episodes.  Eight long-lived evaluators avoid reloading the model for
# every episode: GPUs 0..5 run 19 episodes and GPUs 6..7 run 18 episodes.

repo_root=${REPO_ROOT:-/mnt/vepfs02/output/kevin_wang/memorywam/code/fastwam_physical_settle_rate_debt_k8}
python_bin=${PYTHON_BIN:-/root/kevin_wang/envs/fastwam_guidemem/bin/python}
checkpoint=${CHECKPOINT:-/mnt/vepfs01/output/kevin.wang/memorywam/train/physical_settle_t7c5_k8_swapt_resume24k_to30k_lr2e4_seed42_v1/checkpoints/weights/step_028000.pt}
artifact_root=${ARTIFACT_ROOT:-/mnt/vepfs01/output/kevin.wang/memorywam/eval_runs}
run_tag=${RUN_TAG:-swapt_physical_28k_150_policyseed0_$(date +%Y%m%d_%H%M%S)}
log_root=${LOG_ROOT:-${artifact_root}/${run_tag}}
preflight_only=${PREFLIGHT_ONLY:-0}
require_idle_gpus=${REQUIRE_IDLE_GPUS:-1}

gpus=(0 1 2 3 4 5 6 7)
episode_counts=(19 19 19 19 19 19 18 18)
# These ranges are deliberately far apart, so evaluator-side seed increments
# and unstable expert-check skips cannot overlap between workers.
start_seeds=(3000000 4000000 5000000 6000000 7000000 8000000 9000000 10000000)

die(){ echo "ERROR: $*" >&2; exit 2; }
[[ ${preflight_only} == 0 || ${preflight_only} == 1 ]] || die "PREFLIGHT_ONLY must be 0 or 1"
[[ -d ${repo_root} ]] || die "Missing repository: ${repo_root}"
[[ -x ${python_bin} ]] || die "Missing Python: ${python_bin}"
[[ -s ${checkpoint} ]] || die "Missing checkpoint: ${checkpoint}"
[[ $(stat -c %s "${checkpoint}") -gt 12000000000 ]] || die "Checkpoint is incomplete: ${checkpoint}"
[[ ${#gpus[@]} -eq 8 && ${#episode_counts[@]} -eq 8 && ${#start_seeds[@]} -eq 8 ]] || die "Expected eight worker specifications"

total_requested=0
for count in "${episode_counts[@]}"; do
  total_requested=$((total_requested + count))
done
[[ ${total_requested} -eq 150 ]] || die "Episode allocation must total 150, got ${total_requested}"

export DIFFSYNTH_MODEL_BASE_PATH=${DIFFSYNTH_MODEL_BASE_PATH:-/mnt/vepfs02/datasets/kevin_wang/code/projects/FastWAM/checkpoints}
export DIFFSYNTH_SKIP_DOWNLOAD=true
export PYTHONPATH=${repo_root}/src:${repo_root}${PYTHONPATH:+:${PYTHONPATH}}
export PYTHONUTF8=1 LANG=C.UTF-8 LC_ALL=C.UTF-8

echo "host=$(hostname)"
echo "checkpoint=${checkpoint}"
echo "policy_seed=0 total_episodes=${total_requested}"
echo "episode_counts=${episode_counts[*]}"
echo "start_scene_seeds=${start_seeds[*]}"
echo "output=${log_root}"

[[ ${preflight_only} == 0 ]] || { echo "preflight_status=ok"; exit 0; }
[[ ! -e ${log_root} ]] || die "Refusing to overwrite existing evaluation: ${log_root}"

if [[ ${require_idle_gpus} == 1 ]]; then
  active=$(nvidia-smi -i 0,1,2,3,4,5,6,7 --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d' | sort -u)
  [[ -z ${active} ]] || die "Selected GPUs are busy: ${active}"
fi

mkdir -p "${log_root}"
printf '%s\n' "${log_root}" > /tmp/latest_eval_swapt_physical_28k_150.txt
{
  echo "checkpoint=${checkpoint}"
  echo "policy_seed=0"
  echo "total_episodes=${total_requested}"
  echo "episode_counts=${episode_counts[*]}"
  echo "start_scene_seeds=${start_seeds[*]}"
} >"${log_root}/contract.txt"

pids=()
for i in 0 1 2 3 4 5 6 7; do
  gpu=${gpus[$i]}
  count=${episode_counts[$i]}
  start_seed=${start_seeds[$i]}
  worker_dir=${log_root}/gpu${gpu}_start${start_seed}_n${count}
  mkdir -p "${worker_dir}"
  echo "launch gpu=${gpu} episodes=${count} start_scene_seed=${start_seed}"
  (
    cd "${repo_root}"
    "${python_bin}" -u experiments/robotwin/eval_robotwin_single.py \
      --config-name sim_robotwin_physical_settle_rate_debt \
      "ckpt=${checkpoint}" \
      "gpu_id=${gpu}" \
      "seed=0" \
      "EVALUATION.start_scene_seed=${start_seed}" \
      "EVALUATION.task_name=swap_T" \
      "EVALUATION.task_config=demo_clean" \
      "EVALUATION.eval_num_episodes=${count}" \
      "EVALUATION.output_dir=${worker_dir}"
  ) >"${worker_dir}/launcher.log" 2>&1 &
  pids+=("$!")
done

worker_failures=0
for pid in "${pids[@]}"; do
  wait "${pid}" || worker_failures=$((worker_failures + 1))
done

total_finished=0
total_success=0
{
  echo "RESULTS"
  for worker_dir in "${log_root}"/gpu*_start*_n*; do
    log=${worker_dir}/launcher.log
    finished=$(grep -c "RMBENCH_EVAL_DIAGNOSTICS" "${log}" 2>/dev/null || true)
    success=$(grep -c '"official_success": true' "${log}" 2>/dev/null || true)
    total_finished=$((total_finished + finished))
    total_success=$((total_success + success))
    echo "$(basename "${worker_dir}") success=${success}/${finished}"
  done
  echo "TOTAL success=${total_success}/${total_finished} requested=${total_requested} worker_failures=${worker_failures}"
} | tee "${log_root}/summary.txt"

[[ ${worker_failures} -eq 0 ]] || exit 1
[[ ${total_finished} -eq ${total_requested} ]] || exit 3
exit 0
