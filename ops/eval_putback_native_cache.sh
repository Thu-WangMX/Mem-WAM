#!/usr/bin/env bash
set -euo pipefail

repo_root="${REPO_ROOT:-/mnt/vepfs02/output/kevin_wang/memorywam/code/fastwam_native_cache_consolidation}"
python_bin="${PYTHON_BIN:-/root/kevin_wang/envs/fastwam_guidemem/bin/python}"
model_base="${DIFFSYNTH_MODEL_BASE_PATH:-/mnt/vepfs02/output/kevin_wang/memorywam/model_assets/diffsynth_official}"
checkpoint="${CHECKPOINT:-/mnt/vepfs02/output/kevin_wang/memorywam/train/native_cache_consolidation_putback_e2e_25k_seed42/checkpoints/weights/step_001000.pt}"
seeds_text="${SEEDS:-0}"
gpus_text="${GPUS:-0}"
eval_num_episodes="${EVAL_NUM_EPISODES:-1}"
preflight_only="${PREFLIGHT_ONLY:-0}"
run_tag="${RUN_TAG:-native_cache_$(date +%Y%m%d_%H%M%S)}"
log_root="${LOG_ROOT:-/mnt/vepfs02/output/kevin_wang/memorywam/eval_runs/${run_tag}}"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

[[ -d "${repo_root}" ]] || die "repository not found: ${repo_root}"
[[ -x "${python_bin}" ]] || die "python not executable: ${python_bin}"
[[ -d "${model_base}" ]] || die "model asset root not found: ${model_base}"
[[ -f "${checkpoint}" ]] || die "checkpoint not found: ${checkpoint}"
[[ "${eval_num_episodes}" =~ ^[1-9][0-9]*$ ]] || die "EVAL_NUM_EPISODES must be positive"
[[ "${preflight_only}" == "0" || "${preflight_only}" == "1" ]] || die "PREFLIGHT_ONLY must be 0 or 1"

read -r -a seeds <<<"${seeds_text//,/ }"
read -r -a gpus <<<"${gpus_text//,/ }"
[[ ${#seeds[@]} -gt 0 ]] || die "SEEDS must not be empty"
[[ ${#seeds[@]} -eq ${#gpus[@]} ]] || die "SEEDS and GPUS must have the same number of entries"

declare -A seen_gpus=()
for index in "${!seeds[@]}"; do
  seed="${seeds[$index]}"
  gpu="${gpus[$index]}"
  [[ "${seed}" =~ ^[0-9]+$ ]] || die "invalid seed: ${seed}"
  [[ "${gpu}" =~ ^[0-9]+$ ]] || die "invalid GPU index: ${gpu}"
  [[ -z "${seen_gpus[$gpu]:-}" ]] || die "GPU index repeated: ${gpu}"
  seen_gpus[$gpu]=1
done

export DIFFSYNTH_MODEL_BASE_PATH="${model_base}"
export DIFFSYNTH_SKIP_DOWNLOAD=true
export PYTHONUTF8=1
export LANG=C.UTF-8
export LC_ALL=C.UTF-8

cd "${repo_root}"
checkpoint_summary="$(${python_bin} scripts/validate_native_cache_checkpoint.py "${checkpoint}")"

echo "host=$(hostname)"
echo "repo_root=${repo_root}"
echo "checkpoint_summary=${checkpoint_summary}"
echo "seeds=${seeds[*]}"
echo "gpus=${gpus[*]}"
echo "eval_num_episodes=${eval_num_episodes}"
echo "config=sim_robotwin_native_cache"
echo "log_root=${log_root}"

if [[ "${preflight_only}" == "1" ]]; then
  echo "preflight_status=ok"
  exit 0
fi

command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi is unavailable"
gpu_count="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')"
for gpu in "${gpus[@]}"; do
  (( gpu < gpu_count )) || die "GPU ${gpu} is outside available range 0..$((gpu_count - 1))"
done
[[ ! -e "${log_root}" ]] || die "LOG_ROOT already exists: ${log_root}"
mkdir -p "${log_root}"

{
  echo "host=$(hostname)"
  echo "repo_root=${repo_root}"
  echo "checkpoint_summary=${checkpoint_summary}"
  echo "seeds=${seeds[*]}"
  echo "gpus=${gpus[*]}"
  echo "eval_num_episodes=${eval_num_episodes}"
  echo "config=sim_robotwin_native_cache"
} >"${log_root}/run_contract.txt"

pids=()
for index in "${!seeds[@]}"; do
  seed="${seeds[$index]}"
  gpu="${gpus[$index]}"
  output_dir="${log_root}/${run_tag}_seed${seed}"
  mkdir -p "${output_dir}"
  "${python_bin}" -u experiments/robotwin/eval_robotwin_single.py \
    --config-name sim_robotwin_native_cache \
    "ckpt=${checkpoint}" \
    "gpu_id=${gpu}" \
    "seed=${seed}" \
    "EVALUATION.task_name=put_back_block" \
    "EVALUATION.task_config=demo_clean" \
    "EVALUATION.eval_num_episodes=${eval_num_episodes}" \
    "EVALUATION.output_dir=${output_dir}" \
    >"${log_root}/seed${seed}.launcher.log" 2>&1 &
  pids+=("$!")
  echo "launched seed=${seed} gpu=${gpu} pid=$!"
done

failed=0
for index in "${!pids[@]}"; do
  if wait "${pids[$index]}"; then
    echo "seed=${seeds[$index]} process_status=ok"
  else
    echo "seed=${seeds[$index]} process_status=failed"
    failed=$((failed + 1))
  fi
done

for seed in "${seeds[@]}"; do
  echo "===== seed=${seed} ====="
  grep -E "FASTWAM_EVAL_CONTRACT|FASTWAM_NATIVE_CACHE_METRICS|success|Success|Evaluation finished|Traceback" \
    "${log_root}/seed${seed}.launcher.log" | tail -n 30 || true
done

exit "${failed}"

