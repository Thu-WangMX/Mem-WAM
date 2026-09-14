#!/usr/bin/env bash
set -euo pipefail

repo_root="${REPO_ROOT:-/mnt/vepfs02/output/kevin_wang/memorywam/code/fastwam_dynamic_surprise_memory}"
python_bin="${PYTHON_BIN:-/root/kevin_wang/envs/fastwam_guidemem/bin/python}"
checkpoint="${CHECKPOINT:-/mnt/vepfs02/output/kevin_wang/memorywam/train/dynamic_surprise_memory_putback_k8_25k_seed42/checkpoints/weights/step_025000.pt}"
gpus_text="${GPUS:-0,1,2,3}"
indices_text="${INDICES:-0,1,2,3}"
run_tag="${RUN_TAG:-dynamic_surprise_25k_rmbench4_$(date +%Y%m%d_%H%M%S)}"
log_root="${LOG_ROOT:-/mnt/vepfs02/output/kevin_wang/memorywam/eval_runs/${run_tag}}"
preflight_only="${PREFLIGHT_ONLY:-0}"

# Fixed RMBench contracts requested for this comparison: scene seed / policy seed.
scene_seeds=(100000 200000 1300000 1400000)
policy_seeds=(1000 1002 1008 1010)
read -r -a gpus <<<"${gpus_text//,/ }"
read -r -a indices <<<"${indices_text//,/ }"

die() { echo "ERROR: $*" >&2; exit 1; }
[[ -d "${repo_root}" ]] || die "repository not found: ${repo_root}"
[[ -x "${python_bin}" ]] || die "python not executable: ${python_bin}"
[[ -f "${checkpoint}" ]] || die "checkpoint not found: ${checkpoint}"
[[ ${#gpus[@]} -eq 4 ]] || die "GPUS must contain exactly four unique GPU ids"
[[ "${preflight_only}" == 0 || "${preflight_only}" == 1 ]] || die "PREFLIGHT_ONLY must be 0 or 1"
[[ $(printf '%s\n' "${gpus[@]}" | sort -u | wc -l | tr -d ' ') -eq 4 ]] || die "GPU ids must be unique"
for index in "${indices[@]}"; do
  [[ "${index}" =~ ^[0-3]$ ]] || die "INDICES entries must be in 0..3"
done
[[ $(printf '%s\n' "${indices[@]}" | sort -u | wc -l | tr -d ' ') -eq ${#indices[@]} ]] || die "INDICES must be unique"

export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-/mnt/vepfs02/output/kevin_wang/memorywam/model_assets/diffsynth_official}"
export DIFFSYNTH_SKIP_DOWNLOAD=true
export PYTHONPATH="${repo_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUTF8=1 LANG=C.UTF-8 LC_ALL=C.UTF-8

cd "${repo_root}"
echo "benchmark=RMBench task=put_back_block task_config=demo_clean"
echo "checkpoint=${checkpoint}"
echo "scene_seeds=${scene_seeds[*]} policy_seeds=${policy_seeds[*]} gpus=${gpus[*]}"
echo "selected_indices=${indices[*]}"
echo "online_surprise=sigma1_gamma1.5_window5_min2_max8 memory_tokens=8"
if [[ "${preflight_only}" == 1 ]]; then
  "${python_bin}" - <<'PY'
from hydra import compose, initialize_config_dir
from pathlib import Path
root = Path.cwd()
with initialize_config_dir(version_base="1.3", config_dir=str(root / "configs")):
    cfg = compose(config_name="sim_robotwin_dynamic_surprise")
assert cfg.model.native_cache.memory_tokens == 8
assert cfg.EVALUATION.dynamic_surprise_online is True
assert cfg.EVALUATION.policy_name == "fastwam_dynamic_surprise_policy"
print("preflight_status=ok")
PY
  exit 0
fi

mkdir -p "${log_root}"
pids=()
for index in "${indices[@]}"; do
  scene_seed="${scene_seeds[$index]}"
  policy_seed="${policy_seeds[$index]}"
  gpu="${gpus[$index]}"
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
    "EVALUATION.output_dir=${output_dir}" \
    >"${log_root}/scene${scene_seed}_policy${policy_seed}.log" 2>&1 &
  pids+=("$!")
  echo "launched scene=${scene_seed} policy=${policy_seed} physical_gpu=${gpu} pid=$!"
done

failed=0
for pid in "${pids[@]}"; do
  wait "${pid}" || failed=$((failed + 1))
done
for log in "${log_root}"/*.log; do
  echo "===== ${log} ====="
  grep -E "FASTWAM_DYNAMIC_SURPRISE|FASTWAM_NATIVE_CACHE_METRICS|success|Success|Traceback" "${log}" | tail -n 50 || true
done
echo "log_root=${log_root} failed=${failed}"
exit "${failed}"
