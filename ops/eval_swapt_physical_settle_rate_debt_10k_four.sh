#!/usr/bin/env bash
set -euo pipefail

repo_root=${REPO_ROOT:-/mnt/vepfs02/output/kevin_wang/memorywam/code/fastwam_physical_settle_rate_debt_k8}
python_bin=${PYTHON_BIN:-/root/kevin_wang/envs/fastwam_guidemem/bin/python}
checkpoint=${CHECKPOINT:-/mnt/vepfs01/output/kevin.wang/memorywam/train/physical_settle_t7c5_k8_swapt_scratch_joint16k_lr2e4_seed42_v1/checkpoints/weights/step_010000.pt}
manifest_root=${PHYSICAL_SETTLE_RATE_DEBT_MANIFEST:-/mnt/vepfs01/output/kevin.wang/memorywam/analysis/swapt_physical_settle_rate_debt_t7c5_k8_v1}
lerobot_root=${SWAPT_LEROBOT_ROOT:-/mnt/vepfs02/output/kevin_wang/memorywam/data/fastwam_fullkv_remaining8_rgb_v3/swap_T/lerobot/swap_T}
gpus_text=${GPUS:-0,1,2,3}
policy_seeds_text=${POLICY_SEEDS:-0,0,0,0}
scene_seeds_text=${SCENE_SEEDS:-100000,200000,1300000,1400000}
run_tag=${RUN_TAG:-swapt_physical_settle_rate_debt_10k_parity4_$(date +%Y%m%d_%H%M%S)}
log_root=${LOG_ROOT:-/mnt/vepfs01/output/kevin.wang/memorywam/eval_runs/${run_tag}}
preflight_only=${PREFLIGHT_ONLY:-0}

read -r -a scene_seeds <<<"${scene_seeds_text//,/ }"
read -r -a policy_seeds <<<"${policy_seeds_text//,/ }"
read -r -a gpus <<<"${gpus_text//,/ }"

die() { echo "ERROR: $*" >&2; exit 2; }
[[ -d ${repo_root} ]] || die "missing repository: ${repo_root}"
[[ -x ${python_bin} ]] || die "missing python: ${python_bin}"
[[ -s ${checkpoint} ]] || die "missing checkpoint: ${checkpoint}"
[[ -s ${manifest_root}/manifest.json ]] || die "missing manifest: ${manifest_root}"
count=${#gpus[@]}
[[ ${count} -ge 1 ]] || die "GPUS must not be empty"
[[ ${#policy_seeds[@]} -eq ${count} ]] || die "POLICY_SEEDS count must match GPUS"
[[ ${#scene_seeds[@]} -eq ${count} ]] || die "SCENE_SEEDS count must match GPUS"
[[ $(printf '%s\n' "${gpus[@]}" | sort -u | wc -l | tr -d ' ') -eq ${count} ]] || die "GPU ids must be unique"

export DIFFSYNTH_MODEL_BASE_PATH=${DIFFSYNTH_MODEL_BASE_PATH:-/mnt/vepfs02/datasets/kevin_wang/code/projects/FastWAM/checkpoints}
export DIFFSYNTH_SKIP_DOWNLOAD=true
export PYTHONPATH=${repo_root}/src:${repo_root}${PYTHONPATH:+:${PYTHONPATH}}
export PYTHONUTF8=1 LANG=C.UTF-8 LC_ALL=C.UTF-8

cd "${repo_root}"
"${python_bin}" scripts/check_physical_settle_rate_debt_online_parity.py \
  --manifest "${manifest_root}" \
  --lerobot-root "${lerobot_root}" \
  --task swap_T
"${python_bin}" - <<'PY'
from hydra import compose, initialize_config_dir
from pathlib import Path

root = Path.cwd()
with initialize_config_dir(version_base="1.3", config_dir=str(root / "configs")):
    cfg = compose(config_name="sim_robotwin_physical_settle_rate_debt")
assert cfg.EVALUATION.physical_settle_rate_debt_online is True
assert cfg.model.native_cache.mode == "layerwise"
assert cfg.model.native_cache.memory_tokens == 8
assert cfg.model.native_cache.anchor_frames == 2
assert cfg.model.native_cache.recent_frames == 4
assert cfg.model.native_cache.recursive is False
print("preflight_status=ok task=swap_T checkpoint=10k selector=physical_settle_rate_debt_online")
PY
[[ ${preflight_only} == 0 ]] || exit 0

mkdir -p "${log_root}"
pids=()
for ((index=0; index<count; index++)); do
  scene_seed=${scene_seeds[$index]}
  policy_seed=${policy_seeds[$index]}
  gpu=${gpus[$index]}
  output_dir=${log_root}/scene${scene_seed}_policy${policy_seed}
  mkdir -p "${output_dir}"
  "${python_bin}" -u experiments/robotwin/eval_robotwin_single.py \
    --config-name sim_robotwin_physical_settle_rate_debt \
    "ckpt=${checkpoint}" \
    "gpu_id=${gpu}" \
    "seed=${policy_seed}" \
    "EVALUATION.start_scene_seed=${scene_seed}" \
    "EVALUATION.task_name=swap_T" \
    "EVALUATION.task_config=demo_clean" \
    "EVALUATION.eval_num_episodes=1" \
    "EVALUATION.output_dir=${output_dir}" \
    >"${log_root}/scene${scene_seed}_policy${policy_seed}.log" 2>&1 &
  pids+=("$!")
  echo "launched scene=${scene_seed} policy=${policy_seed} gpu=${gpu} pid=$!"
done

failed=0
for pid in "${pids[@]}"; do
  wait "${pid}" || failed=$((failed + 1))
done
for log in "${log_root}"/*.log; do
  echo "===== ${log} ====="
  grep -E "FASTWAM_PHYSICAL_SETTLE_RATE|FASTWAM_NATIVE_CACHE_METRICS|RMBENCH_EVAL_DIAGNOSTICS|Success rate|Traceback|CUDA out of memory|BrokenPipeError" "${log}" | tail -n 160 || true
done
echo "log_root=${log_root} failed=${failed}"
exit "${failed}"
