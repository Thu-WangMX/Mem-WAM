#!/usr/bin/env bash
set -euo pipefail

repo_root=${REPO_ROOT:-/mnt/vepfs02/output/kevin_wang/memorywam/code/fastwam_multires_latent_kernel_k8}
python_bin=${PYTHON_BIN:-/root/kevin_wang/envs/fastwam_guidemem/bin/python}
checkpoint=${CHECKPOINT:-/mnt/vepfs01/output/kevin.wang/memorywam/train/multires_latent_kernel_k8_putback_init_joint20k_lr2e4_seed42_v1/checkpoints/weights/step_020000.pt}
gpus_text=${GPUS:-2,3,4,5}
indices_text=${INDICES:-0,1,2,3}
policy_seeds_text=${POLICY_SEEDS:-1000,1002,1008,1010}
scene_seeds_text=${SCENE_SEEDS:-100000,200000,1300000,1400000}
run_tag=${RUN_TAG:-latent_kernel_regime_20k_rmbench4_$(date +%Y%m%d_%H%M%S)}
log_root=${LOG_ROOT:-/mnt/vepfs01/output/kevin.wang/memorywam/eval_runs/${run_tag}}
preflight_only=${PREFLIGHT_ONLY:-0}

read -r -a scene_seeds <<<"${scene_seeds_text//,/ }"
read -r -a policy_seeds <<<"${policy_seeds_text//,/ }"
read -r -a gpus <<<"${gpus_text//,/ }"
read -r -a indices <<<"${indices_text//,/ }"

die() { echo "ERROR: $*" >&2; exit 2; }
[[ -d ${repo_root} ]] || die "missing repository: ${repo_root}"
[[ -x ${python_bin} ]] || die "missing python: ${python_bin}"
[[ -s ${checkpoint} ]] || die "missing checkpoint: ${checkpoint}"
[[ ${#gpus[@]} -eq 4 ]] || die "GPUS must contain four ids"
[[ ${#policy_seeds[@]} -eq 4 ]] || die "POLICY_SEEDS must contain four seeds"
[[ ${#scene_seeds[@]} -eq 4 ]] || die "SCENE_SEEDS must contain four seeds"
[[ $(printf '%s\n' "${gpus[@]}" | sort -u | wc -l | tr -d ' ') -eq 4 ]] || die "GPU ids must be unique"
for index in "${indices[@]}"; do
  [[ ${index} =~ ^[0-3]$ ]] || die "INDICES entries must be in 0..3"
done

export DIFFSYNTH_MODEL_BASE_PATH=${DIFFSYNTH_MODEL_BASE_PATH:-/mnt/vepfs02/datasets/kevin_wang/code/projects/FastWAM/checkpoints}
export DIFFSYNTH_SKIP_DOWNLOAD=true
export PYTHONPATH=${repo_root}/src:${repo_root}${PYTHONPATH:+:${PYTHONPATH}}
export PYTHONUTF8=1 LANG=C.UTF-8 LC_ALL=C.UTF-8

cd "${repo_root}"
"${python_bin}" - <<'PY'
from hydra import compose, initialize_config_dir
from pathlib import Path

root = Path.cwd()
with initialize_config_dir(version_base="1.3", config_dir=str(root / "configs")):
    cfg = compose(config_name="sim_robotwin_latent_kernel_regime")
assert cfg.EVALUATION.latent_kernel_regime_online is True
assert cfg.EVALUATION.policy_name == "fastwam_latent_kernel_regime_policy"
assert cfg.model.native_cache.memory_tokens == 8
assert cfg.model.native_cache.anchor_frames == 2
assert cfg.model.native_cache.recent_frames == 4
print("preflight_status=ok selector=latent_kernel_regime_online anchor2_recent4_l4to8_k8")
PY
[[ ${preflight_only} == 0 ]] || exit 0

mkdir -p "${log_root}"
pids=()
for index in "${indices[@]}"; do
  scene_seed=${scene_seeds[$index]}
  policy_seed=${policy_seeds[$index]}
  gpu=${gpus[$index]}
  output_dir=${log_root}/scene${scene_seed}_policy${policy_seed}
  mkdir -p "${output_dir}"
  "${python_bin}" -u experiments/robotwin/eval_robotwin_single.py \
    --config-name sim_robotwin_latent_kernel_regime \
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
  echo "launched scene=${scene_seed} policy=${policy_seed} gpu=${gpu} pid=$!"
done

failed=0
for pid in "${pids[@]}"; do
  wait "${pid}" || failed=$((failed + 1))
done
for log in "${log_root}"/*.log; do
  echo "===== ${log} ====="
  grep -E "FASTWAM_LATENT_KERNEL|FASTWAM_NATIVE_CACHE_METRICS|success|Success|Traceback" "${log}" | tail -n 80 || true
done
echo "log_root=${log_root} failed=${failed}"
exit "${failed}"
