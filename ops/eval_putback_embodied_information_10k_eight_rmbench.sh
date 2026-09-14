#!/usr/bin/env bash
set -euo pipefail

repo_root="${REPO_ROOT:-/mnt/vepfs02/output/kevin.wang/memorywam/code/fastwam_dynamic_multiframe_selection}"
python_bin="${PYTHON_BIN:-/root/kevin_wang/envs/fastwam_guidemem/bin/python}"
checkpoint="${CHECKPOINT:-/mnt/vepfs02/output/kevin.wang/memorywam/train/embodied_information_k8_putback_e2e_40k_seed42/checkpoints/weights/step_010000.pt}"
run_tag="${RUN_TAG:-embodied_information_10k_rmbench8_$(date +%Y%m%d_%H%M%S)}"
log_root="${LOG_ROOT:-/mnt/vepfs02/output/kevin.wang/memorywam/eval_runs/${run_tag}}"
artifact_root="${ARTIFACT_ROOT:-${log_root}/artifacts}"
gpus_text="${GPUS:-0,1,2,3,4,5,6,7}"
indices_text="${PAIR_INDICES:-0,1,2,3,4,5,6,7}"
preflight_only="${PREFLIGHT_ONLY:-0}"

scene_seeds=(100000 200000 300000 400000 1300000 1400000 1500000 1600000)
policy_seeds=(1000 1002 1004 1006 1008 1010 1012 1014)
read -r -a gpus <<<"${gpus_text//,/ }"
read -r -a indices <<<"${indices_text//,/ }"

die() { echo "ERROR: $*" >&2; exit 2; }
[[ -d "${repo_root}" ]] || die "repository not found: ${repo_root}"
[[ -x "${python_bin}" ]] || die "python not executable: ${python_bin}"
[[ -s "${checkpoint}" ]] || die "checkpoint not found: ${checkpoint}"
[[ ${#gpus[@]} -eq ${#indices[@]} && ${#gpus[@]} -gt 0 ]] || die "GPUS and PAIR_INDICES must have equal nonzero length"
[[ $(printf '%s\n' "${gpus[@]}" | sort -u | wc -l | tr -d ' ') -eq ${#gpus[@]} ]] || die "GPU ids must be unique"
for index in "${indices[@]}"; do
  [[ "${index}" =~ ^[0-7]$ ]] || die "pair index must be 0-7: ${index}"
done

export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-/mnt/vepfs02/output/kevin.wang/memorywam/model_assets/diffsynth_official}"
export DIFFSYNTH_SKIP_DOWNLOAD=true
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${repo_root}/src:${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUTF8=1 LANG=C.UTF-8 LC_ALL=C.UTF-8

cd "${repo_root}"
"${python_bin}" - <<'PY'
from hydra import compose, initialize_config_dir
from pathlib import Path

with initialize_config_dir(version_base="1.3", config_dir=str(Path.cwd() / "configs")):
    cfg = compose(config_name="sim_robotwin_embodied_information")
assert cfg.model.native_cache.enabled is True
assert cfg.model.native_cache.mode == "layerwise"
assert cfg.model.native_cache.memory_tokens == 8
assert cfg.EVALUATION.replan_steps == 16
assert cfg.EVALUATION.dynamic_surprise_online is False
assert cfg.EVALUATION.wrist_event_online is False
assert cfg.EVALUATION.embodied_information_online is True
assert "kevin.wang" in cfg.EVALUATION.embodied_information_lock
print("preflight_status=ok selector=embodied_information initialization_wam=true strict_online=true")
PY

if [[ "${preflight_only}" == "1" ]]; then exit 0; fi
[[ ! -e "${log_root}" ]] || die "refusing to overwrite evaluation root: ${log_root}"
mkdir -p "${log_root}" "${artifact_root}"

pids=()
labels=()
for position in "${!indices[@]}"; do
  index="${indices[$position]}"
  gpu="${gpus[$position]}"
  scene_seed="${scene_seeds[$index]}"
  policy_seed="${policy_seeds[$index]}"
  label="scene${scene_seed}_policy${policy_seed}"
  labels+=("${label}")
  "${python_bin}" -u experiments/robotwin/eval_robotwin_single.py \
    --config-name sim_robotwin_embodied_information \
    "ckpt=${checkpoint}" \
    "gpu_id=${gpu}" \
    "seed=${policy_seed}" \
    "EVALUATION.start_scene_seed=${scene_seed}" \
    "EVALUATION.task_name=put_back_block" \
    "EVALUATION.task_config=demo_clean" \
    "EVALUATION.eval_num_episodes=1" \
    "EVALUATION.output_dir=${log_root}/${label}" \
    "EVALUATION.artifact_root=${artifact_root}" \
    >"${log_root}/${label}.log" 2>&1 &
  pids+=("$!")
  echo "launched index=${index} scene=${scene_seed} policy=${policy_seed} gpu=${gpu} pid=$!"
done

failed=0
for position in "${!pids[@]}"; do
  if wait "${pids[$position]}"; then
    echo "${labels[$position]} status=ok"
  else
    echo "${labels[$position]} status=failed"
    failed=$((failed + 1))
  fi
done
summary="${log_root}/summary.txt"
{
  echo "checkpoint=${checkpoint}"
  echo "selector=embodied_information initialization_wam=true strict_online=true"
  for label in "${labels[@]}"; do
    result="$(grep -a 'Success rate:' "${log_root}/${label}.log" | tail -n 1 || true)"
    events="$(grep -ac 'FASTWAM_EMBODIED_DETECTOR' "${log_root}/${label}.log" || true)"
    arrivals="$(grep -ac 'FASTWAM_EMBODIED_PLANNING_ARRIVAL' "${log_root}/${label}.log" || true)"
    echo "${label} ${result:-missing_result} detector_updates=${events} planning_arrivals=${arrivals}"
  done
  echo "failed_processes=${failed}"
} | tee "${summary}"
exit "${failed}"

