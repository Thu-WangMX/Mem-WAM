#!/usr/bin/env bash
set -euo pipefail

repo_root="${REPO_ROOT:-/mnt/vepfs02/output/kevin_wang/memorywam/code/fastwam_wrist_latent_event_memory}"
python_bin="${PYTHON_BIN:-/root/kevin_wang/envs/fastwam_guidemem/bin/python}"
training_root="${TRAINING_ROOT:-/mnt/vepfs01/output/kevin_wang/memorywam/train/wrist_latent_event_memory_putback_k8_40k_seed42}"
checkpoint="${CHECKPOINT:-${training_root}/checkpoints/weights/step_005000.pt}"
training_config="${TRAINING_CONFIG:-${training_root}/config.yaml}"
boundary_manifest="${BOUNDARY_MANIFEST:-/mnt/vepfs02/output/kevin_wang/memorywam/analysis/putback_wrist_latent_event_predictor_v1/boundary_manifest/manifest.json}"
predictor_repo="${PREDICTOR_REPO:-/mnt/vepfs02/output/kevin_wang/memorywam/code/fastwam_wrist_latent_event_predictor}"
predictor_checkpoint="${PREDICTOR_CHECKPOINT:-/mnt/vepfs02/output/kevin_wang/memorywam/analysis/putback_wrist_latent_event_predictor_v1/best.pt}"
gpus_text="${GPUS:-0,1,2,3}"
run_tag="${RUN_TAG:-wrist_event_5k_rmbench4_$(date +%Y%m%d_%H%M%S)}"
log_root="${LOG_ROOT:-/mnt/vepfs01/output/kevin_wang/memorywam/eval_runs/${run_tag}}"
artifact_root="${ARTIFACT_ROOT:-${log_root}/artifacts}"
wait_timeout_seconds="${WAIT_TIMEOUT_SECONDS:-172800}"
wait_poll_seconds="${WAIT_POLL_SECONDS:-60}"
preflight_only="${PREFLIGHT_ONLY:-0}"

scene_seeds=(100000 200000 1300000 1400000)
policy_seeds=(1000 1002 1008 1010)
read -r -a gpus <<<"${gpus_text//,/ }"

die() { echo "ERROR: $*" >&2; exit 2; }
[[ -d "${repo_root}" ]] || die "repository not found: ${repo_root}"
[[ -x "${python_bin}" ]] || die "python not executable: ${python_bin}"
[[ -d "${predictor_repo}/src" ]] || die "predictor repository src not found: ${predictor_repo}/src"
[[ -s "${predictor_checkpoint}" ]] || die "predictor checkpoint not found: ${predictor_checkpoint}"
[[ -s "${boundary_manifest}" ]] || die "boundary manifest not found: ${boundary_manifest}"
[[ "${preflight_only}" == "0" || "${preflight_only}" == "1" ]] || die "PREFLIGHT_ONLY must be 0 or 1"
[[ "${wait_timeout_seconds}" =~ ^[0-9]+$ ]] || die "WAIT_TIMEOUT_SECONDS must be a non-negative integer"
[[ "${wait_poll_seconds}" =~ ^[1-9][0-9]*$ ]] || die "WAIT_POLL_SECONDS must be a positive integer"
[[ ${#gpus[@]} -eq 4 ]] || die "GPUS must contain exactly four unique GPU ids"
[[ $(printf '%s\n' "${gpus[@]}" | sort -u | wc -l | tr -d ' ') -eq 4 ]] || die "GPU ids must be unique"
for gpu in "${gpus[@]}"; do
  [[ "${gpu}" =~ ^[0-9]+$ ]] || die "GPU identifiers must be numeric: ${gpu}"
done

export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-/mnt/vepfs02/output/kevin_wang/memorywam/model_assets/diffsynth_official}"
export DIFFSYNTH_SKIP_DOWNLOAD=true
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${repo_root}/src:${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUTF8=1 LANG=C.UTF-8 LC_ALL=C.UTF-8

checkpoint_state="waiting"
if [[ -s "${checkpoint}" ]]; then
  checkpoint_state="ready"
fi
echo "benchmark=RMBench task=put_back_block task_config=demo_clean"
echo "checkpoint=${checkpoint}"
echo "checkpoint_state=${checkpoint_state}"
echo "scene_policy_pairs=100000:1000,200000:1002,1300000:1008,1400000:1010"
echo "gpus=${gpus[*]}"
echo "contract=offline_frozen_wrist_boundaries+online_same_predictor threshold=0.5 history=3 min=2 max=8 replan=16 layerwise_k8"
echo "artifact_root=${artifact_root}"

cd "${repo_root}"
"${python_bin}" - <<'PY'
from hydra import compose, initialize_config_dir
from pathlib import Path

root = Path.cwd()
with initialize_config_dir(version_base="1.3", config_dir=str(root / "configs")):
    cfg = compose(config_name="sim_robotwin_wrist_event")
assert cfg.model.native_cache.enabled is True
assert cfg.model.native_cache.mode == "layerwise"
assert cfg.model.native_cache.memory_tokens == 8
assert cfg.model.native_cache.group_size == 4
assert cfg.model.native_cache.anchor_frames == 2
assert cfg.model.native_cache.recent_frames == 4
assert cfg.model.native_cache.recursive is False
assert cfg.EVALUATION.dynamic_surprise_online is False
assert cfg.EVALUATION.wrist_event_online is True
assert cfg.EVALUATION.wrist_event_threshold == 0.5
assert cfg.EVALUATION.wrist_event_history == 3
assert cfg.EVALUATION.wrist_event_min_segment == 2
assert cfg.EVALUATION.wrist_event_max_segment == 8
assert cfg.EVALUATION.replan_steps == 16
print("preflight_status=ok config=sim_robotwin_wrist_event")
PY

if [[ "${preflight_only}" == "1" ]]; then
  exit 0
fi

started_wait="$(date +%s)"
while [[ ! -s "${checkpoint}" || ! -s "${training_config}" ]]; do
  elapsed=$(( $(date +%s) - started_wait ))
  if [[ ${elapsed} -ge ${wait_timeout_seconds} ]]; then
    die "timed out waiting for atomic 5k checkpoint/config after ${elapsed}s: ${checkpoint}"
  fi
  echo "waiting_for_5k elapsed_seconds=${elapsed} checkpoint_ready=$([[ -s "${checkpoint}" ]] && echo true || echo false) config_ready=$([[ -s "${training_config}" ]] && echo true || echo false)"
  sleep "${wait_poll_seconds}"
done

"${python_bin}" scripts/validate_putback_wrist_event_eval.py \
  --repo "${repo_root}" \
  --training-config "${training_config}" \
  --manifest "${boundary_manifest}" \
  --predictor "${predictor_checkpoint}" \
  --policy-checkpoint "${checkpoint}" \
  --expected-step 5000 \
  --eval-config-name sim_robotwin_wrist_event

mkdir -p "${log_root}" "${artifact_root}"
pids=()
for index in 0 1 2 3; do
  scene_seed="${scene_seeds[$index]}"
  policy_seed="${policy_seeds[$index]}"
  gpu="${gpus[$index]}"
  log_file="${log_root}/scene${scene_seed}_policy${policy_seed}.log"
  "${python_bin}" -u experiments/robotwin/eval_robotwin_single.py \
    --config-name sim_robotwin_wrist_event \
    "ckpt=${checkpoint}" \
    "gpu_id=${gpu}" \
    "seed=${policy_seed}" \
    "EVALUATION.start_scene_seed=${scene_seed}" \
    "EVALUATION.task_name=put_back_block" \
    "EVALUATION.task_config=demo_clean" \
    "EVALUATION.eval_num_episodes=1" \
    "EVALUATION.output_dir=${log_root}/scene${scene_seed}_policy${policy_seed}" \
    "EVALUATION.artifact_root=${artifact_root}" \
    "EVALUATION.wrist_event_predictor_repo=${predictor_repo}" \
    "EVALUATION.wrist_event_checkpoint=${predictor_checkpoint}" \
    >"${log_file}" 2>&1 &
  pids+=("$!")
  echo "launched scene=${scene_seed} policy=${policy_seed} physical_gpu=${gpu} pid=$! log=${log_file}"
done

failed=0
for pid in "${pids[@]}"; do
  wait "${pid}" || failed=$((failed + 1))
done

summary_path="${log_root}/summary.json"
"${python_bin}" scripts/summarize_putback_wrist_event_eval.py \
  --run-root "${log_root}" \
  --output "${summary_path}"
echo "evaluation_complete process_failures=${failed} summary=${summary_path}"
exit "${failed}"
