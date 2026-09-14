#!/usr/bin/env bash
set -euo pipefail

repo_root="${REPO_ROOT:-/mnt/vepfs02/output/kevin.wang/memorywam/code/fastwam_event_conditioned_rate}"
runtime_bin="${RUNTIME_BIN:-/root/kevin_wang/envs/fastwam_guidemem/bin}"
memorywam_root="${MEMORYWAM_ROOT:-/mnt/vepfs02/output/kevin.wang/memorywam}"
source_run="${SOURCE_RUN:-/mnt/vepfs01/output/kevin.wang/memorywam/train/control_information_v6_event_full_forced_half_battery_e2e_30k_seed42}"
resume_weights="${RESUME_WEIGHTS:-${source_run}/checkpoints/weights/step_030000.pt}"
output_dir="${OUTPUT_DIR:-/mnt/vepfs01/output/kevin.wang/memorywam/train/control_information_v6_event_full_forced_half_battery_e2e_50k_seed42_resume30k_weightonly}"
manifest_root="${MANIFEST_ROOT:-${memorywam_root}/data/battery_control_information_planning_segments_v3_causal_online}"
runtime_lock="${RUNTIME_LOCK:-${memorywam_root}/analysis/putback_control_information_selector_v3/selector_runtime/locked_selector.json}"
battery_root="${BATTERY_ROOT:-/mnt/vepfs02/output/kevin_wang/memorywam/data/fastwam_fullkv_remaining8_rgb_v3/battery_try}"
preflight_only="${PREFLIGHT_ONLY:-0}"

export PATH="${runtime_bin}:${PATH}"
export PYTHONPATH="${repo_root}/src:${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-${memorywam_root}/model_assets/diffsynth_official}"
export DIFFSYNTH_SKIP_DOWNLOAD=true
export RMBENCH_BATTERY_LEROBOT="${RMBENCH_BATTERY_LEROBOT:-${battery_root}/lerobot/battery_try}"
export MEMORYWAM_BATTERY_STATS="${MEMORYWAM_BATTERY_STATS:-${battery_root}/stats/dataset_stats.json}"
export MEMORYWAM_BATTERY_TEXT_CACHE="${MEMORYWAM_BATTERY_TEXT_CACHE:-${battery_root}/text_cache}"
export MEMORYWAM_BATTERY_CONTINUOUS_LATENTS="${MEMORYWAM_BATTERY_CONTINUOUS_LATENTS:-${battery_root}/temporal_fullkv_continuous_episode_stride16_v4}"
export FASTWAM_ACTION_DIT_INIT="${FASTWAM_ACTION_DIT_INIT:-${memorywam_root}/model_assets/fastwam_fullkv_official_init/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt}"
export DYNAMIC_SURPRISE_MANIFEST="${manifest_root}"
export DYNAMIC_SURPRISE_BOUNDARY_STEP=-1
export CONTROL_INFORMATION_EVENT_RATE_OUTPUT="${output_dir}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PYTHONUTF8=1 LANG=C.UTF-8 LC_ALL=C.UTF-8
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

die() { echo "ERROR: $*" >&2; exit 2; }

[[ "${preflight_only}" == "0" || "${preflight_only}" == "1" ]] || die "PREFLIGHT_ONLY must be 0 or 1"
[[ -d "${repo_root}" && -x "${runtime_bin}/python" ]] || die "repository/runtime missing"
[[ "${resume_weights}" == */checkpoints/weights/step_030000.pt ]] || die "resume must be the 30k portable checkpoint"
[[ ! -e "${output_dir}" ]] || die "refusing to overwrite output: ${output_dir}"

IFS=',' read -r -a visible_gpus <<<"${CUDA_VISIBLE_DEVICES}"
[[ "${#visible_gpus[@]}" -eq 8 ]] || die "formal resume requires exactly eight GPUs"
[[ "$(printf '%s\n' "${visible_gpus[@]}" | sort -u | wc -l | tr -d ' ')" -eq 8 ]] || die "GPU indices must be unique"

for required in \
  "${repo_root}/scripts/train.py" \
  "${repo_root}/tests/test_event_conditioned_rate.py" \
  "${repo_root}/tests/test_weight_only_resume_step.py" \
  "${repo_root}/configs/task/rmbench_battery_control_information_event_rate_30k.yaml" \
  "${resume_weights}" \
  "${manifest_root}/manifest.json" \
  "${runtime_lock}" \
  "${RMBENCH_BATTERY_LEROBOT}/meta/info.json" \
  "${MEMORYWAM_BATTERY_STATS}" \
  "${MEMORYWAM_BATTERY_CONTINUOUS_LATENTS}/manifest.json" \
  "${FASTWAM_ACTION_DIT_INIT}" \
  "${DIFFSYNTH_MODEL_BASE_PATH}/Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model-00001-of-00003.safetensors" \
  "${DIFFSYNTH_MODEL_BASE_PATH}/Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model-00002-of-00003.safetensors" \
  "${DIFFSYNTH_MODEL_BASE_PATH}/Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model-00003-of-00003.safetensors" \
  "${DIFFSYNTH_MODEL_BASE_PATH}/DiffSynth-Studio/Wan-Series-Converted-Safetensors/models_t5_umt5-xxl-enc-bf16.safetensors" \
  "${DIFFSYNTH_MODEL_BASE_PATH}/DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors"; do
  [[ -s "${required}" ]] || die "missing required artifact: ${required}"
done
[[ -d "${MEMORYWAM_BATTERY_TEXT_CACHE}" ]] || die "missing Battery text cache"

mkdir -p "$(dirname "${output_dir}")"
available_bytes="$(df --output=avail -B1 "$(dirname "${output_dir}")" | tail -1 | tr -d ' ')"
[[ "${available_bytes}" -ge 68719476736 ]] || die "output filesystem needs at least 64 GiB free"

cd "${repo_root}"
"${runtime_bin}/python" -m pytest -q \
  tests/test_event_conditioned_rate.py \
  tests/test_weight_only_resume_step.py

"${runtime_bin}/python" - "${resume_weights}" <<'PY'
import sys
from fastwam.utils.compact_checkpoint import validate_portable_checkpoint

validate_portable_checkpoint(sys.argv[1], expected_step=30000)
print("resume_checkpoint_validation=pass")
PY

resolved_config="$("${runtime_bin}/python" scripts/train.py \
  task=rmbench_battery_control_information_event_rate_30k \
  output_dir="${output_dir}" \
  resume="${resume_weights}" \
  +weight_only_resume_step=30000 \
  max_steps=50000 \
  save_every=0 \
  'save_steps=[40000,45000,50000]' \
  save_training_state=false \
  save_final_checkpoint=true \
  save_portable_fsdp_checkpoint=true \
  +warmup_ratio=0.0 \
  --cfg job)"
printf '%s\n' "${resolved_config}" | "${runtime_bin}/python" -c '
import sys
from omegaconf import OmegaConf

cfg = OmegaConf.create(sys.stdin.read())
expected = {
    "max_steps": 50000,
    "save_every": 0,
    "save_steps": [40000, 45000, 50000],
    "save_training_state": False,
    "save_final_checkpoint": True,
    "save_portable_fsdp_checkpoint": True,
    "weight_only_resume_step": 30000,
    "warmup_ratio": 0.0,
    "seed": 42,
    "batch_size": 1,
    "gradient_accumulation_steps": 1,
    "learning_rate": 2e-4,
    "lr_scheduler_type": "constant",
    "mixed_precision": "bf16",
    "native_cache_train_mode": "full",
    "memory_allocation_rule": "event_full_forced_half",
    "model.native_cache.mode": "layerwise",
    "model.native_cache.memory_tokens": 48,
    "model.native_cache.dynamic_tokens_per_frame": 8,
    "model.native_cache.group_size": 4,
    "model.native_cache.anchor_frames": 2,
    "model.native_cache.recent_frames": 4,
    "model.native_cache.recursive": False,
}
for path, want in expected.items():
    got = OmegaConf.select(cfg, path)
    if got != want:
        raise SystemExit(f"config mismatch {path}: {got!r} != {want!r}")
if str(cfg.resume) != sys.argv[1]:
    raise SystemExit(f"resume mismatch: {cfg.resume!r} != {sys.argv[1]!r}")
' "${resume_weights}"

echo "resume_mode=portable_weights_only_optimizer_reset"
echo "resume_weights=${resume_weights}"
echo "cumulative_start_step=30000 actual_additional_updates=20000"
echo "portable_steps=step_040000,step_045000,step_050000"
echo "save_training_state=false optimizer_scheduler_state_not_saved=true"
echo "learning_rate=2e-4 scheduler=constant restart_warmup=disabled"
echo "output_dir=${output_dir}"
echo "gpus=${CUDA_VISIBLE_DEVICES}"
if [[ "${preflight_only}" == "1" ]]; then
  echo "preflight_status=ok"
  exit 0
fi

mkdir -p "${output_dir}/source_snapshot"
for source_file in \
  "${repo_root}/src/fastwam/trainer.py" \
  "${repo_root}/src/fastwam/models/wan22/fastwam.py" \
  "${repo_root}/src/fastwam/memory/native_cache.py" \
  "${repo_root}/src/fastwam/memory/event_conditioned_rate.py" \
  "${repo_root}/configs/model/fastwam_event_conditioned_rate.yaml" \
  "${repo_root}/configs/data/rmbench_battery_control_information.yaml" \
  "${repo_root}/configs/task/rmbench_battery_control_information_event_rate_30k.yaml" \
  "${repo_root}/ops/resume_battery_v6_event_full_forced_half_30k_to50k.sh"; do
  cp "${source_file}" "${output_dir}/source_snapshot/$(basename "${source_file}")"
done
sha256sum "${output_dir}"/source_snapshot/* > "${output_dir}/source_snapshot/SHA256SUMS"

exec > >(tee -a "${output_dir}/train.log") 2>&1
echo "training_resume_start=$(date --iso-8601=seconds)"
echo "source_snapshot=${output_dir}/source_snapshot/SHA256SUMS"
accelerate launch \
  --config_file scripts/accelerate_configs/accelerate_fsdp_full_shard.yaml \
  --num_processes 8 \
  scripts/train.py \
  task=rmbench_battery_control_information_event_rate_30k \
  output_dir="${output_dir}" \
  resume="${resume_weights}" \
  +weight_only_resume_step=30000 \
  max_steps=50000 \
  save_every=0 \
  'save_steps=[40000,45000,50000]' \
  save_training_state=false \
  save_final_checkpoint=true \
  save_portable_fsdp_checkpoint=true \
  +warmup_ratio=0.0
