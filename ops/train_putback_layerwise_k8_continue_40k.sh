#!/usr/bin/env bash
set -euo pipefail

repo_root="${REPO_ROOT:-/mnt/vepfs02/output/kevin_wang/memorywam/code/fastwam_native_cache_consolidation}"
runtime_bin="${RUNTIME_BIN:-/root/kevin_wang/envs/fastwam_guidemem/bin}"
output_dir="${OUTPUT_DIR:-/mnt/vepfs02/output/kevin_wang/memorywam/train/layerwise_block_memory_k8_putback_e2e_25k_seed42}"
preflight_only="${PREFLIGHT_ONLY:-0}"
resume_state="${RESUME_STATE:-${output_dir}/checkpoints/state/step_025000}"

export PATH="${runtime_bin}:${PATH}"
export PYTHONPATH="${repo_root}/src:${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-/mnt/vepfs02/datasets/kevin_wang/code/projects/FastWAM/checkpoints}"
export DIFFSYNTH_SKIP_DOWNLOAD=true
export RMBENCH_PUTBACK_LEROBOT="${RMBENCH_PUTBACK_LEROBOT:-/mnt/vepfs02/output/kevin_wang/memorywam/data/rmbench_lerobot_v21_rgb_v3/put_back_block}"
export MEMORYWAM_PUTBACK_STATS="${MEMORYWAM_PUTBACK_STATS:-/mnt/vepfs02/output/kevin_wang/memorywam/data/fastwam_putback_stats_rgb_v3/dataset_stats.json}"
export MEMORYWAM_PUTBACK_CONTINUOUS_LATENTS="${MEMORYWAM_PUTBACK_CONTINUOUS_LATENTS:-/mnt/vepfs02/output/kevin_wang/memorywam/data/fastwam_putback_continuous_episode_stride16_v4}"
export MEMORYWAM_PUTBACK_TEXT_CACHE="${MEMORYWAM_PUTBACK_TEXT_CACHE:-/mnt/vepfs02/output/kevin_wang/memorywam/data/fastwam_putback_text_cache_rgb_v3}"
export FASTWAM_ACTION_DIT_INIT="${FASTWAM_ACTION_DIT_INIT:-/mnt/vepfs02/output/kevin_wang/memorywam/model_assets/fastwam_fullkv_official_init/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt}"
export LAYERWISE_K8_CONTINUE_40K_OUTPUT="${output_dir}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PYTHONUTF8=1
export LANG=C.UTF-8
export LC_ALL=C.UTF-8

die() {
  echo "ERROR: $*" >&2
  exit 2
}

[[ "${preflight_only}" == "0" || "${preflight_only}" == "1" ]] || die "PREFLIGHT_ONLY must be 0 or 1"
[[ -d "${repo_root}" ]] || die "Repository not found: ${repo_root}"
[[ -x "${runtime_bin}/python" ]] || die "Python is not executable: ${runtime_bin}/python"
[[ -d "${DIFFSYNTH_MODEL_BASE_PATH}" ]] || die "Model asset root not found: ${DIFFSYNTH_MODEL_BASE_PATH}"

putback_asset_root="${PUTBACK_ASSET_ROOT:-$(dirname "${RMBENCH_PUTBACK_LEROBOT}")}"
build_manifest="${putback_asset_root}/_build_manifest.json"
rgb_contract="${putback_asset_root}/_build_logs/rgb_contract.json"
latent_manifest="${MEMORYWAM_PUTBACK_CONTINUOUS_LATENTS}/manifest.json"

for required in \
  "${build_manifest}" \
  "${rgb_contract}" \
  "${RMBENCH_PUTBACK_LEROBOT}/meta/info.json" \
  "${MEMORYWAM_PUTBACK_STATS}" \
  "${latent_manifest}" \
  "${FASTWAM_ACTION_DIT_INIT}"; do
  [[ -s "${required}" ]] || die "Missing or empty required artifact: ${required}"
done
[[ -d "${MEMORYWAM_PUTBACK_TEXT_CACHE}" ]] || die "Text cache directory not found: ${MEMORYWAM_PUTBACK_TEXT_CACHE}"
find -L "${MEMORYWAM_PUTBACK_TEXT_CACHE}" -type f -size +0c -print -quit | grep -q . || die "Text cache contains no nonempty files: ${MEMORYWAM_PUTBACK_TEXT_CACHE}"

episode_count="$("${runtime_bin}/python" - "${build_manifest}" "${rgb_contract}" "${RMBENCH_PUTBACK_LEROBOT}/meta/info.json" "${latent_manifest}" <<'PY'
import json
import sys
from pathlib import Path

build_path, rgb_path, info_path, latent_path = map(Path, sys.argv[1:])
build = json.loads(build_path.read_text())
rgb = json.loads(rgb_path.read_text())
info = json.loads(info_path.read_text())
latent = json.loads(latent_path.read_text()).get("metadata", {})

expected_build = {"color_contract": "simulator_rgb_preserved", "episodes_per_task": 50}
expected_latent = {
    "schema_version": "fastwam_full_kv_continuous_episode_vae_latents_v4",
    "complete": True,
    "episode_count": 50,
    "replan_stride": 16,
    "color_contract": "simulator_rgb_preserved",
}
for key, expected in expected_build.items():
    if build.get(key) != expected:
        raise SystemExit(f"{build_path}: expected {key}={expected!r}, got {build.get(key)!r}")
if rgb.get("passed") is not True:
    raise SystemExit(f"{rgb_path}: RGB contract did not pass")
if int(info.get("total_episodes", -1)) != 50:
    raise SystemExit(f"{info_path}: expected total_episodes=50, got {info.get('total_episodes')!r}")
for key, expected in expected_latent.items():
    if latent.get(key) != expected:
        raise SystemExit(f"{latent_path}: expected {key}={expected!r}, got {latent.get(key)!r}")
print(50)
PY
)"

IFS=',' read -r -a visible_gpus <<<"${CUDA_VISIBLE_DEVICES}"
[[ ${#visible_gpus[@]} -eq 8 ]] || die "Formal run requires exactly eight visible GPUs: ${CUDA_VISIBLE_DEVICES}"
declare -A seen_gpus=()
for gpu in "${visible_gpus[@]}"; do
  [[ "${gpu}" =~ ^[0-9]+$ ]] || die "GPU identifiers must be numeric indices, got: ${gpu}"
  [[ -z "${seen_gpus[$gpu]:-}" ]] || die "GPU index is repeated: ${gpu}"
  seen_gpus[$gpu]=1
done

resume_args=()
[[ -d "${output_dir}" ]] || die "Continuation output directory not found: ${output_dir}"
[[ -d "${resume_state}" ]] || die "Resume state directory not found: ${resume_state}"
normalized_output="$(readlink -f "${output_dir}")"
normalized_state="$(readlink -f "${resume_state}")"
[[ "${normalized_state}/" == "${normalized_output}/"* ]] || die "Resume state must belong to OUTPUT_DIR: ${resume_state}"
[[ -s "${resume_state}/trainer_state.json" && -s "${resume_state}/scheduler.bin" ]] || die "Incomplete resume state: ${resume_state}"
[[ -d "${resume_state}/pytorch_model_fsdp_0" && -d "${resume_state}/optimizer_0" ]] || die "Incomplete resume state: ${resume_state}"
model_shards="$(find "${resume_state}/pytorch_model_fsdp_0" -maxdepth 1 -name '*.distcp' -size +0c | wc -l | tr -d ' ')"
optimizer_shards="$(find "${resume_state}/optimizer_0" -maxdepth 1 -name '*.distcp' -size +0c | wc -l | tr -d ' ')"
random_states="$(find "${resume_state}" -maxdepth 1 -name 'random_states_*.pkl' -size +0c | wc -l | tr -d ' ')"
[[ "${model_shards}" -ge 8 && "${optimizer_shards}" -ge 8 && "${random_states}" -ge 8 ]] || die "Incomplete resume state: ${resume_state}"
resume_step="$("${runtime_bin}/python" - "${resume_state}/trainer_state.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
step = int(json.loads(path.read_text(encoding="utf-8"))["global_step"])
if step != 25000:
    raise SystemExit(f"expected global_step=25000, got {step}")
print(step)
PY
)" || die "Resume trainer state expected global_step=25000"
for target_step in 030000 035000 040000; do
  [[ ! -e "${output_dir}/checkpoints/state/step_${target_step}" ]] || die "Target training state already exists: step_${target_step}"
  [[ ! -e "${output_dir}/checkpoints/weights/step_${target_step}.pt" ]] || die "Target weights already exist: step_${target_step}"
done
resume_value="${normalized_state}"
resume_args+=("resume=${normalized_state}")

cd "${repo_root}"
resolved_config="$("${runtime_bin}/python" scripts/train.py task=rmbench_putback_layerwise_k8_continue_40k --cfg job)"
config_contract="$(printf '%s\n' "${resolved_config}" | "${runtime_bin}/python" -c '
import sys
from omegaconf import OmegaConf

cfg = OmegaConf.create(sys.stdin.read())
expected = {
    "batch_size": 1,
    "gradient_accumulation_steps": 1,
    "learning_rate": 2.0e-4,
    "lr_scheduler_type": "constant",
    "weight_decay": 1.0e-2,
    "max_grad_norm": 1.0,
    "mixed_precision": "bf16",
    "seed": 42,
    "max_steps": 40000,
    "save_every": 5000,
    "save_steps": [],
    "save_final_checkpoint": True,
    "save_training_state": True,
    "save_portable_fsdp_checkpoint": True,
    "keep_training_state_checkpoints": 1,
    "resume": None,
    "native_cache_train_mode": "full",
    "data.train.minimum_history_frames": 1,
    "model.native_cache.mode": "layerwise",
    "model.native_cache.memory_tokens": 8,
    "model.native_cache.group_size": 4,
    "model.native_cache.anchor_frames": 2,
    "model.native_cache.recent_frames": 4,
    "model.native_cache.recursive": False,
    "model.mot_checkpoint_mixed_attn": True,
    "model.video_scheduler.training_sampling_scheme": "logit_normal",
    "model.action_scheduler.training_sampling_scheme": "logit_normal",
}
for path, want in expected.items():
    got = OmegaConf.select(cfg, path)
    if got != want:
        raise SystemExit(f"resolved config mismatch: {path}: expected {want!r}, got {got!r}")
print("layerwise\t8\t4\t2\t4\tfalse\t1\t8")
')"
IFS=$'\t' read -r memory_mode memory_tokens group_size anchor_frames recent_frames recursive minimum_history_frames global_batch_size <<<"${config_contract}"

echo "host=$(hostname)"
echo "repo_root=${repo_root}"
echo "task=rmbench_putback_layerwise_k8_continue_40k"
echo "dataset=${RMBENCH_PUTBACK_LEROBOT}"
echo "episodes=${episode_count}"
echo "resume_step=${resume_step}"
echo "max_steps=40000"
echo "additional_steps=15000"
echo "checkpoint_steps=30000,35000,40000"
echo "memory_mode=${memory_mode}"
echo "memory_tokens=${memory_tokens}"
echo "group_size=${group_size}"
echo "anchor_frames=${anchor_frames}"
echo "recent_frames=${recent_frames}"
echo "recursive=${recursive}"
echo "minimum_history_frames=${minimum_history_frames}"
echo "global_batch_size=${global_batch_size}"
echo "gradient_checkpointing=true"
echo "resume=${resume_value}"
echo "gpus=${CUDA_VISIBLE_DEVICES}"
echo "output_dir=${output_dir}"

if [[ "${preflight_only}" == "1" ]]; then
  echo "preflight_status=ok"
  exit 0
fi

command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi is unavailable"
physical_gpu_count="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')"
[[ "${physical_gpu_count}" -ge 8 ]] || die "Host exposes only ${physical_gpu_count} physical GPUs"

mkdir -p "${output_dir}"
exec > >(tee -a "${output_dir}/train.log") 2>&1
echo "training_start=$(date --iso-8601=seconds)"
accelerate launch \
  --config_file scripts/accelerate_configs/accelerate_fsdp_full_shard.yaml \
  --num_processes 8 \
  scripts/train.py \
  task=rmbench_putback_layerwise_k8_continue_40k \
  "${resume_args[@]}"

