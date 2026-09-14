#!/usr/bin/env bash
set -euo pipefail

repo_root="${REPO_ROOT:-/mnt/vepfs02/output/kevin_wang/memorywam/code/fastwam_wrist_latent_event_memory}"
runtime_bin="${RUNTIME_BIN:-/root/kevin_wang/envs/fastwam_guidemem/bin}"
output_dir="${OUTPUT_DIR:-/mnt/vepfs01/output/kevin_wang/memorywam/train/wrist_latent_event_memory_putback_k8_40k_seed42}"
manifest_root="${WRIST_EVENT_MANIFEST:-/mnt/vepfs02/output/kevin_wang/memorywam/analysis/putback_wrist_latent_event_predictor_v1/boundary_manifest}"
resume_state="${RESUME_STATE:-${output_dir}/checkpoints/state/step_005000}"
preflight_only="${PREFLIGHT_ONLY:-0}"

export REPO_ROOT="${repo_root}" RUNTIME_BIN="${runtime_bin}"
export WRIST_EVENT_OUTPUT="${output_dir}"
export WRIST_EVENT_MANIFEST="${manifest_root}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PYTHONUTF8=1 LANG=C.UTF-8 LC_ALL=C.UTF-8
export PYTHONDONTWRITEBYTECODE=1
source "${repo_root}/ops/putback_dynamic_surprise_env.sh"

die() {
  echo "ERROR: $*" >&2
  exit 2
}

[[ "${preflight_only}" == "0" || "${preflight_only}" == "1" ]] || die "PREFLIGHT_ONLY must be 0 or 1"
[[ -d "${repo_root}" ]] || die "repository not found: ${repo_root}"
[[ -x "${runtime_bin}/python" ]] || die "Python is not executable: ${runtime_bin}/python"
[[ -d "${output_dir}" ]] || die "continuation output directory not found: ${output_dir}"

for required in \
  "${RMBENCH_PUTBACK_LEROBOT}/meta/info.json" \
  "${MEMORYWAM_PUTBACK_STATS}" \
  "${MEMORYWAM_PUTBACK_CONTINUOUS_LATENTS}/manifest.json" \
  "${MEMORYWAM_PUTBACK_TEXT_CACHE}" \
  "${FASTWAM_ACTION_DIT_INIT}" \
  "${manifest_root}/manifest.json"; do
  [[ -e "${required}" ]] || die "missing required artifact: ${required}"
done

IFS=',' read -r -a visible_gpus <<<"${CUDA_VISIBLE_DEVICES}"
[[ ${#visible_gpus[@]} -eq 8 ]] || die "formal resume requires exactly eight visible GPUs"
declare -A seen_gpus=()
for gpu in "${visible_gpus[@]}"; do
  [[ "${gpu}" =~ ^[0-9]+$ ]] || die "GPU identifiers must be numeric: ${gpu}"
  [[ -z "${seen_gpus[$gpu]:-}" ]] || die "GPU index is repeated: ${gpu}"
  seen_gpus[$gpu]=1
done

normalized_output="$(readlink -f "${output_dir}")"
[[ -d "${resume_state}" ]] || die "resume state directory not found: ${resume_state}"
normalized_state="$(readlink -f "${resume_state}")"
[[ "${normalized_state}/" == "${normalized_output}/"* ]] || die "resume state must belong to OUTPUT_DIR: ${resume_state}"
[[ "$(basename "${normalized_state}")" == "step_005000" ]] || die "resume state must be step_005000: ${resume_state}"
[[ -s "${normalized_state}/trainer_state.json" && -s "${normalized_state}/scheduler.bin" ]] || die "Incomplete resume state: ${resume_state}"
[[ -d "${normalized_state}/pytorch_model_fsdp_0" && -d "${normalized_state}/optimizer_0" ]] || die "Incomplete resume state: ${resume_state}"
model_shards="$(find "${normalized_state}/pytorch_model_fsdp_0" -maxdepth 1 -name '*.distcp' -size +0c | wc -l | tr -d ' ')"
optimizer_shards="$(find "${normalized_state}/optimizer_0" -maxdepth 1 -name '*.distcp' -size +0c | wc -l | tr -d ' ')"
random_states="$(find "${normalized_state}" -maxdepth 1 -name 'random_states_*.pkl' -size +0c | wc -l | tr -d ' ')"
[[ "${model_shards}" -ge 8 && "${optimizer_shards}" -ge 8 && "${random_states}" -ge 8 ]] || die "Incomplete resume state: ${resume_state}"

resume_step="$(${runtime_bin}/python - "${normalized_state}/trainer_state.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
step = int(json.loads(path.read_text(encoding="utf-8"))["global_step"])
if step != 5000:
    raise SystemExit(f"expected global_step=5000, got {step}")
print(step)
PY
)" || die "resume trainer state expected global_step=5000"

for target_step in 010000 015000 020000 025000 030000 035000 040000; do
  [[ ! -e "${output_dir}/checkpoints/state/step_${target_step}" ]] || die "target training state already exists: step_${target_step}"
  [[ ! -e "${output_dir}/checkpoints/weights/step_${target_step}.pt" ]] || die "target weights already exist: step_${target_step}"
done

cd "${repo_root}"
episode_count="$(${runtime_bin}/python - "${manifest_root}" <<'PY'
import sys
from fastwam.memory.wrist_event import WristEventManifestStore

store = WristEventManifestStore(sys.argv[1], expected_episode_count=50)
for episode in range(50):
    if not store.segments_for_episode(episode):
        raise SystemExit(f"episode {episode} has no segments")
print(store.metadata["episode_count"])
PY
)"

resolved_config="$(${runtime_bin}/python scripts/train.py task=rmbench_putback_wrist_event_k8_40k --cfg job)"
config_contract="$(printf '%s\n' "${resolved_config}" | ${runtime_bin}/python -c '
import sys
from omegaconf import OmegaConf

cfg = OmegaConf.create(sys.stdin.read())
expected = {
    "max_steps": 40000,
    "save_every": 5000,
    "save_steps": [],
    "batch_size": 1,
    "gradient_accumulation_steps": 1,
    "learning_rate": 2.0e-4,
    "lr_scheduler_type": "constant",
    "weight_decay": 1.0e-2,
    "max_grad_norm": 1.0,
    "mixed_precision": "bf16",
    "seed": 42,
    "resume": None,
    "save_training_state": True,
    "save_portable_fsdp_checkpoint": True,
    "keep_training_state_checkpoints": 1,
    "process_group_timeout_seconds": 7200,
    "native_cache_train_mode": "full",
    "data.train._target_": "fastwam.datasets.lerobot.wrist_event_dataset.WristEventRobotVideoDataset",
    "data.train.minimum_history_frames": 1,
    "model.native_cache.enabled": True,
    "model.native_cache.mode": "layerwise",
    "model.native_cache.memory_tokens": 8,
    "model.native_cache.group_size": 4,
    "model.native_cache.anchor_frames": 2,
    "model.native_cache.recent_frames": 4,
    "model.native_cache.recursive": False,
    "model.video_scheduler.training_sampling_scheme": "logit_normal",
    "model.action_scheduler.training_sampling_scheme": "logit_normal",
}
for path, want in expected.items():
    got = OmegaConf.select(cfg, path)
    if got != want:
        raise SystemExit(f"resolved config mismatch: {path}: expected {want!r}, got {got!r}")
print("layerwise\t8\t4\t2\t4\tfalse\t1\t8\t7200")
')"
IFS=$'\t' read -r memory_mode memory_tokens group_size anchor_frames recent_frames recursive minimum_history_frames global_batch_size process_group_timeout_seconds <<<"${config_contract}"

echo "host=$(hostname)"
echo "repo_root=${repo_root}"
echo "task=rmbench_putback_wrist_event_k8_40k"
echo "dataset=${RMBENCH_PUTBACK_LEROBOT}"
echo "episodes=${episode_count}"
echo "resume_step=${resume_step}"
echo "resume=${normalized_state}"
echo "max_steps=40000"
echo "additional_steps=35000"
echo "checkpoint_steps=10000,15000,20000,25000,30000,35000,40000"
echo "process_group_timeout_seconds=${process_group_timeout_seconds}"
echo "memory_mode=${memory_mode}"
echo "memory_tokens=${memory_tokens}"
echo "group_size=${group_size}"
echo "anchor_frames=${anchor_frames}"
echo "recent_frames=${recent_frames}"
echo "recursive=${recursive}"
echo "minimum_history_frames=${minimum_history_frames}"
echo "global_batch_size=${global_batch_size}"
echo "gpus=${CUDA_VISIBLE_DEVICES}"
echo "output_dir=${output_dir}"

if [[ "${preflight_only}" == "1" ]]; then
  echo "preflight_status=ok"
  exit 0
fi

command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi is unavailable"
physical_gpu_count="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')"
[[ "${physical_gpu_count}" -ge 8 ]] || die "host exposes only ${physical_gpu_count} physical GPUs"

exec > >(tee -a "${output_dir}/train.log") 2>&1
echo "resume_training_start=$(date --iso-8601=seconds)"
exec accelerate launch \
  --config_file scripts/accelerate_configs/accelerate_fsdp_full_shard.yaml \
  --num_processes 8 \
  scripts/train.py \
  task=rmbench_putback_wrist_event_k8_40k \
  "resume=${normalized_state}"
