#!/usr/bin/env bash
set -euo pipefail

repo_root="${REPO_ROOT:-/mnt/vepfs02/output/kevin_wang/memorywam/code/fastwam_wrist_latent_event_memory}"
runtime_bin="${RUNTIME_BIN:-/root/kevin_wang/envs/fastwam_guidemem/bin}"
output_dir="${OUTPUT_DIR:-/mnt/vepfs01/output/kevin_wang/memorywam/train/wrist_latent_event_memory_putback_k8_40k_seed42}"
manifest_root="${WRIST_EVENT_MANIFEST:-/mnt/vepfs02/output/kevin_wang/memorywam/analysis/putback_wrist_latent_event_predictor_v1/boundary_manifest}"
preflight_only="${PREFLIGHT_ONLY:-0}"
min_free_gib="${MIN_FREE_GIB:-180}"

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
[[ "${min_free_gib}" =~ ^[1-9][0-9]*$ ]] || die "MIN_FREE_GIB must be a positive integer"
[[ -d "${repo_root}" ]] || die "repository not found: ${repo_root}"
[[ -x "${runtime_bin}/python" ]] || die "Python is not executable: ${runtime_bin}/python"

output_parent="$(dirname "${output_dir}")"
[[ -d "${output_parent}" ]] || die "output parent not found: ${output_parent}"
available_kib="$(df -Pk "${output_parent}" | awk 'NR == 2 {print $4}')"
required_kib=$((min_free_gib * 1024 * 1024))
[[ "${available_kib}" -ge "${required_kib}" ]] || die \
  "output filesystem lacks checkpoint budget: available=$((available_kib / 1024 / 1024))GiB required=${min_free_gib}GiB path=${output_parent}"

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
[[ ${#visible_gpus[@]} -eq 8 ]] || die "formal run requires exactly eight visible GPUs"
declare -A seen_gpus=()
for gpu in "${visible_gpus[@]}"; do
  [[ "${gpu}" =~ ^[0-9]+$ ]] || die "GPU identifiers must be numeric: ${gpu}"
  [[ -z "${seen_gpus[$gpu]:-}" ]] || die "GPU index is repeated: ${gpu}"
  seen_gpus[$gpu]=1
done

cd "${repo_root}"
episode_count="$("${runtime_bin}/python" - "${manifest_root}" <<'PY'
import sys
from fastwam.memory.wrist_event import WristEventManifestStore

store = WristEventManifestStore(sys.argv[1], expected_episode_count=50)
for episode in range(50):
    segments = store.segments_for_episode(episode)
    if not segments:
        raise SystemExit(f"episode {episode} has no segments")
print(store.metadata["episode_count"])
PY
)"

resolved_config="$("${runtime_bin}/python" scripts/train.py task=rmbench_putback_wrist_event_k8_40k --cfg job)"
printf '%s\n' "${resolved_config}" | "${runtime_bin}/python" -c '
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
    "mixed_precision": "bf16",
    "seed": 42,
    "native_cache_train_mode": "full",
    "data.train._target_": "fastwam.datasets.lerobot.wrist_event_dataset.WristEventRobotVideoDataset",
    "data.train.minimum_history_frames": 1,
    "model.native_cache.mode": "layerwise",
    "model.native_cache.memory_tokens": 8,
    "model.native_cache.group_size": 4,
    "model.native_cache.anchor_frames": 2,
    "model.native_cache.recent_frames": 4,
    "model.native_cache.recursive": False,
}
for path, want in expected.items():
    got = OmegaConf.select(cfg, path)
    if got != want:
        raise SystemExit(f"resolved config mismatch: {path}: expected {want!r}, got {got!r}")
'

echo "preflight_ok episode_count=${episode_count} task=put_back_block memory_tokens=8 max_steps=40000"
echo "manifest=${manifest_root}"
echo "checkpoint_steps=5000,10000,15000,20000,25000,30000,35000,40000"
echo "gpus=${CUDA_VISIBLE_DEVICES}"
echo "output_dir=${output_dir}"
echo "checkpoint_budget_gib=${min_free_gib}"

if [[ "${preflight_only}" == "1" ]]; then
  exit 0
fi
[[ ! -e "${output_dir}" ]] || die "refusing to overwrite existing output: ${output_dir}"
command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi is unavailable"
[[ "$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')" -ge 8 ]] || die "host exposes fewer than eight GPUs"

mkdir -p "${output_dir}"
exec > >(tee -a "${output_dir}/train.log") 2>&1
echo "training_start=$(date --iso-8601=seconds)"
exec accelerate launch \
  --config_file scripts/accelerate_configs/accelerate_fsdp_full_shard.yaml \
  --num_processes 8 \
  scripts/train.py \
  task=rmbench_putback_wrist_event_k8_40k
