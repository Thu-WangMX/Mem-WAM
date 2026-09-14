#!/usr/bin/env bash
set -euo pipefail

repo_root="${REPO_ROOT:-/mnt/vepfs02/output/kevin.wang/memorywam/code/fastwam_event_conditioned_rate}"
runtime_bin="${RUNTIME_BIN:-/root/kevin_wang/envs/fastwam_guidemem/bin}"
memorywam_root="${MEMORYWAM_ROOT:-/mnt/vepfs02/output/kevin.wang/memorywam}"
battery_root="${BATTERY_ROOT:-/mnt/vepfs02/output/kevin_wang/memorywam/data/fastwam_fullkv_remaining8_rgb_v3/battery_try}"
output_dir="${OUTPUT_DIR:-/mnt/vepfs01/output/kevin.wang/memorywam/train/control_information_v6_event_full_forced_half_battery_e2e_30k_seed42}"
manifest_root="${MANIFEST_ROOT:-${memorywam_root}/data/battery_control_information_planning_segments_v3_causal_online}"
runtime_lock="${RUNTIME_LOCK:-${memorywam_root}/analysis/putback_control_information_selector_v3/selector_runtime/locked_selector.json}"
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

for required in \
  "${repo_root}/scripts/train.py" \
  "${repo_root}/tests/test_event_conditioned_rate.py" \
  "${repo_root}/configs/model/fastwam_event_conditioned_rate.yaml" \
  "${repo_root}/configs/data/rmbench_battery_control_information.yaml" \
  "${repo_root}/configs/task/rmbench_battery_control_information_event_rate_30k.yaml" \
  "${RMBENCH_BATTERY_LEROBOT}/meta/info.json" \
  "${MEMORYWAM_BATTERY_STATS}" \
  "${MEMORYWAM_BATTERY_CONTINUOUS_LATENTS}/manifest.json" \
  "${FASTWAM_ACTION_DIT_INIT}" \
  "${manifest_root}/manifest.json" \
  "${runtime_lock}" \
  "${DIFFSYNTH_MODEL_BASE_PATH}/Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model-00001-of-00003.safetensors" \
  "${DIFFSYNTH_MODEL_BASE_PATH}/Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model-00002-of-00003.safetensors" \
  "${DIFFSYNTH_MODEL_BASE_PATH}/Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model-00003-of-00003.safetensors" \
  "${DIFFSYNTH_MODEL_BASE_PATH}/DiffSynth-Studio/Wan-Series-Converted-Safetensors/models_t5_umt5-xxl-enc-bf16.safetensors" \
  "${DIFFSYNTH_MODEL_BASE_PATH}/DiffSynth-Studio/Wan-Series-Converted-Safetensors/Wan2.2_VAE.safetensors"; do
  [[ -s "${required}" ]] || die "missing required artifact: ${required}"
done
[[ -d "${MEMORYWAM_BATTERY_TEXT_CACHE}" ]] || die "missing Battery text cache"

IFS=',' read -r -a visible_gpus <<<"${CUDA_VISIBLE_DEVICES}"
[[ "${#visible_gpus[@]}" -eq 8 ]] || die "formal run requires exactly eight GPUs"
[[ "$(printf '%s\n' "${visible_gpus[@]}" | sort -u | wc -l | tr -d ' ')" -eq 8 ]] || die "GPU indices must be unique"
[[ ! -e "${output_dir}" ]] || die "refusing to overwrite output: ${output_dir}"

cd "${repo_root}"
"${runtime_bin}/python" -m pytest -q tests/test_event_conditioned_rate.py

manifest_report="$("${runtime_bin}/python" - "${manifest_root}" "${runtime_lock}" <<'PY'
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

from fastwam.memory.dynamic_surprise import DynamicSurpriseManifestStore
from fastwam.memory.event_conditioned_rate import memory_tokens_for_segment

manifest_root = Path(sys.argv[1]).resolve()
runtime_lock = Path(sys.argv[2]).resolve()
metadata = json.loads((manifest_root / "manifest.json").read_text())["metadata"]
runtime_sha = hashlib.sha256(runtime_lock.read_bytes()).hexdigest()
assert metadata["complete"] is True
assert metadata["task"] == "battery_try"
assert metadata["episode_count"] == 50
assert metadata["boundary_step"] == -1
assert metadata["detector_stride"] == 4
assert metadata["replan_stride"] == 16
assert metadata["selector"] == "locked_segment_relative_control_information_v3"
assert metadata["selector_transfer"] == "putback_locked_v3_to_battery_zero_refit"
assert metadata["runtime_lock_sha256"] == runtime_sha
assert metadata["retroactive_boundary_count"] == 0

store = DynamicSurpriseManifestStore(
    manifest_root,
    expected_boundary_step=-1,
    expected_episode_count=50,
    expected_task="battery_try",
)
lengths = Counter()
reasons = Counter()
token_histogram = Counter()
full_rate_total = 0
event_rate_total = 0
for episode in range(50):
    records = store.segment_records_for_episode(episode)
    for record in records[:-1]:
        span = record.end - record.start
        lengths[span] += 1
        reasons[record.reason] += 1
        allocated = memory_tokens_for_segment(span, record.reason)
        token_histogram[allocated] += 1
        full_rate_total += 8 * span
        event_rate_total += allocated
assert lengths and min(lengths) >= 2 and max(lengths) <= 8
assert set(reasons) == {"segment_relative_control_information", "forced_maximum"}
print("selector=locked_segment_relative_control_information_v3")
print("selector_transfer=putback_locked_v3_to_battery_zero_refit")
print("manifest_segment_lengths=" + json.dumps(dict(sorted(lengths.items()))))
print("boundary_reasons=" + json.dumps(dict(sorted(reasons.items()))))
print("memory_token_histogram=" + json.dumps(dict(sorted(token_histogram.items()))))
print(f"token_reduction_vs_k8l={1.0 - event_rate_total / full_rate_total:.6%}")
print("runtime_lock_sha256=" + runtime_sha)
PY
)"
printf '%s\n' "${manifest_report}"

resolved_config="$("${runtime_bin}/python" scripts/train.py task=rmbench_battery_control_information_event_rate_30k --cfg job)"
printf '%s\n' "${resolved_config}" | "${runtime_bin}/python" -c '
import sys
from omegaconf import OmegaConf
cfg = OmegaConf.create(sys.stdin.read())
expected = {
    "max_steps": 30000,
    "save_every": 10000,
    "save_steps": [5000],
    "save_training_state": False,
    "save_final_checkpoint": True,
    "save_portable_fsdp_checkpoint": True,
    "seed": 42,
    "batch_size": 1,
    "gradient_accumulation_steps": 1,
    "learning_rate": 2e-4,
    "mixed_precision": "bf16",
    "native_cache_train_mode": "full",
    "memory_allocation_rule": "event_full_forced_half",
    "data.train.minimum_history_frames": 1,
    "data.train.expected_manifest_task": "battery_try",
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
'

echo "task=rmbench_battery_control_information_event_rate_30k"
echo "method=v6 event=K8L forced=K8ceil(L/2) max_tokens=48"
echo "anchor_frames=2 recent_frames=4 recursive=false"
echo "max_steps=30000 checkpoint_steps=5000,10000,20000,30000"
echo "save_training_state=false portable_weights_only=true"
echo "initialization=official_fullkv_actiondit"
echo "gpus=${CUDA_VISIBLE_DEVICES} output_dir=${output_dir}"
if [[ "${preflight_only}" == "1" ]]; then
  echo "preflight_status=ok"
  exit 0
fi

mkdir -p "${output_dir}/source_snapshot"
for source_file in \
  "${repo_root}/src/fastwam/models/wan22/fastwam.py" \
  "${repo_root}/src/fastwam/memory/native_cache.py" \
  "${repo_root}/src/fastwam/memory/dynamic_surprise.py" \
  "${repo_root}/src/fastwam/datasets/lerobot/dynamic_surprise_dataset.py" \
  "${repo_root}/src/fastwam/memory/event_conditioned_rate.py" \
  "${repo_root}/src/fastwam/memory/planning_aligned_manifest.py" \
  "${repo_root}/configs/model/fastwam_event_conditioned_rate.yaml" \
  "${repo_root}/configs/data/rmbench_battery_control_information.yaml" \
  "${repo_root}/configs/task/rmbench_battery_control_information_event_rate_30k.yaml" \
  "${repo_root}/ops/train_battery_control_information_event_rate_30k.sh"; do
  cp "${source_file}" "${output_dir}/source_snapshot/$(basename "${source_file}")"
done
sha256sum "${output_dir}"/source_snapshot/* > "${output_dir}/source_snapshot/SHA256SUMS"
exec > >(tee -a "${output_dir}/train.log") 2>&1
echo "training_start=$(date --iso-8601=seconds)"
echo "source_snapshot=${output_dir}/source_snapshot/SHA256SUMS"
accelerate launch \
  --config_file scripts/accelerate_configs/accelerate_fsdp_full_shard.yaml \
  --num_processes 8 \
  scripts/train.py task=rmbench_battery_control_information_event_rate_30k
