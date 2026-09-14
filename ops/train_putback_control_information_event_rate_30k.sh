#!/usr/bin/env bash
set -euo pipefail

repo_root="${REPO_ROOT:-/mnt/vepfs02/output/kevin.wang/memorywam/code/fastwam_event_conditioned_rate}"
runtime_bin="${RUNTIME_BIN:-/root/kevin_wang/envs/fastwam_guidemem/bin}"
output_dir="${OUTPUT_DIR:-/mnt/vepfs02/output/kevin.wang/memorywam/train/control_information_v4_event_full_forced_half_putback_e2e_30k_seed42}"
manifest_root="${MANIFEST_ROOT:-/mnt/vepfs02/output/kevin.wang/memorywam/data/putback_control_information_planning_segments_v3}"
runtime_lock="${RUNTIME_LOCK:-/mnt/vepfs02/output/kevin.wang/memorywam/analysis/putback_control_information_selector_v3/selector_runtime/locked_selector.json}"
preflight_only="${PREFLIGHT_ONLY:-0}"

export PATH="${runtime_bin}:${PATH}"
export PYTHONPATH="${repo_root}/src:${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-/mnt/vepfs02/output/kevin.wang/memorywam/model_assets/diffsynth_official}"
export DIFFSYNTH_SKIP_DOWNLOAD=true
export RMBENCH_PUTBACK_LEROBOT="${RMBENCH_PUTBACK_LEROBOT:-/mnt/vepfs02/output/kevin_wang/memorywam/data/rmbench_lerobot_v21_rgb_v3/put_back_block}"
export MEMORYWAM_PUTBACK_STATS="${MEMORYWAM_PUTBACK_STATS:-/mnt/vepfs02/output/kevin_wang/memorywam/data/fastwam_putback_stats_rgb_v3/dataset_stats.json}"
export MEMORYWAM_PUTBACK_CONTINUOUS_LATENTS="${MEMORYWAM_PUTBACK_CONTINUOUS_LATENTS:-/mnt/vepfs02/output/kevin_wang/memorywam/data/fastwam_putback_continuous_episode_stride16_v4}"
export MEMORYWAM_PUTBACK_TEXT_CACHE="${MEMORYWAM_PUTBACK_TEXT_CACHE:-/mnt/vepfs02/output/kevin_wang/memorywam/data/fastwam_putback_text_cache_rgb_v3}"
export FASTWAM_ACTION_DIT_INIT="${FASTWAM_ACTION_DIT_INIT:-/mnt/vepfs02/output/kevin.wang/memorywam/model_assets/fastwam_fullkv_official_init/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt}"
export DYNAMIC_SURPRISE_MANIFEST="${manifest_root}"
export DYNAMIC_SURPRISE_BOUNDARY_STEP=-1
export CONTROL_INFORMATION_EVENT_RATE_OUTPUT="${output_dir}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PYTHONUTF8=1 LANG=C.UTF-8 LC_ALL=C.UTF-8
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

die() { echo "ERROR: $*" >&2; exit 2; }
[[ "${preflight_only}" == "0" || "${preflight_only}" == "1" ]] || die "PREFLIGHT_ONLY must be 0 or 1"
[[ -d "${repo_root}" && -x "${runtime_bin}/python" ]] || die "repository or runtime missing"
for required in \
  "${repo_root}/scripts/train.py" \
  "${repo_root}/tests/test_event_conditioned_rate.py" \
  "${repo_root}/configs/model/fastwam_event_conditioned_rate.yaml" \
  "${repo_root}/configs/task/rmbench_putback_control_information_event_rate_30k.yaml" \
  "${RMBENCH_PUTBACK_LEROBOT}/meta/info.json" \
  "${MEMORYWAM_PUTBACK_STATS}" \
  "${MEMORYWAM_PUTBACK_CONTINUOUS_LATENTS}/manifest.json" \
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
[[ -d "${MEMORYWAM_PUTBACK_TEXT_CACHE}" ]] || die "missing text cache"
IFS=',' read -r -a visible_gpus <<<"${CUDA_VISIBLE_DEVICES}"
[[ ${#visible_gpus[@]} -eq 8 ]] || die "formal run requires exactly eight GPUs"
[[ ! -e "${output_dir}" ]] || die "refusing to overwrite output: ${output_dir}"

cd "${repo_root}"
"${runtime_bin}/python" -m pytest -q tests/test_event_conditioned_rate.py
manifest_report="$("${runtime_bin}/python" - "${manifest_root}" "${runtime_lock}" <<'PY'
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import torch

from fastwam.memory.dynamic_surprise import DynamicSurpriseManifestStore
from fastwam.memory.event_conditioned_rate import memory_tokens_for_segment
from fastwam.memory.native_cache import build_dynamic_layerwise_training_layout

manifest_root = Path(sys.argv[1]).resolve()
runtime_lock = Path(sys.argv[2]).resolve()
store = DynamicSurpriseManifestStore(
    manifest_root,
    expected_boundary_step=-1,
    expected_episode_count=50,
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
metadata = json.loads((manifest_root / "manifest.json").read_text())["metadata"]
actual_runtime_sha = hashlib.sha256(runtime_lock.read_bytes()).hexdigest()
expected_runtime_sha = "c27e63084f3b579f1d04d6ca62126ca8caa8f6032ae94286cab196402c6cfb96"
assert metadata["complete"] is True
assert metadata["selector"] == "locked_segment_relative_control_information_v3"
assert metadata["runtime_lock_sha256"] == expected_runtime_sha
assert actual_runtime_sha == expected_runtime_sha
assert metadata["episode_count"] == 50
assert metadata["boundary_step"] == -1
assert metadata["retroactive_boundary_count"] == 0
assert metadata["detector_stride"] == 4
assert metadata["replan_stride"] == 16
assert metadata["memory_tokens_per_group"] == 8
assert lengths and min(lengths) >= 2 and max(lengths) <= 6

layout = build_dynamic_layerwise_training_layout(
    clean_frames=7,
    noisy_frames=1,
    tokens_per_frame=2,
    action_tokens=3,
    memory_groups=((0, 1), (2, 3, 4, 5)),
    memory_tokens=48,
    memory_token_counts=(16, 32),
    anchor_frames=2,
    recent_frames=4,
    device=torch.device("cpu"),
)
assert [
    segment.stop - segment.start
    for segment in layout.segments
    if segment.kind == "memory"
] == [16, 32]
assert layout.retained_clean_tokens == 60

segment_count = sum(lengths.values())
assert set(reasons) == {"segment_relative_control_information", "forced_maximum"}
average_tokens = event_rate_total / segment_count
reduction = 1.0 - event_rate_total / full_rate_total
print("implementation=event_conditioned_full_half_anchor2_recent4")
print("allocation_rule=information:K=8L,forced:K=8ceil(L/2)")
print("selector=" + metadata["selector"])
print("runtime_lock_sha256=" + actual_runtime_sha)
print("episodes=50")
print("manifest_segment_lengths=" + json.dumps(dict(sorted(lengths.items()))))
print("boundary_reasons=" + json.dumps(dict(sorted(reasons.items()))))
print("memory_token_histogram=" + json.dumps(dict(sorted(token_histogram.items()))))
print(f"average_memory_tokens={average_tokens:.6f}")
print(f"token_reduction_vs_k8l={reduction:.6%}")
print("maximum_memory_tokens=48")
PY
)"
printf '%s\n' "${manifest_report}"

resolved_config="$("${runtime_bin}/python" scripts/train.py task=rmbench_putback_control_information_event_rate_30k --cfg job)"
printf '%s\n' "${resolved_config}" | "${runtime_bin}/python" -c '
import sys
from omegaconf import OmegaConf
cfg = OmegaConf.create(sys.stdin.read())
expected = {
    "max_steps": 30000,
    "save_every": 10000,
    "save_steps": [5000],
    "save_training_state": False,
    "save_training_state_every": 10000,
    "save_final_checkpoint": True,
    "save_portable_fsdp_checkpoint": True,
    "keep_training_state_checkpoints": 1,
    "resume": None,
    "seed": 42,
    "batch_size": 1,
    "gradient_accumulation_steps": 1,
    "learning_rate": 2e-4,
    "mixed_precision": "bf16",
    "native_cache_train_mode": "full",
    "memory_allocation_rule": "event_full_forced_half",
    "data.train.minimum_history_frames": 1,
    "model.native_cache.mode": "layerwise",
    "model.native_cache.memory_tokens": 48,
    "model.native_cache.dynamic_tokens_per_frame": 8,
    "model.native_cache.anchor_frames": 2,
    "model.native_cache.recent_frames": 4,
    "model.native_cache.recursive": False,
}
for path, want in expected.items():
    got = OmegaConf.select(cfg, path)
    if got != want:
        raise SystemExit(f"config mismatch {path}: {got!r} != {want!r}")
'

echo "task=rmbench_putback_control_information_event_rate_30k"
echo "max_steps=30000"
echo "allocation_rule=information:K=8L,forced:K=8ceil(L/2) maximum_memory_tokens=48"
echo "anchor_frames=2 recent_frames=4"
echo "checkpoint_steps=5000,10000,20000,30000"
echo "save_training_state=false portable_weights_only=true"
echo "detector_stride=4 compressor_observation_stride=16"
echo "initialization=official_fullkv_actiondit"
echo "gpus=${CUDA_VISIBLE_DEVICES} output_dir=${output_dir}"
if [[ "${preflight_only}" == "1" ]]; then
  echo "preflight_status=ok"
  exit 0
fi

mkdir -p "${output_dir}"
mkdir -p "${output_dir}/source_snapshot"
for source_file in \
  "${repo_root}/src/fastwam/models/wan22/fastwam.py" \
  "${repo_root}/src/fastwam/memory/native_cache.py" \
  "${repo_root}/src/fastwam/datasets/lerobot/dynamic_surprise_dataset.py" \
  "${repo_root}/src/fastwam/memory/event_conditioned_rate.py" \
  "${repo_root}/configs/model/fastwam_event_conditioned_rate.yaml" \
  "${repo_root}/configs/task/rmbench_putback_control_information_event_rate_30k.yaml" \
  "${repo_root}/ops/train_putback_control_information_event_rate_30k.sh"; do
  cp "${source_file}" "${output_dir}/source_snapshot/$(basename "${source_file}")"
done
sha256sum "${output_dir}"/source_snapshot/* > "${output_dir}/source_snapshot/SHA256SUMS"
exec > >(tee -a "${output_dir}/train.log") 2>&1
echo "training_start=$(date --iso-8601=seconds)"
echo "source_snapshot=${output_dir}/source_snapshot/SHA256SUMS"
accelerate launch \
  --config_file scripts/accelerate_configs/accelerate_fsdp_full_shard.yaml \
  --num_processes 8 \
  scripts/train.py task=rmbench_putback_control_information_event_rate_30k
