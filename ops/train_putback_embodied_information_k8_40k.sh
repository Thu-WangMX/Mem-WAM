#!/usr/bin/env bash
set -euo pipefail

repo_root="${REPO_ROOT:-/mnt/vepfs02/output/kevin_wang/memorywam/code/fastwam_dynamic_multiframe_selection}"
runtime_bin="${RUNTIME_BIN:-/root/kevin_wang/envs/fastwam_guidemem/bin}"
output_dir="${OUTPUT_DIR:-/mnt/vepfs02/output/kevin_wang/memorywam/train/embodied_information_k8_putback_e2e_40k_seed42}"
manifest_root="${MANIFEST_ROOT:-/mnt/vepfs02/output/kevin_wang/memorywam/data/putback_embodied_information_planning_segments_v1}"
preflight_only="${PREFLIGHT_ONLY:-0}"

export PATH="${runtime_bin}:${PATH}"
export PYTHONPATH="${repo_root}/src:${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-/mnt/vepfs02/datasets/kevin_wang/code/projects/FastWAM/checkpoints}"
export DIFFSYNTH_SKIP_DOWNLOAD=true
export RMBENCH_PUTBACK_LEROBOT="${RMBENCH_PUTBACK_LEROBOT:-/mnt/vepfs02/output/kevin_wang/memorywam/data/rmbench_lerobot_v21_rgb_v3/put_back_block}"
export MEMORYWAM_PUTBACK_STATS="${MEMORYWAM_PUTBACK_STATS:-/mnt/vepfs02/output/kevin_wang/memorywam/data/fastwam_putback_stats_rgb_v3/dataset_stats.json}"
export MEMORYWAM_PUTBACK_CONTINUOUS_LATENTS="${MEMORYWAM_PUTBACK_CONTINUOUS_LATENTS:-/mnt/vepfs02/output/kevin_wang/memorywam/data/fastwam_putback_continuous_episode_stride16_v4}"
export MEMORYWAM_PUTBACK_TEXT_CACHE="${MEMORYWAM_PUTBACK_TEXT_CACHE:-/mnt/vepfs02/output/kevin_wang/memorywam/data/fastwam_putback_text_cache_rgb_v3}"
export FASTWAM_ACTION_DIT_INIT="${FASTWAM_ACTION_DIT_INIT:-/mnt/vepfs02/output/kevin_wang/memorywam/model_assets/fastwam_fullkv_official_init/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt}"
export DYNAMIC_SURPRISE_MANIFEST="${manifest_root}"
export DYNAMIC_SURPRISE_BOUNDARY_STEP=-1
export EMBODIED_INFORMATION_OUTPUT="${output_dir}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PYTHONUTF8=1 LANG=C.UTF-8 LC_ALL=C.UTF-8
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"

die() { echo "ERROR: $*" >&2; exit 2; }
[[ "${preflight_only}" == "0" || "${preflight_only}" == "1" ]] || die "PREFLIGHT_ONLY must be 0 or 1"
[[ -d "${repo_root}" && -x "${runtime_bin}/python" ]] || die "repository or runtime missing"
for required in \
  "${RMBENCH_PUTBACK_LEROBOT}/meta/info.json" \
  "${MEMORYWAM_PUTBACK_STATS}" \
  "${MEMORYWAM_PUTBACK_CONTINUOUS_LATENTS}/manifest.json" \
  "${FASTWAM_ACTION_DIT_INIT}" \
  "${manifest_root}/manifest.json"; do
  [[ -s "${required}" ]] || die "missing required artifact: ${required}"
done
[[ -d "${MEMORYWAM_PUTBACK_TEXT_CACHE}" ]] || die "missing text cache"
IFS=',' read -r -a visible_gpus <<<"${CUDA_VISIBLE_DEVICES}"
[[ ${#visible_gpus[@]} -eq 8 ]] || die "formal run requires exactly eight GPUs"
[[ ! -e "${output_dir}" ]] || die "refusing to overwrite output: ${output_dir}"

cd "${repo_root}"
"${runtime_bin}/python" - "${manifest_root}" <<'PY'
import json, sys
from collections import Counter
from fastwam.memory.dynamic_surprise import DynamicSurpriseManifestStore
store = DynamicSurpriseManifestStore(sys.argv[1], expected_boundary_step=-1,
                                     expected_episode_count=50)
lengths = Counter()
for episode in range(50):
    segments = store.segments_for_episode(episode)
    lengths.update(right-left for left, right in segments)
meta = json.load(open(sys.argv[1] + "/manifest.json"))["metadata"]
assert meta["selector"] == "locked_embodied_information_v1"
assert meta["retroactive_boundary_count"] == 0
assert len(lengths) >= 2 and max(lengths) <= 8
print("manifest_segment_lengths=" + json.dumps(dict(sorted(lengths.items()))))
PY

resolved_config="$("${runtime_bin}/python" scripts/train.py task=rmbench_putback_embodied_information_k8_40k --cfg job)"
printf '%s\n' "${resolved_config}" | "${runtime_bin}/python" -c '
import sys
from omegaconf import OmegaConf
cfg=OmegaConf.create(sys.stdin.read())
expected={"max_steps":40000,"save_every":5000,"save_steps":[],"seed":42,
"batch_size":1,"gradient_accumulation_steps":1,"learning_rate":2e-4,
"mixed_precision":"bf16","native_cache_train_mode":"full",
"data.train.minimum_history_frames":1,"model.native_cache.mode":"layerwise",
"model.native_cache.memory_tokens":8,"model.native_cache.group_size":4,
"model.native_cache.anchor_frames":2,"model.native_cache.recent_frames":4,
"model.native_cache.recursive":False}
for path,want in expected.items():
    got=OmegaConf.select(cfg,path)
    if got != want: raise SystemExit(f"config mismatch {path}: {got!r} != {want!r}")
'

echo "task=rmbench_putback_embodied_information_k8_40k"
echo "max_steps=40000 checkpoint_steps=5000,10000,15000,20000,25000,30000,35000,40000"
echo "detector_stride=4 compressor_observation_stride=16 memory_tokens_per_group=8"
echo "gpus=${CUDA_VISIBLE_DEVICES} output_dir=${output_dir}"
if [[ "${preflight_only}" == "1" ]]; then echo "preflight_status=ok"; exit 0; fi

mkdir -p "${output_dir}"
exec > >(tee -a "${output_dir}/train.log") 2>&1
echo "training_start=$(date --iso-8601=seconds)"
accelerate launch \
  --config_file scripts/accelerate_configs/accelerate_fsdp_full_shard.yaml \
  --num_processes 8 \
  scripts/train.py task=rmbench_putback_embodied_information_k8_40k
