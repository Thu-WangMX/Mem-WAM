#!/usr/bin/env bash
set -euo pipefail

repo_root="${REPO_ROOT:-/mnt/vepfs02/output/kevin_wang/memorywam/code/fastwam_physical_dynamic_l_k8}"
runtime_bin="${RUNTIME_BIN:-/root/kevin_wang/envs/fastwam_guidemem/bin}"
output_dir="${OUTPUT_DIR:-/mnt/vepfs01/output/kevin.wang/memorywam/train/physical_dynamic_l_k8_putback_init_joint20k_lr2e4_seed42_v2}"
manifest_root="${PHYSICAL_DYNAMIC_L_MANIFEST:-/mnt/vepfs01/output/kevin.wang/memorywam/analysis/putback_physical_dynamic_l_k8_manifest_v1}"
task_name=rmbench_putback_physical_dynamic_l_k8_init_20k
zero2_config=scripts/accelerate_configs/accelerate_zero2_ds.yaml
preflight_only="${PREFLIGHT_ONLY:-0}"
require_idle_gpus="${REQUIRE_IDLE_GPUS:-1}"

die() { echo "ERROR: $*" >&2; exit 2; }

[[ "${preflight_only}" == 0 || "${preflight_only}" == 1 ]] || die "PREFLIGHT_ONLY must be 0 or 1"
[[ "${require_idle_gpus}" == 0 || "${require_idle_gpus}" == 1 ]] || die "REQUIRE_IDLE_GPUS must be 0 or 1"
[[ -d "${repo_root}" ]] || die "Missing isolated repository: ${repo_root}"
[[ -x "${runtime_bin}/python" ]] || die "Missing Python runtime: ${runtime_bin}/python"
[[ -s "${repo_root}/${zero2_config}" ]] || die "Missing ZeRO2 config: ${repo_root}/${zero2_config}"
[[ "$(readlink -m "${output_dir}")" == /mnt/vepfs01/output/* ]] || die "OUTPUT_DIR must be on vepfs01"
[[ "$(readlink -m "${manifest_root}")" == /mnt/vepfs01/output/* ]] || die "manifest must be on vepfs01"

export REPO_ROOT="${repo_root}" RUNTIME_BIN="${runtime_bin}"
export PATH="${runtime_bin}:${PATH}"
export PYTHONPATH="${repo_root}/src:${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-/mnt/vepfs02/datasets/kevin_wang/code/projects/FastWAM/checkpoints}"
export DIFFSYNTH_SKIP_DOWNLOAD=true
export RMBENCH_PUTBACK_LEROBOT="${RMBENCH_PUTBACK_LEROBOT:-/mnt/vepfs02/output/kevin_wang/memorywam/data/rmbench_lerobot_v21_rgb_v3/put_back_block}"
export MEMORYWAM_PUTBACK_STATS="${MEMORYWAM_PUTBACK_STATS:-/mnt/vepfs02/output/kevin_wang/memorywam/data/fastwam_putback_stats_rgb_v3/dataset_stats.json}"
export MEMORYWAM_PUTBACK_CONTINUOUS_LATENTS="${MEMORYWAM_PUTBACK_CONTINUOUS_LATENTS:-/mnt/vepfs02/output/kevin_wang/memorywam/data/fastwam_putback_continuous_episode_stride16_v4}"
export MEMORYWAM_PUTBACK_TEXT_CACHE="${MEMORYWAM_PUTBACK_TEXT_CACHE:-/mnt/vepfs02/output/kevin_wang/memorywam/data/fastwam_putback_text_cache_rgb_v3}"
export FASTWAM_ACTION_DIT_INIT="${FASTWAM_ACTION_DIT_INIT:-/mnt/vepfs02/output/kevin_wang/memorywam/model_assets/fastwam_fullkv_official_init/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt}"
export PHYSICAL_DYNAMIC_L_MANIFEST="${manifest_root}"
export PUTBACK_PHYSICAL_DYNAMIC_L_OUTPUT="${output_dir}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export PYTHONUTF8=1 LANG=C.UTF-8 LC_ALL=C.UTF-8 PYTHONDONTWRITEBYTECODE=1

for required in \
  "${FASTWAM_ACTION_DIT_INIT}" \
  "${RMBENCH_PUTBACK_LEROBOT}/meta/info.json" \
  "${MEMORYWAM_PUTBACK_STATS}" \
  "${MEMORYWAM_PUTBACK_CONTINUOUS_LATENTS}/manifest.json"; do
  [[ -s "${required}" ]] || die "Missing or empty required artifact: ${required}"
done
[[ -d "${MEMORYWAM_PUTBACK_TEXT_CACHE}" ]] || die "Missing text cache"

cd "${repo_root}"
if [[ ! -e "${manifest_root}" ]]; then
  mkdir -p "$(dirname "${manifest_root}")"
  "${runtime_bin}/python" scripts/build_putback_physical_dynamic_l_manifest.py \
    --lerobot-root "${RMBENCH_PUTBACK_LEROBOT}" \
    --output "${manifest_root}"
fi

manifest_summary="$(${runtime_bin}/python - "${manifest_root}" <<'PY'
import collections
import json
import sys
from fastwam.memory.physical_dynamic_l import PhysicalDynamicLManifestStore

store = PhysicalDynamicLManifestStore(sys.argv[1], expected_episode_count=50)
lengths = collections.Counter()
reasons = collections.Counter()
for episode in range(50):
    records = store.segments_for_episode(episode)
    lengths.update(row.length for row in records)
    reasons.update(row.reason for row in records)
if len(lengths) < 2:
    raise SystemExit(f"physical selector degenerated to fixed length: {dict(lengths)}")
print(json.dumps({"lengths": dict(sorted(lengths.items())), "reasons": dict(sorted(reasons.items()))}))
PY
)"

resolved_config="$(${runtime_bin}/python scripts/train.py "task=${task_name}" --cfg job --resolve)"
RESOLVED_CONFIG="${resolved_config}" OUTPUT_EXPECTED="${output_dir}" \
  ACTION_INIT_EXPECTED="${FASTWAM_ACTION_DIT_INIT}" MANIFEST_EXPECTED="${manifest_root}" \
  "${runtime_bin}/python" - <<'PY' || die "Resolved initialization-training contract mismatch"
import os
from omegaconf import OmegaConf

cfg = OmegaConf.create(os.environ["RESOLVED_CONFIG"])
expected = {
    "output_dir": os.environ["OUTPUT_EXPECTED"],
    "initial_weights": None,
    "resume": None,
    "learning_rate": 2e-4,
    "max_steps": 20000,
    "save_every": 0,
    "save_steps": [5000, 10000, 15000, 16000, 17000, 18000, 19000, 20000],
    "save_training_state": True,
    "save_training_state_every": 10000,
    "batch_size": 1,
    "num_gpus": 8,
    "gradient_accumulation_steps": 1,
    "mixed_precision": "bf16",
    "seed": 42,
    "native_cache_train_mode": "full",
    "data.train._target_": "fastwam.datasets.lerobot.physical_dynamic_l_dataset.PhysicalDynamicLRobotVideoDataset",
    "data.train.physical_manifest_path": os.environ["MANIFEST_EXPECTED"],
    "data.train.memory_tokens": 8,
    "data.train.minimum_history_frames": 1,
    "model.native_cache.mode": "layerwise",
    "model.native_cache.memory_tokens": 8,
    "model.native_cache.anchor_frames": 2,
    "model.native_cache.recent_frames": 4,
    "model.native_cache.recursive": False,
    "model.loss.lambda_video": 1.0,
    "model.loss.lambda_action": 1.0,
    "model.action_dit_pretrained_path": os.environ["ACTION_INIT_EXPECTED"],
    "model.skip_dit_load_from_pretrain": False,
}
for path, want in expected.items():
    got = OmegaConf.select(cfg, path)
    if got != want:
        raise SystemExit(f"{path}: expected {want!r}, got {got!r}")
text = os.environ["RESOLVED_CONFIG"].lower()
if "robotwin_uncond_3cam_384.pt" in text:
    raise SystemExit("RobotWin pretrained checkpoint leaked into resolved config")
PY

echo "task=put_back_block selector=train_free_physical_dynamic_L4to8 memory=fixed_K8"
echo "initialization=generic_wan22_plus_official_actiondit robotwin_pretrained=false"
echo "initial_weights=null resume=null joint=reader_video_action_and_compressor"
echo "steps=0_to_20000 lr=2e-4 batch_per_gpu=1 gpus=8 backend=deepspeed_zero2"
echo "anchor=2 recent=4 min_L=4 nominal_L=6 max_L=8"
echo "checkpoint_steps=5000,10000,15000,16000,17000,18000,19000,20000"
echo "training_state_steps=10000,20000 keep_training_state_checkpoints=1"
echo "physical_manifest=${manifest_root} summary=${manifest_summary}"
echo "output_dir=${output_dir}"

if [[ "${preflight_only}" == 1 ]]; then
  echo "preflight=ok"
  exit 0
fi
[[ ! -e "${output_dir}" ]] || die "Refusing to overwrite existing output: ${output_dir}"
IFS=',' read -r -a visible_gpus <<<"${CUDA_VISIBLE_DEVICES}"
[[ ${#visible_gpus[@]} -eq 8 ]] || die "Formal training requires exactly eight visible GPUs"
if [[ "${require_idle_gpus}" == 1 ]]; then
  active_compute_pids="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d' | sort -u)"
  [[ -z "${active_compute_pids}" ]] || die "GPU compute processes are active"
fi

mkdir -p "${output_dir}/contracts"
printf '%s\n' "${resolved_config}" >"${output_dir}/contracts/train_20k_resolved.yaml"
printf '%s\n' "${manifest_summary}" >"${output_dir}/contracts/physical_manifest_summary.json"
exec > >(tee -a "${output_dir}/train.log") 2>&1
accelerate launch \
  --config_file "${zero2_config}" \
  --num_processes 8 \
  scripts/train.py "task=${task_name}"

for step in 5000 10000 15000 16000 17000 18000 19000 20000; do
  checkpoint="${output_dir}/checkpoints/weights/step_$(printf '%06d' "${step}").pt"
  [[ -s "${checkpoint}" ]] || die "Missing expected checkpoint: ${checkpoint}"
done
echo "complete output_dir=${output_dir}"
