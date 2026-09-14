#!/usr/bin/env bash
set -euo pipefail

repo_root=${REPO_ROOT:-/mnt/vepfs02/output/kevin_wang/memorywam/code/fastwam_multires_latent_kernel_k8}
runtime_bin=${RUNTIME_BIN:-/root/kevin_wang/envs/fastwam_guidemem/bin}
memorywam_root=${MEMORYWAM_ROOT:-/mnt/vepfs02/output/kevin_wang/memorywam}
asset_root=${OBSERVE_AND_PICKUP_ASSET_ROOT:-${memorywam_root}/data/memorywam_fullattention_observe_and_pickup_rgb_v3}
output_dir=${OUTPUT_DIR:-/mnt/vepfs01/output/kevin.wang/memorywam/train/latent_kernel_k8_observe_and_pickup_robotwin_joint16k_lr1e5_seed42_v1}
manifest_root=${LATENT_KERNEL_REGIME_MANIFEST:-/mnt/vepfs01/output/kevin.wang/memorywam/analysis/observe_and_pickup_latent_kernel_regime_k8_manifest_v1}
base_checkpoint=${FASTWAM_ROBOTWIN_PRETRAINED:-${memorywam_root}/model_assets/fastwam_release/robotwin_uncond_3cam_384.pt}
task_name=rmbench_observe_and_pickup_latent_kernel_regime_k8_robotwin_joint_16k_lr1e5
zero2_config=scripts/accelerate_configs/accelerate_zero2_ds.yaml
preflight_only=${PREFLIGHT_ONLY:-0}
require_idle_gpus=${REQUIRE_IDLE_GPUS:-1}

die() { echo "ERROR: $*" >&2; exit 2; }
[[ ${preflight_only} == 0 || ${preflight_only} == 1 ]] || die "PREFLIGHT_ONLY must be 0 or 1"
[[ ${require_idle_gpus} == 0 || ${require_idle_gpus} == 1 ]] || die "REQUIRE_IDLE_GPUS must be 0 or 1"
[[ -d ${repo_root} ]] || die "Missing isolated repository: ${repo_root}"
[[ -x ${runtime_bin}/python ]] || die "Missing Python runtime"
[[ -s ${repo_root}/${zero2_config} ]] || die "Missing ZeRO2 config"
[[ $(readlink -m "${output_dir}") == /mnt/vepfs01/output/* ]] || die "OUTPUT_DIR must be on vepfs01"

export PATH=${runtime_bin}:${PATH}
export PYTHONPATH=${repo_root}/src:${repo_root}${PYTHONPATH:+:${PYTHONPATH}}
export DIFFSYNTH_MODEL_BASE_PATH=${DIFFSYNTH_MODEL_BASE_PATH:-/mnt/vepfs02/datasets/kevin_wang/code/projects/FastWAM/checkpoints}
export DIFFSYNTH_SKIP_DOWNLOAD=true
export RMBENCH_COVER_BLOCKS_LEROBOT=${asset_root}/lerobot/observe_and_pickup
export MEMORYWAM_COVER_BLOCKS_STATS=${asset_root}/stats/dataset_stats.json
export MEMORYWAM_COVER_BLOCKS_CONTINUOUS_LATENTS=${asset_root}/temporal_fullkv_continuous_episode_stride16_v4
export MEMORYWAM_COVER_BLOCKS_TEXT_CACHE=${asset_root}/text_cache
export FASTWAM_ROBOTWIN_PRETRAINED=${base_checkpoint}
export LATENT_KERNEL_REGIME_MANIFEST=${manifest_root}
export OBSERVE_LATENT_KERNEL_OUTPUT=${output_dir}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export PYTHONUTF8=1 LANG=C.UTF-8 LC_ALL=C.UTF-8 PYTHONDONTWRITEBYTECODE=1

for required in \
  "${base_checkpoint}" \
  "${RMBENCH_COVER_BLOCKS_LEROBOT}/meta/info.json" \
  "${MEMORYWAM_COVER_BLOCKS_STATS}" \
  "${MEMORYWAM_COVER_BLOCKS_CONTINUOUS_LATENTS}/manifest.json"; do
  [[ -s ${required} ]] || die "Missing or empty artifact: ${required}"
done
[[ -d ${MEMORYWAM_COVER_BLOCKS_TEXT_CACHE} ]] || die "Missing text cache"

expected_size=${FASTWAM_ROBOTWIN_PRETRAINED_SIZE:-12041813092}
expected_sha256=${FASTWAM_ROBOTWIN_PRETRAINED_SHA256:-776475b22566a791854ecf31cf3b50f25e7d8d94c343132ec16eb94994aa9e63}
[[ $(stat -c '%s' "${base_checkpoint}") == ${expected_size} ]] || die "RobotWin checkpoint size mismatch"
[[ $(sha256sum "${base_checkpoint}" | awk '{print $1}') == ${expected_sha256} ]] || die "RobotWin checkpoint SHA256 mismatch"

cd "${repo_root}"
if [[ ! -e ${manifest_root} ]]; then
  mkdir -p "$(dirname "${manifest_root}")"
  "${runtime_bin}/python" scripts/build_latent_kernel_regime_manifest.py \
    --latent-root "${MEMORYWAM_COVER_BLOCKS_CONTINUOUS_LATENTS}" \
    --output "${manifest_root}" \
    --task observe_and_pickup
fi

manifest_summary=$("${runtime_bin}/python" - "${manifest_root}" <<'PY'
import collections, json, sys
from fastwam.memory.latent_kernel_regime import LatentKernelManifestStore
store = LatentKernelManifestStore(sys.argv[1], expected_episode_count=50, expected_task="observe_and_pickup")
lengths = collections.Counter(row.length for ep in range(50) for row in store.segments_for_episode(ep))
if len(lengths) < 2:
    raise SystemExit(f"latent selector degenerated to fixed length: {dict(lengths)}")
print(json.dumps({"lengths": dict(sorted(lengths.items())), "metadata": store.metadata}, sort_keys=True))
PY
)

resolved_config=$("${runtime_bin}/python" scripts/train.py "task=${task_name}" --cfg job --resolve)
RESOLVED_CONFIG=${resolved_config} OUTPUT_EXPECTED=${output_dir} BASE_EXPECTED=${base_checkpoint} MANIFEST_EXPECTED=${manifest_root} \
"${runtime_bin}/python" - <<'PY' || die "Resolved one-stage contract mismatch"
import os
from omegaconf import OmegaConf
cfg = OmegaConf.create(os.environ["RESOLVED_CONFIG"])
expected = {
    "output_dir": os.environ["OUTPUT_EXPECTED"], "initial_weights": os.environ["BASE_EXPECTED"],
    "resume": None, "learning_rate": 1e-5, "lr_scheduler_type": "constant",
    "max_steps": 16000, "save_every": 2000, "batch_size": 1, "num_gpus": 8,
    "gradient_accumulation_steps": 1, "weight_decay": 1e-2, "max_grad_norm": 1.0,
    "mixed_precision": "bf16", "seed": 42, "native_cache_train_mode": "full",
    "data.train._target_": "fastwam.datasets.lerobot.latent_kernel_regime_dataset.LatentKernelRegimeRobotVideoDataset",
    "data.train.latent_kernel_manifest_path": os.environ["MANIFEST_EXPECTED"],
    "data.train.manifest_task": "observe_and_pickup", "data.train.minimum_history_frames": 6,
    "model.native_cache.memory_tokens": 8, "model.native_cache.anchor_frames": 2,
    "model.native_cache.recent_frames": 4, "model.native_cache.recursive": False,
    "model.loss.lambda_video": 1.0, "model.loss.lambda_action": 1.0,
}
for path, want in expected.items():
    got = OmegaConf.select(cfg, path)
    if got != want: raise SystemExit(f"{path}: expected {want!r}, got {got!r}")
PY

echo "task=observe_and_pickup method=latent_kernel_L4to8_K8"
echo "one_stage_joint=true initial_weights=${base_checkpoint} lr=1e-5 steps=16000 min_history=6"
echo "checkpoint_steps=2000,4000,6000,8000,10000,12000,14000,16000"
echo "manifest=${manifest_root} summary=${manifest_summary}"
echo "output_dir=${output_dir}"
[[ ${preflight_only} == 0 ]] || { echo "preflight_status=ok"; exit 0; }
[[ ! -e ${output_dir} ]] || die "Refusing to overwrite output: ${output_dir}"
IFS=',' read -r -a gpus <<<"${CUDA_VISIBLE_DEVICES}"
[[ ${#gpus[@]} -eq 8 ]] || die "Formal training requires exactly 8 visible GPUs"
if [[ ${require_idle_gpus} == 1 ]]; then
  active=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d' | sort -u)
  [[ -z ${active} ]] || die "GPU compute processes are active"
fi
mkdir -p "${output_dir}/contracts"
printf '%s\n' "${resolved_config}" >"${output_dir}/contracts/train_resolved.yaml"
printf '%s\n' "${manifest_summary}" >"${output_dir}/contracts/manifest_summary.json"
exec > >(tee -a "${output_dir}/train.log") 2>&1
accelerate launch --config_file "${zero2_config}" --num_processes 8 scripts/train.py "task=${task_name}"

