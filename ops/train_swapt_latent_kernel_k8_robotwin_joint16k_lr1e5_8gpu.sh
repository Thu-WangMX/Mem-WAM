#!/usr/bin/env bash
set -euo pipefail
repo_root=${REPO_ROOT:-/mnt/vepfs02/output/kevin_wang/memorywam/code/fastwam_multires_latent_kernel_k8}
runtime_bin=${RUNTIME_BIN:-/root/kevin_wang/envs/fastwam_guidemem/bin}
memorywam_root=${MEMORYWAM_ROOT:-/mnt/vepfs02/output/kevin_wang/memorywam}
asset_root=${SWAPT_ASSET_ROOT:-${memorywam_root}/data/fastwam_fullkv_remaining8_rgb_v3/swap_T}
output_dir=${OUTPUT_DIR:-/mnt/vepfs01/output/kevin.wang/memorywam/train/latent_kernel_k8_swapt_robotwin_joint16k_lr1e5_seed42_v1}
manifest_root=${LATENT_KERNEL_REGIME_MANIFEST:-${memorywam_root}/analysis/trainfree_multires_latent_kernel_v1/swapT}
base_checkpoint=${FASTWAM_ROBOTWIN_PRETRAINED:-${memorywam_root}/model_assets/fastwam_release/robotwin_uncond_3cam_384.pt}
task_name=rmbench_swapt_latent_kernel_regime_k8_robotwin_joint_16k_lr1e5
preflight_only=${PREFLIGHT_ONLY:-0}
require_idle_gpus=${REQUIRE_IDLE_GPUS:-1}
die(){ echo "ERROR: $*" >&2; exit 2; }

[[ -d ${repo_root} && -x ${runtime_bin}/python ]] || die "Missing repository or runtime"
[[ -s ${base_checkpoint} && -s ${manifest_root}/manifest.json ]] || die "Missing RobotWin checkpoint or latent manifest"
[[ $(readlink -m "${output_dir}") == /mnt/vepfs01/output/* ]] || die "OUTPUT_DIR must be on vepfs01"
export PATH=${runtime_bin}:${PATH}
export PYTHONPATH=${repo_root}/src:${repo_root}${PYTHONPATH:+:${PYTHONPATH}}
export DIFFSYNTH_MODEL_BASE_PATH=${DIFFSYNTH_MODEL_BASE_PATH:-/mnt/vepfs02/datasets/kevin_wang/code/projects/FastWAM/checkpoints}
export DIFFSYNTH_SKIP_DOWNLOAD=true
export RMBENCH_COVER_BLOCKS_LEROBOT=${asset_root}/lerobot/swap_T
export MEMORYWAM_COVER_BLOCKS_STATS=${asset_root}/stats/dataset_stats.json
export MEMORYWAM_COVER_BLOCKS_CONTINUOUS_LATENTS=${asset_root}/temporal_fullkv_continuous_episode_stride16_v4
export MEMORYWAM_COVER_BLOCKS_TEXT_CACHE=${asset_root}/text_cache
export FASTWAM_ROBOTWIN_PRETRAINED=${base_checkpoint}
export FASTWAM_ACTION_DIT_INIT=${FASTWAM_ACTION_DIT_INIT:-${memorywam_root}/model_assets/fastwam_fullkv_official_init/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt}
export LATENT_KERNEL_REGIME_MANIFEST=${manifest_root}
export SWAPT_LATENT_KERNEL_OUTPUT=${output_dir}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export PYTHONUTF8=1 LANG=C.UTF-8 LC_ALL=C.UTF-8 PYTHONDONTWRITEBYTECODE=1
for f in "${FASTWAM_ACTION_DIT_INIT}" "${RMBENCH_COVER_BLOCKS_LEROBOT}/meta/info.json" "${MEMORYWAM_COVER_BLOCKS_STATS}" "${MEMORYWAM_COVER_BLOCKS_CONTINUOUS_LATENTS}/manifest.json"; do [[ -s ${f} ]] || die "Missing ${f}"; done
[[ -d ${MEMORYWAM_COVER_BLOCKS_TEXT_CACHE} ]] || die "Missing text cache"
[[ $(stat -c '%s' "${base_checkpoint}") == ${FASTWAM_ROBOTWIN_PRETRAINED_SIZE:-12041813092} ]] || die "RobotWin size mismatch"
[[ $(sha256sum "${base_checkpoint}" | awk '{print $1}') == ${FASTWAM_ROBOTWIN_PRETRAINED_SHA256:-776475b22566a791854ecf31cf3b50f25e7d8d94c343132ec16eb94994aa9e63} ]] || die "RobotWin SHA mismatch"
cd "${repo_root}"
summary=$("${runtime_bin}/python" - "${manifest_root}" <<'PY'
import collections,json,sys
from fastwam.memory.latent_kernel_regime import LatentKernelManifestStore
s=LatentKernelManifestStore(sys.argv[1],expected_episode_count=50,expected_task="swap_T")
h=collections.Counter(x.length for e in range(50) for x in s.segments_for_episode(e))
assert len(h)>=2 and sum(h.values())>0, h
print(json.dumps(dict(sorted(h.items()))))
PY
)
resolved=$("${runtime_bin}/python" scripts/train.py "task=${task_name}" --cfg job --resolve)
RESOLVED=${resolved} OUTPUT=${output_dir} BASE=${base_checkpoint} MANIFEST=${manifest_root} "${runtime_bin}/python" - <<'PY' || die "Resolved contract mismatch"
import os
from omegaconf import OmegaConf
c=OmegaConf.create(os.environ["RESOLVED"])
e={"output_dir":os.environ["OUTPUT"],"initial_weights":os.environ["BASE"],"resume":None,"learning_rate":1e-5,"max_steps":16000,"save_every":2000,"batch_size":1,"num_gpus":8,"data.train.minimum_history_frames":6,"data.train.latent_kernel_manifest_path":os.environ["MANIFEST"],"data.train.manifest_task":"swap_T","native_cache_train_mode":"full","model.loss.lambda_video":1.0,"model.loss.lambda_action":1.0}
for k,v in e.items():
 g=OmegaConf.select(c,k)
 if g!=v: raise SystemExit(f"{k}: {g!r} != {v!r}")
PY
echo "task=swap_T method=latent_kernel one_stage_joint=true lr=1e-5 steps=16000 save_every=2000 manifest_lengths=${summary} output=${output_dir}"
[[ ${preflight_only} == 0 ]] || { echo preflight_status=ok; exit 0; }
[[ ! -e ${output_dir} ]] || die "Refusing to overwrite ${output_dir}"
IFS=',' read -r -a gpus <<<"${CUDA_VISIBLE_DEVICES}"; [[ ${#gpus[@]} -eq 8 ]] || die "Need 8 GPUs"
if [[ ${require_idle_gpus} == 1 ]]; then active=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d'|sort -u); [[ -z ${active} ]] || die "GPUs are busy"; fi
mkdir -p "${output_dir}/contracts"; printf '%s\n' "${resolved}" >"${output_dir}/contracts/train_resolved.yaml"; printf '%s\n' "${summary}" >"${output_dir}/contracts/manifest_summary.json"
exec > >(tee -a "${output_dir}/train.log") 2>&1
accelerate launch --config_file scripts/accelerate_configs/accelerate_zero2_ds.yaml --num_processes 8 scripts/train.py "task=${task_name}"
