#!/usr/bin/env bash
set -euo pipefail

repo_root=${REPO_ROOT:-/mnt/vepfs02/output/kevin_wang/memorywam/code/fastwam_multimodal_rate_controlled_k8}
runtime_bin=${RUNTIME_BIN:-/root/kevin_wang/envs/fastwam_guidemem/bin}
memorywam_root=${MEMORYWAM_ROOT:-/mnt/vepfs02/output/kevin_wang/memorywam}
asset_root=${SWAPT_ASSET_ROOT:-${memorywam_root}/data/fastwam_fullkv_remaining8_rgb_v3/swap_T}
output_dir=${OUTPUT_DIR:-/mnt/vepfs01/output/kevin.wang/memorywam/train/fixed_l7_k8_swapt_scratch_joint16k_lr2e4_seed42_v1}
task_name=rmbench_swapt_fixed_l7_k8_scratch_joint_16k_lr2e4
num_processes=${NUM_PROCESSES:-8}
[[ ${num_processes} == 4 || ${num_processes} == 8 ]] || { echo "ERROR: NUM_PROCESSES must be 4 or 8" >&2; exit 2; }
gradient_accumulation_steps=$((8 / num_processes))
preflight_only=${PREFLIGHT_ONLY:-0}
require_idle_gpus=${REQUIRE_IDLE_GPUS:-1}
die(){ echo "ERROR: $*" >&2; exit 2; }

[[ -d ${repo_root} && -x ${runtime_bin}/python ]] || die "Missing repository or runtime"
[[ $(readlink -m "${output_dir}") == /mnt/vepfs01/output/* ]] || die "OUTPUT_DIR must be on vepfs01"
export PATH=${runtime_bin}:${PATH}
export PYTHONPATH=${repo_root}/src:${repo_root}${PYTHONPATH:+:${PYTHONPATH}}
export DIFFSYNTH_MODEL_BASE_PATH=${DIFFSYNTH_MODEL_BASE_PATH:-/mnt/vepfs02/datasets/kevin_wang/code/projects/FastWAM/checkpoints}
export DIFFSYNTH_SKIP_DOWNLOAD=true
export RMBENCH_COVER_BLOCKS_LEROBOT=${asset_root}/lerobot/swap_T
export MEMORYWAM_COVER_BLOCKS_STATS=${asset_root}/stats/dataset_stats.json
export MEMORYWAM_COVER_BLOCKS_CONTINUOUS_LATENTS=${asset_root}/temporal_fullkv_continuous_episode_stride16_v4
export MEMORYWAM_COVER_BLOCKS_TEXT_CACHE=${asset_root}/text_cache
export FASTWAM_ACTION_DIT_INIT=${FASTWAM_ACTION_DIT_INIT:-${memorywam_root}/model_assets/fastwam_fullkv_official_init/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt}
export SWAPT_FIXED_L7_K8_OUTPUT=${output_dir}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export PYTHONUTF8=1 LANG=C.UTF-8 LC_ALL=C.UTF-8 PYTHONDONTWRITEBYTECODE=1
for f in "${FASTWAM_ACTION_DIT_INIT}" "${RMBENCH_COVER_BLOCKS_LEROBOT}/meta/info.json" "${MEMORYWAM_COVER_BLOCKS_STATS}" "${MEMORYWAM_COVER_BLOCKS_CONTINUOUS_LATENTS}/manifest.json"; do [[ -s ${f} ]] || die "Missing ${f}"; done
[[ -d ${MEMORYWAM_COVER_BLOCKS_TEXT_CACHE} ]] || die "Missing text cache"

cd "${repo_root}"
resolved=$("${runtime_bin}/python" scripts/train.py "task=${task_name}" "num_gpus=${num_processes}" "gradient_accumulation_steps=${gradient_accumulation_steps}" --cfg job --resolve)
RESOLVED=${resolved} OUTPUT=${output_dir} NUM_PROCESSES=${num_processes} GRAD_ACCUM=${gradient_accumulation_steps} "${runtime_bin}/python" - <<'PY' || die "Resolved contract mismatch"
import os
from omegaconf import OmegaConf
c=OmegaConf.create(os.environ["RESOLVED"])
e={
    "output_dir":os.environ["OUTPUT"],
    "initial_weights":None,
    "resume":None,
    "learning_rate":2e-4,
    "max_steps":16000,
    "save_every":2000,
    "batch_size":1,
    "num_gpus":int(os.environ["NUM_PROCESSES"]),
    "gradient_accumulation_steps":int(os.environ["GRAD_ACCUM"]),
    "data.train.minimum_history_frames":1,
    "data.train.fixed_segment_length":7,
    "data.train.memory_tokens":8,
    "data.train.anchor_frames":2,
    "data.train.confirmation_delay":1,
    "native_cache_train_mode":"full",
    "model.loss.lambda_video":1.0,
    "model.loss.lambda_action":1.0,
}
for k,v in e.items():
    got=OmegaConf.select(c,k)
    if got!=v: raise SystemExit(f"{k}: {got!r} != {v!r}")
if "robotwin_uncond_3cam_384.pt" in os.environ["RESOLVED"].lower():
    raise SystemExit("RobotWin checkpoint leaked into scratch config")
PY
echo "task=swap_T method=fixed_l7_k8 matched_control=true scratch=true joint=true lr=2e-4 steps=16000 save_every=2000 processes=${num_processes} grad_accum=${gradient_accumulation_steps} global_batch=8 output=${output_dir}"
[[ ${preflight_only} == 0 ]] || { echo preflight_status=ok; exit 0; }
[[ ! -e ${output_dir} ]] || die "Refusing to overwrite ${output_dir}"
IFS=',' read -r -a gpus <<<"${CUDA_VISIBLE_DEVICES}"; [[ ${#gpus[@]} -eq ${num_processes} ]] || die "Visible GPU count must equal NUM_PROCESSES"
if [[ ${require_idle_gpus} == 1 ]]; then active=$(nvidia-smi -i "${CUDA_VISIBLE_DEVICES}" --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d'|sort -u); [[ -z ${active} ]] || die "Selected GPUs are busy"; fi
mkdir -p "${output_dir}/contracts"; printf '%s\n' "${resolved}" >"${output_dir}/contracts/train_resolved.yaml"
exec > >(tee -a "${output_dir}/train.log") 2>&1
accelerate launch --config_file scripts/accelerate_configs/accelerate_zero2_ds.yaml --num_processes "${num_processes}" scripts/train.py "task=${task_name}" "num_gpus=${num_processes}" "gradient_accumulation_steps=${gradient_accumulation_steps}"
