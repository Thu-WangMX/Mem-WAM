#!/usr/bin/env bash
set -euo pipefail

experiment=${EXPERIMENT:?Set EXPERIMENT to a supported paired experiment}
repo_root=${REPO_ROOT:-/mnt/vepfs02/output/kevin_wang/memorywam/code/fastwam_multimodal_rate_controlled_k8}
runtime_bin=${RUNTIME_BIN:-/root/kevin_wang/envs/fastwam_guidemem/bin}
memorywam_root=${MEMORYWAM_ROOT:-/mnt/vepfs02/output/kevin_wang/memorywam}
preflight_only=${PREFLIGHT_ONLY:-0}
require_idle_gpus=${REQUIRE_IDLE_GPUS:-1}
run_online_parity=${RUN_ONLINE_PARITY:-1}
die(){ echo "ERROR: $*" >&2; exit 2; }

method=fixed
manifest_root=
case ${experiment} in
  putback_dynamic)
    method=dynamic
    task_name=rmbench_putback_multimodal_rate_controlled_k8_scratch_joint_20k_lr2e4
    manifest_task=put_back_block
    manifest_root=${MULTIMODAL_RATE_CONTROLLED_MANIFEST:-/mnt/vepfs01/output/kevin.wang/memorywam/analysis/putback_multimodal_rate_controlled_l4to8_k8_v2}
    output_dir=${OUTPUT_DIR:-/mnt/vepfs01/output/kevin.wang/memorywam/train/multimodal_rate_k8_putback_scratch_joint20k_lr2e4_seed42_v1}
    max_steps=20000
    save_steps=10000,15000,16000,17000,18000,19000,20000
    ;;
  putback_fixed)
    task_name=rmbench_putback_fixed_l7_k8_scratch_joint_20k_lr2e4
    manifest_task=put_back_block
    output_dir=${OUTPUT_DIR:-/mnt/vepfs01/output/kevin.wang/memorywam/train/fixed_l7_k8_putback_scratch_joint20k_lr2e4_seed42_v1}
    max_steps=20000
    save_steps=10000,15000,16000,17000,18000,19000,20000
    ;;
  battery_dynamic)
    method=dynamic
    task_name=rmbench_battery_multimodal_rate_controlled_k8_scratch_joint_30k_lr2e4
    manifest_task=battery_try
    manifest_root=${MULTIMODAL_RATE_CONTROLLED_MANIFEST:-/mnt/vepfs01/output/kevin.wang/memorywam/analysis/battery_multimodal_rate_controlled_l4to8_k8_v2}
    output_dir=${OUTPUT_DIR:-/mnt/vepfs01/output/kevin.wang/memorywam/train/multimodal_rate_k8_battery_scratch_joint30k_lr2e4_seed42_v1}
    max_steps=30000
    save_steps=15000,20000,25000,30000
    ;;
  battery_fixed)
    task_name=rmbench_battery_fixed_l6_k8_scratch_joint_30k_lr2e4
    manifest_task=battery_try
    output_dir=${OUTPUT_DIR:-/mnt/vepfs01/output/kevin.wang/memorywam/train/fixed_l6_k8_battery_scratch_joint30k_lr2e4_seed42_v1}
    max_steps=30000
    save_steps=15000,20000,25000,30000
    ;;
  *) die "Unsupported EXPERIMENT=${experiment}" ;;
esac

[[ -d ${repo_root} && -x ${runtime_bin}/python ]] || die "Missing repository or runtime"
[[ $(readlink -m "${output_dir}") == /mnt/vepfs01/output/* ]] || die "OUTPUT_DIR must be on vepfs01"
export PATH=${runtime_bin}:${PATH}
export PYTHONPATH=${repo_root}/src:${repo_root}${PYTHONPATH:+:${PYTHONPATH}}
export DIFFSYNTH_MODEL_BASE_PATH=${DIFFSYNTH_MODEL_BASE_PATH:-/mnt/vepfs02/datasets/kevin_wang/code/projects/FastWAM/checkpoints}
export DIFFSYNTH_SKIP_DOWNLOAD=true
export FASTWAM_ACTION_DIT_INIT=${FASTWAM_ACTION_DIT_INIT:-${memorywam_root}/model_assets/fastwam_fullkv_official_init/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export PYTHONUTF8=1 LANG=C.UTF-8 LC_ALL=C.UTF-8 PYTHONDONTWRITEBYTECODE=1

if [[ ${manifest_task} == put_back_block ]]; then
  export RMBENCH_PUTBACK_LEROBOT=${RMBENCH_PUTBACK_LEROBOT:-${memorywam_root}/data/rmbench_lerobot_v21_rgb_v3/put_back_block}
  export MEMORYWAM_PUTBACK_STATS=${MEMORYWAM_PUTBACK_STATS:-${memorywam_root}/data/fastwam_putback_stats_rgb_v3/dataset_stats.json}
  export MEMORYWAM_PUTBACK_CONTINUOUS_LATENTS=${MEMORYWAM_PUTBACK_CONTINUOUS_LATENTS:-${memorywam_root}/data/fastwam_putback_continuous_episode_stride16_v4}
  export MEMORYWAM_PUTBACK_TEXT_CACHE=${MEMORYWAM_PUTBACK_TEXT_CACHE:-${memorywam_root}/data/fastwam_putback_text_cache_rgb_v3}
  export PUTBACK_MULTIMODAL_RATE_OUTPUT=${output_dir}
  export PUTBACK_FIXED_L7_K8_OUTPUT=${output_dir}
  lerobot_root=${RMBENCH_PUTBACK_LEROBOT}
  stats_path=${MEMORYWAM_PUTBACK_STATS}
  latent_root=${MEMORYWAM_PUTBACK_CONTINUOUS_LATENTS}
  text_root=${MEMORYWAM_PUTBACK_TEXT_CACHE}
else
  battery_root=${BATTERY_ASSET_ROOT:-${memorywam_root}/data/fastwam_fullkv_remaining8_rgb_v3/battery_try}
  export RMBENCH_BATTERY_LEROBOT=${RMBENCH_BATTERY_LEROBOT:-${battery_root}/lerobot/battery_try}
  export MEMORYWAM_BATTERY_STATS=${MEMORYWAM_BATTERY_STATS:-${battery_root}/stats/dataset_stats.json}
  export MEMORYWAM_BATTERY_CONTINUOUS_LATENTS=${MEMORYWAM_BATTERY_CONTINUOUS_LATENTS:-${battery_root}/temporal_fullkv_continuous_episode_stride16_v4}
  export MEMORYWAM_BATTERY_TEXT_CACHE=${MEMORYWAM_BATTERY_TEXT_CACHE:-${battery_root}/text_cache}
  export BATTERY_MULTIMODAL_RATE_OUTPUT=${output_dir}
  export BATTERY_FIXED_L6_K8_OUTPUT=${output_dir}
  lerobot_root=${RMBENCH_BATTERY_LEROBOT}
  stats_path=${MEMORYWAM_BATTERY_STATS}
  latent_root=${MEMORYWAM_BATTERY_CONTINUOUS_LATENTS}
  text_root=${MEMORYWAM_BATTERY_TEXT_CACHE}
fi
[[ -s ${FASTWAM_ACTION_DIT_INIT} && -s ${lerobot_root}/meta/info.json && -s ${stats_path} && -s ${latent_root}/manifest.json ]] || die "Missing model or data artifacts"
[[ -d ${text_root} ]] || die "Missing text cache"

cd "${repo_root}"
summary='{"selector":"fixed"}'
if [[ ${method} == dynamic ]]; then
  [[ -s ${manifest_root}/manifest.json ]] || die "Missing multimodal manifest"
  export MULTIMODAL_RATE_CONTROLLED_MANIFEST=${manifest_root}
  summary=$("${runtime_bin}/python" - "${manifest_root}" "${manifest_task}" <<'PY'
import collections,json,sys
from fastwam.memory.multimodal_rate_controlled import MultimodalRateManifestStore
s=MultimodalRateManifestStore(sys.argv[1],expected_episode_count=50,expected_task=sys.argv[2])
h=collections.Counter(x.length for e in range(50) for x in s.segments_for_episode(e))
r=collections.Counter(x.reason for e in range(50) for x in s.segments_for_episode(e))
assert h and min(h)>=4 and max(h)<=8
print(json.dumps({"lengths":dict(sorted(h.items())),"reasons":dict(sorted(r.items()))}))
PY
  )
  if [[ ${run_online_parity} == 1 ]]; then
    "${runtime_bin}/python" scripts/check_multimodal_rate_controlled_online_parity.py --manifest "${manifest_root}" --latent-root "${latent_root}" --lerobot-root "${lerobot_root}" --task "${manifest_task}"
  fi
fi

resolved=$("${runtime_bin}/python" scripts/train.py "task=${task_name}" --cfg job --resolve)
RESOLVED=${resolved} OUTPUT=${output_dir} MAX_STEPS=${max_steps} SAVE_STEPS=${save_steps} METHOD=${method} MANIFEST=${manifest_root} "${runtime_bin}/python" - <<'PY' || die "Resolved contract mismatch"
import os
from omegaconf import OmegaConf
c=OmegaConf.create(os.environ["RESOLVED"])
expected={
 "output_dir":os.environ["OUTPUT"],"initial_weights":None,"resume":None,
 "learning_rate":2e-4,"max_steps":int(os.environ["MAX_STEPS"]),"save_every":0,
 "save_steps":[int(x) for x in os.environ["SAVE_STEPS"].split(",")],
 "batch_size":1,"num_gpus":8,"data.train.minimum_history_frames":1,
 "native_cache_train_mode":"full","model.native_cache.memory_tokens":8,
 "model.native_cache.anchor_frames":2,"model.native_cache.recent_frames":4,
 "model.loss.lambda_video":1.0,"model.loss.lambda_action":1.0,
}
if os.environ["METHOD"]=="dynamic":
 expected["data.train.multimodal_manifest_path"]=os.environ["MANIFEST"]
 expected["data.train.memory_tokens"]=8
else:
 expected["data.train.memory_tokens"]=8
 expected["data.train.fixed_segment_length"]=7 if "putback" in os.environ["RESOLVED"].lower() else 6
for path,want in expected.items():
 got=OmegaConf.select(c,path)
 if got!=want: raise SystemExit(f"{path}: {got!r} != {want!r}")
if "robotwin_uncond_3cam_384.pt" in os.environ["RESOLVED"].lower():
 raise SystemExit("RobotWin checkpoint leaked into scratch config")
PY
echo "experiment=${experiment} task=${manifest_task} method=${method} scratch=true joint=true lr=2e-4 steps=${max_steps} save_steps=${save_steps} summary=${summary} output=${output_dir}"
[[ ${preflight_only} == 0 ]] || { echo preflight_status=ok; exit 0; }
[[ ! -e ${output_dir} ]] || die "Refusing to overwrite ${output_dir}"
IFS=',' read -r -a gpus <<<"${CUDA_VISIBLE_DEVICES}"; [[ ${#gpus[@]} -eq 8 ]] || die "Need 8 GPUs"
if [[ ${require_idle_gpus} == 1 ]]; then active=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d'|sort -u); [[ -z ${active} ]] || die "GPUs are busy"; fi
mkdir -p "${output_dir}/contracts"
printf '%s\n' "${resolved}" >"${output_dir}/contracts/train_resolved.yaml"
printf '%s\n' "${summary}" >"${output_dir}/contracts/segmentation_summary.json"
exec > >(tee -a "${output_dir}/train.log") 2>&1
accelerate launch --config_file scripts/accelerate_configs/accelerate_zero2_ds.yaml --num_processes 8 scripts/train.py "task=${task_name}"
IFS=',' read -r -a expected_steps <<<"${save_steps}"
for step in "${expected_steps[@]}"; do
  checkpoint=${output_dir}/checkpoints/weights/step_$(printf '%06d' "${step}").pt
  [[ -s ${checkpoint} ]] || die "Missing expected checkpoint ${checkpoint}"
done
