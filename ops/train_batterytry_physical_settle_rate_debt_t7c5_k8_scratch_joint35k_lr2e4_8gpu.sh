#!/usr/bin/env bash
set -euo pipefail

repo_root=${REPO_ROOT:-/mnt/vepfs02/output/kevin_wang/memorywam/code/fastwam_physical_settle_rate_debt_k8}
runtime_bin=${RUNTIME_BIN:-/root/kevin_wang/envs/fastwam_guidemem/bin}
memorywam_root=${MEMORYWAM_ROOT:-/mnt/vepfs02/output/kevin_wang/memorywam}
asset_root=${BATTERYTRY_ASSET_ROOT:-${memorywam_root}/data/fastwam_fullkv_remaining8_rgb_v3/battery_try}
output_dir=${OUTPUT_DIR:-/mnt/vepfs01/output/kevin.wang/memorywam/train/physical_settle_t7c5_k8_batterytry_scratch_joint35k_lr2e4_seed42_v1}
manifest_root=${PHYSICAL_SETTLE_RATE_DEBT_MANIFEST:-/mnt/vepfs01/output/kevin.wang/memorywam/analysis/batterytry_physical_settle_rate_debt_t7c5_k8_v1}
task_name=rmbench_batterytry_physical_settle_rate_debt_k8_scratch_joint_35k_lr2e4
num_processes=${NUM_PROCESSES:-8}
[[ ${num_processes} == 4 || ${num_processes} == 8 ]] || { echo "ERROR: NUM_PROCESSES must be 4 or 8" >&2; exit 2; }
gradient_accumulation_steps=$((8 / num_processes))
preflight_only=${PREFLIGHT_ONLY:-0}
require_idle_gpus=${REQUIRE_IDLE_GPUS:-1}
run_online_parity=${RUN_ONLINE_PARITY:-1}
die(){ echo "ERROR: $*" >&2; exit 2; }

[[ -d ${repo_root} && -x ${runtime_bin}/python ]] || die "Missing repository or runtime"
[[ $(readlink -m "${output_dir}") == /mnt/vepfs01/output/* ]] || die "OUTPUT_DIR must be on vepfs01"
[[ $(readlink -m "${manifest_root}") == /mnt/vepfs01/output/* ]] || die "Manifest must be on vepfs01"
export PATH=${runtime_bin}:${PATH}
export PYTHONPATH=${repo_root}/src:${repo_root}${PYTHONPATH:+:${PYTHONPATH}}
export DIFFSYNTH_MODEL_BASE_PATH=${DIFFSYNTH_MODEL_BASE_PATH:-/mnt/vepfs02/datasets/kevin_wang/code/projects/FastWAM/checkpoints}
export DIFFSYNTH_SKIP_DOWNLOAD=true
export RMBENCH_BATTERYTRY_LEROBOT=${asset_root}/lerobot/battery_try
export MEMORYWAM_BATTERYTRY_STATS=${asset_root}/stats/dataset_stats.json
export MEMORYWAM_BATTERYTRY_CONTINUOUS_LATENTS=${asset_root}/temporal_fullkv_continuous_episode_stride16_v4
export MEMORYWAM_BATTERYTRY_TEXT_CACHE=${asset_root}/text_cache
export FASTWAM_ACTION_DIT_INIT=${FASTWAM_ACTION_DIT_INIT:-${memorywam_root}/model_assets/fastwam_fullkv_official_init/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt}
export PHYSICAL_SETTLE_RATE_DEBT_MANIFEST=${manifest_root}
export BATTERYTRY_PHYSICAL_SETTLE_RATE_OUTPUT=${output_dir}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export PYTHONUTF8=1 LANG=C.UTF-8 LC_ALL=C.UTF-8 PYTHONDONTWRITEBYTECODE=1
for f in "${FASTWAM_ACTION_DIT_INIT}" "${RMBENCH_BATTERYTRY_LEROBOT}/meta/info.json" "${MEMORYWAM_BATTERYTRY_STATS}" "${MEMORYWAM_BATTERYTRY_CONTINUOUS_LATENTS}/manifest.json"; do [[ -s ${f} ]] || die "Missing ${f}"; done
[[ -d ${MEMORYWAM_BATTERYTRY_TEXT_CACHE} ]] || die "Missing text cache"

cd "${repo_root}"
if [[ ! -e ${manifest_root} ]]; then
  mkdir -p "$(dirname "${manifest_root}")"
  "${runtime_bin}/python" scripts/build_physical_settle_rate_debt_manifest.py \
    --lerobot-root "${RMBENCH_BATTERYTRY_LEROBOT}" \
    --output "${manifest_root}" \
    --task battery_try \
    --episodes 50 \
    --replan-stride 16
fi
[[ -s ${manifest_root}/manifest.json ]] || die "Missing or incomplete physical settle/rate-debt manifest"

summary=$("${runtime_bin}/python" - "${manifest_root}" <<'PY'
import collections,json,sys,torch
from fastwam.memory.physical_settle_rate_debt import PhysicalSettleRateManifestStore
from fastwam.memory.native_cache import build_dynamic_layerwise_training_layout
s=PhysicalSettleRateManifestStore(sys.argv[1],expected_episode_count=50,expected_task="battery_try")
h=collections.Counter(x.length for e in range(50) for x in s.segments_for_episode(e))
r=collections.Counter(x.reason for e in range(50) for x in s.segments_for_episode(e))
assert sum(h.values()) > 0 and all(4 <= length <= 8 for length in h), (h,r)
assert s.metadata["event_threshold"] == 0.8
assert s.metadata["target_mean_segment"] == 7.0
assert s.metadata["maximum_rate_debt"] == 5.0
assert s.metadata["reader_raw_tail"] == "complete_open_segment_suffix"
assert s.metadata["fixed_reader_recent_frames"] is False
layout=build_dynamic_layerwise_training_layout(clean_frames=11,noisy_frames=1,tokens_per_frame=2,action_tokens=3,memory_groups=((2,3,4,5),),memory_tokens=8,anchor_frames=2,recent_frames=4,device=torch.device("cpu"))
assert layout.retained_clean_tokens == 22
source=next(x for x in layout.segments if x.kind=="source")
assert not layout.attention_mask[layout.action_range[0]:layout.action_range[1],source.start:source.stop].any()
print(json.dumps({"lengths":dict(sorted(h.items())),"reasons":dict(sorted(r.items()))}))
PY
)
if [[ ${run_online_parity} == 1 ]]; then
  "${runtime_bin}/python" scripts/check_physical_settle_rate_debt_online_parity.py \
    --manifest "${manifest_root}" \
    --lerobot-root "${RMBENCH_BATTERYTRY_LEROBOT}" \
    --task battery_try
fi
resolved=$("${runtime_bin}/python" scripts/train.py "task=${task_name}" "num_gpus=${num_processes}" "gradient_accumulation_steps=${gradient_accumulation_steps}" --cfg job --resolve)
RESOLVED=${resolved} OUTPUT=${output_dir} MANIFEST=${manifest_root} NUM_PROCESSES=${num_processes} GRAD_ACCUM=${gradient_accumulation_steps} "${runtime_bin}/python" - <<'PY' || die "Resolved contract mismatch"
import os
from omegaconf import OmegaConf
c=OmegaConf.create(os.environ["RESOLVED"])
e={"output_dir":os.environ["OUTPUT"],"initial_weights":None,"resume":None,"learning_rate":2e-4,"max_steps":35000,"save_every":0,"save_steps":[5000,10000,20000,25000,30000,35000],"save_training_state_every":35000,"batch_size":1,"num_gpus":int(os.environ["NUM_PROCESSES"]),"gradient_accumulation_steps":int(os.environ["GRAD_ACCUM"]),"data.train.minimum_history_frames":1,"data.train.physical_settle_manifest_path":os.environ["MANIFEST"],"data.train.manifest_task":"battery_try","native_cache_train_mode":"full","model.loss.lambda_video":1.0,"model.loss.lambda_action":1.0}
for k,v in e.items():
    got=OmegaConf.select(c,k)
    if got!=v: raise SystemExit(f"{k}: {got!r} != {v!r}")
if "robotwin_uncond_3cam_384.pt" in os.environ["RESOLVED"].lower(): raise SystemExit("RobotWin checkpoint leaked into scratch config")
PY
echo "task=battery_try method=physical_settle_rate_debt_t7c5_l4to8_k8 dynamic_raw_tail=true scratch=true joint=true lr=2e-4 steps=35000 save_steps=5k,10k,20k,25k,30k,35k processes=${num_processes} grad_accum=${gradient_accumulation_steps} global_batch=8 manifest=${summary} output=${output_dir}"
[[ ${preflight_only} == 0 ]] || { echo preflight_status=ok; exit 0; }
[[ ! -e ${output_dir} ]] || die "Refusing to overwrite ${output_dir}"
IFS=',' read -r -a gpus <<<"${CUDA_VISIBLE_DEVICES}"
[[ ${#gpus[@]} -eq ${num_processes} ]] || die "Visible GPU count must equal NUM_PROCESSES"
if [[ ${require_idle_gpus} == 1 ]]; then
  active=$(nvidia-smi -i "${CUDA_VISIBLE_DEVICES}" --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d' | sort -u)
  [[ -z ${active} ]] || die "Selected GPUs are busy"
fi
mkdir -p "${output_dir}/contracts"
printf '%s\n' "${resolved}" >"${output_dir}/contracts/train_resolved.yaml"
printf '%s\n' "${summary}" >"${output_dir}/contracts/manifest_summary.json"
exec > >(tee -a "${output_dir}/train.log") 2>&1
accelerate launch --config_file scripts/accelerate_configs/accelerate_zero2_ds.yaml --num_processes "${num_processes}" scripts/train.py "task=${task_name}" "num_gpus=${num_processes}" "gradient_accumulation_steps=${gradient_accumulation_steps}"
