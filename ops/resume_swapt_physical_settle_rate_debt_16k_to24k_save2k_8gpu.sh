#!/usr/bin/env bash
set -euo pipefail

repo_root=${REPO_ROOT:-/mnt/vepfs02/output/kevin_wang/memorywam/code/fastwam_physical_settle_rate_debt_k8}
runtime_bin=${RUNTIME_BIN:-/root/kevin_wang/envs/fastwam_guidemem/bin}
memorywam_root=${MEMORYWAM_ROOT:-/mnt/vepfs02/output/kevin_wang/memorywam}
source_run=${SOURCE_RUN:-/mnt/vepfs01/output/kevin.wang/memorywam/train/physical_settle_t7c5_k8_swapt_scratch_joint16k_lr2e4_seed42_v1}
resume_state=${RESUME_STATE:-${source_run}/checkpoints/state/step_016000}
output_dir=${OUTPUT_DIR:-/mnt/vepfs01/output/kevin.wang/memorywam/train/physical_settle_t7c5_k8_swapt_resume16k_to24k_lr2e4_seed42_v1}
asset_root=${SWAPT_ASSET_ROOT:-${memorywam_root}/data/fastwam_fullkv_remaining8_rgb_v3/swap_T}
manifest_root=${PHYSICAL_SETTLE_RATE_DEBT_MANIFEST:-/mnt/vepfs01/output/kevin.wang/memorywam/analysis/swapt_physical_settle_rate_debt_t7c5_k8_v1}
preflight_only=${PREFLIGHT_ONLY:-0}
require_idle_gpus=${REQUIRE_IDLE_GPUS:-1}

die() { echo "ERROR: $*" >&2; exit 2; }
[[ ${preflight_only} == 0 || ${preflight_only} == 1 ]] || die "PREFLIGHT_ONLY must be 0 or 1"
[[ -d ${repo_root} && -x ${runtime_bin}/python ]] || die "missing repository or runtime"
[[ $(readlink -m "${output_dir}") == /mnt/vepfs01/output/* ]] || die "OUTPUT_DIR must be on vepfs01"
[[ ! -e ${output_dir} ]] || die "refusing to overwrite ${output_dir}"
[[ ${resume_state} == */checkpoints/state/step_016000 ]] || die "resume must use the complete 16k state"

export PATH=${runtime_bin}:${PATH}
export PYTHONPATH=${repo_root}/src:${repo_root}${PYTHONPATH:+:${PYTHONPATH}}
export DIFFSYNTH_MODEL_BASE_PATH=${DIFFSYNTH_MODEL_BASE_PATH:-/mnt/vepfs02/datasets/kevin_wang/code/projects/FastWAM/checkpoints}
export DIFFSYNTH_SKIP_DOWNLOAD=true
export RMBENCH_COVER_BLOCKS_LEROBOT=${asset_root}/lerobot/swap_T
export MEMORYWAM_COVER_BLOCKS_STATS=${asset_root}/stats/dataset_stats.json
export MEMORYWAM_COVER_BLOCKS_CONTINUOUS_LATENTS=${asset_root}/temporal_fullkv_continuous_episode_stride16_v4
export MEMORYWAM_COVER_BLOCKS_TEXT_CACHE=${asset_root}/text_cache
export FASTWAM_ACTION_DIT_INIT=${FASTWAM_ACTION_DIT_INIT:-${memorywam_root}/model_assets/fastwam_fullkv_official_init/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt}
export PHYSICAL_SETTLE_RATE_DEBT_MANIFEST=${manifest_root}
export SWAPT_PHYSICAL_SETTLE_RATE_OUTPUT=${output_dir}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export PYTHONUTF8=1 LANG=C.UTF-8 LC_ALL=C.UTF-8 PYTHONDONTWRITEBYTECODE=1

IFS=',' read -r -a visible_gpus <<<"${CUDA_VISIBLE_DEVICES}"
[[ ${#visible_gpus[@]} -eq 8 ]] || die "formal continuation requires exactly eight GPUs"

for required in \
  "${resume_state}/pytorch_model/mp_rank_00_model_states.pt" \
  "${resume_state}/scheduler.bin" \
  "${resume_state}/trainer_state.json" \
  "${source_run}/checkpoints/weights/step_016000.pt" \
  "${manifest_root}/manifest.json" \
  "${MEMORYWAM_COVER_BLOCKS_STATS}" \
  "${MEMORYWAM_COVER_BLOCKS_CONTINUOUS_LATENTS}/manifest.json" \
  "${FASTWAM_ACTION_DIT_INIT}"; do
  [[ -s ${required} ]] || die "missing resume dependency: ${required}"
done
for rank in {0..7}; do
  shard=${resume_state}/pytorch_model/bf16_zero_pp_rank_${rank}_mp_rank_00_optim_states.pt
  [[ -s ${shard} && $(stat -c %s "${shard}") -gt 8000000000 ]] || die "incomplete optimizer shard ${rank}"
done

cd "${repo_root}"
"${runtime_bin}/python" - "${resume_state}" "${source_run}/checkpoints/weights/step_016000.pt" <<'PY'
import json, sys
from fastwam.utils.compact_checkpoint import validate_portable_checkpoint
state, portable = sys.argv[1:]
payload = json.load(open(state + "/trainer_state.json"))
assert payload["global_step"] == 16000, payload
validate_portable_checkpoint(portable, expected_step=16000)
print("resume_checkpoint_validation=pass")
PY

resolved=$("${runtime_bin}/python" scripts/train.py \
  task=rmbench_swapt_physical_settle_rate_debt_k8_scratch_joint_16k_lr2e4 \
  output_dir="${output_dir}" \
  resume="${resume_state}" \
  max_steps=24000 \
  save_every=2000 \
  keep_training_state_checkpoints=1 \
  --cfg job --resolve)
RESOLVED=${resolved} OUTPUT=${output_dir} RESUME=${resume_state} "${runtime_bin}/python" - <<'PY'
import os
from omegaconf import OmegaConf
c = OmegaConf.create(os.environ["RESOLVED"])
expected = {
    "output_dir": os.environ["OUTPUT"],
    "resume": os.environ["RESUME"],
    "initial_weights": None,
    "max_steps": 24000,
    "save_every": 2000,
    "save_training_state": True,
    "keep_training_state_checkpoints": 1,
    "learning_rate": 2e-4,
    "batch_size": 1,
    "num_gpus": 8,
    "gradient_accumulation_steps": 1,
    "seed": 42,
    "mixed_precision": "bf16",
    "data.train.minimum_history_frames": 1,
    "native_cache_train_mode": "full",
    "model.loss.lambda_video": 1.0,
    "model.loss.lambda_action": 1.0,
    "model.native_cache.mode": "layerwise",
    "model.native_cache.memory_tokens": 8,
    "model.native_cache.anchor_frames": 2,
    "model.native_cache.recent_frames": 4,
    "model.native_cache.recursive": False,
}
for key, want in expected.items():
    got = OmegaConf.select(c, key)
    if got != want:
        raise SystemExit(f"config mismatch {key}: {got!r} != {want!r}")
print("resolved_contract_validation=pass")
PY

echo "task=swap_T resume=16000 target=24000 save_steps=18000,20000,22000,24000 lr=2e-4 global_batch=8 output=${output_dir}"
[[ ${preflight_only} == 0 ]] || { echo "preflight_status=ok"; exit 0; }

if [[ ${require_idle_gpus} == 1 ]]; then
  active=$(nvidia-smi -i "${CUDA_VISIBLE_DEVICES}" --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d' | sort -u)
  [[ -z ${active} ]] || die "selected GPUs are busy"
fi

mkdir -p "${output_dir}/contracts"
printf '%s\n' "${resolved}" >"${output_dir}/contracts/train_resolved.yaml"
exec > >(tee -a "${output_dir}/train.log") 2>&1
echo "training_resume_start=$(date --iso-8601=seconds)"
accelerate launch \
  --config_file scripts/accelerate_configs/accelerate_zero2_ds.yaml \
  --num_processes 8 \
  scripts/train.py \
  task=rmbench_swapt_physical_settle_rate_debt_k8_scratch_joint_16k_lr2e4 \
  output_dir="${output_dir}" \
  resume="${resume_state}" \
  max_steps=24000 \
  save_every=2000 \
  keep_training_state_checkpoints=1
