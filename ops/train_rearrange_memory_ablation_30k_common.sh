#!/usr/bin/env bash
set -euo pipefail

: "${ABLATION:?ABLATION is required}"
case "${ABLATION}" in
  without_anchor|without_recent|without_middle_memory|fixed_l4) ;;
  *) echo "ERROR: unsupported ABLATION=${ABLATION}" >&2; exit 2 ;;
esac

repo_root=${REPO_ROOT:-/mnt/vepfs02/output/kevin_wang/memorywam/code/fastwam_rearrange_memory_ablations_k8}
runtime_bin=${RUNTIME_BIN:-/root/kevin_wang/envs/fastwam_guidemem/bin}
memorywam_root=${MEMORYWAM_ROOT:-/mnt/vepfs02/output/kevin_wang/memorywam}
asset_root=${REARRANGE_ASSET_ROOT:-${memorywam_root}/data/fastwam_fullkv_remaining8_rgb_v3/rearrange_blocks}
output_dir=${OUTPUT_DIR:-/mnt/vepfs01/output/kevin.wang/memorywam/train/rearrange_${ABLATION}_k8_scratch_joint30k_lr2e4_seed42_v1}
manifest_root=${PHYSICAL_SETTLE_RATE_DEBT_MANIFEST:-/mnt/vepfs01/output/kevin.wang/memorywam/analysis/rearrange_blocks_physical_settle_rate_debt_t7c5_k8_v1}
preflight_only=${PREFLIGHT_ONLY:-0}
launch_check_only=${LAUNCH_CHECK_ONLY:-0}
require_idle_gpus=${REQUIRE_IDLE_GPUS:-1}
die(){ echo "ERROR: $*" >&2; exit 2; }

[[ ${preflight_only} == 0 || ${preflight_only} == 1 ]] || die "PREFLIGHT_ONLY must be 0 or 1"
[[ ${launch_check_only} == 0 || ${launch_check_only} == 1 ]] || die "LAUNCH_CHECK_ONLY must be 0 or 1"

[[ -d ${repo_root} && -x ${runtime_bin}/python ]] || die "Missing repository or runtime"
[[ $(readlink -m "${output_dir}") == /mnt/vepfs01/output/* ]] || die "OUTPUT_DIR must be on vepfs01"
export PATH=${runtime_bin}:${PATH}
export PYTHONPATH=${repo_root}/src:${repo_root}${PYTHONPATH:+:${PYTHONPATH}}
export DIFFSYNTH_MODEL_BASE_PATH=${DIFFSYNTH_MODEL_BASE_PATH:-/mnt/vepfs02/datasets/kevin_wang/code/projects/FastWAM/checkpoints}
export DIFFSYNTH_SKIP_DOWNLOAD=true
export FASTWAM_ACTION_DIT_INIT=${FASTWAM_ACTION_DIT_INIT:-${memorywam_root}/model_assets/fastwam_fullkv_official_init/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt}
export RMBENCH_PHYSICAL_TASK_LEROBOT=${asset_root}/lerobot/rearrange_blocks
export MEMORYWAM_PHYSICAL_TASK_STATS=${asset_root}/stats/dataset_stats.json
export MEMORYWAM_PHYSICAL_TASK_CONTINUOUS_LATENTS=${asset_root}/temporal_fullkv_continuous_episode_stride16_v4
export MEMORYWAM_PHYSICAL_TASK_TEXT_CACHE=${asset_root}/text_cache
export PHYSICAL_SETTLE_RATE_DEBT_MANIFEST=${manifest_root}
export PHYSICAL_SETTLE_RATE_OUTPUT=${output_dir}
# Fixed-L4 inherits the generic FullKV data file, whose historical variable
# names say COVER_BLOCKS.  They intentionally point at Rearrange here.
export RMBENCH_COVER_BLOCKS_LEROBOT=${RMBENCH_PHYSICAL_TASK_LEROBOT}
export MEMORYWAM_COVER_BLOCKS_STATS=${MEMORYWAM_PHYSICAL_TASK_STATS}
export MEMORYWAM_COVER_BLOCKS_CONTINUOUS_LATENTS=${MEMORYWAM_PHYSICAL_TASK_CONTINUOUS_LATENTS}
export MEMORYWAM_COVER_BLOCKS_TEXT_CACHE=${MEMORYWAM_PHYSICAL_TASK_TEXT_CACHE}
export REARRANGE_ABLATION_OUTPUT=${output_dir}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}
export PYTHONUTF8=1 LANG=C.UTF-8 LC_ALL=C.UTF-8 PYTHONDONTWRITEBYTECODE=1
export HYDRA_FULL_ERROR=1

for required in \
  "${FASTWAM_ACTION_DIT_INIT}" \
  "${RMBENCH_PHYSICAL_TASK_LEROBOT}/meta/info.json" \
  "${MEMORYWAM_PHYSICAL_TASK_STATS}" \
  "${MEMORYWAM_PHYSICAL_TASK_CONTINUOUS_LATENTS}/manifest.json"; do
  [[ -s ${required} ]] || die "Missing or empty required artifact: ${required}"
done
[[ -d ${MEMORYWAM_PHYSICAL_TASK_TEXT_CACHE} ]] || die "Missing text cache"

reader_anchor=true
reader_memory=true
reader_recent=true
task=rmbench_rearrange_blocks_physical_settle_rate_debt_k8_scratch_joint_40k_lr2e4
extra=(
  max_steps=30000
  'save_steps=[30000]'
  save_training_state=false
  save_final_checkpoint=true
  save_portable_fsdp_checkpoint=true
)
case "${ABLATION}" in
  without_anchor) reader_anchor=false ;;
  without_recent) reader_recent=false ;;
  without_middle_memory) reader_memory=false ;;
  fixed_l4)
    task=rmbench_rearrange_blocks_fixed_l4_k8_scratch_joint_30k_lr2e4
    extra=()
    ;;
esac
if [[ ${ABLATION} != fixed_l4 ]]; then
  [[ -s ${manifest_root}/manifest.json ]] || die "Missing dynamic manifest: ${manifest_root}"
  extra+=(
    "+model.native_cache.reader_use_anchor=${reader_anchor}"
    "+model.native_cache.reader_use_memory=${reader_memory}"
    "+model.native_cache.reader_use_recent=${reader_recent}"
  )
fi

cd "${repo_root}"
# On a fresh cloud container, LeRobot/Hugging Face creates its local Arrow
# index during the first dataset construction.  Starting eight ranks against
# a cold cache can race on temporary files and surface as FileNotFoundError.
# Build the cache once and read representative real samples before launch.
dataset_preflight_args=(--task "${task}")
for override in "${extra[@]}"; do
  dataset_preflight_args+=(--override "${override}")
done
"${runtime_bin}/python" scripts/preflight_rearrange_ablation_dataset.py \
  "${dataset_preflight_args[@]}"
resolved=$("${runtime_bin}/python" scripts/train.py "task=${task}" "${extra[@]}" --cfg job --resolve)
RESOLVED=${resolved} OUTPUT=${output_dir} ABLATION=${ABLATION} READER_ANCHOR=${reader_anchor} READER_MEMORY=${reader_memory} READER_RECENT=${reader_recent} "${runtime_bin}/python" - <<'PY' || die "Resolved contract mismatch"
import os
from omegaconf import OmegaConf
c = OmegaConf.create(os.environ["RESOLVED"])
expected = {
    "output_dir": os.environ["OUTPUT"],
    "initial_weights": None,
    "resume": None,
    "learning_rate": 2e-4,
    "lr_scheduler_type": "constant",
    "max_steps": 30000,
    "save_every": 0,
    "save_steps": [30000],
    "save_training_state": False,
    "save_final_checkpoint": True,
    "save_portable_fsdp_checkpoint": True,
    "batch_size": 1,
    "num_gpus": 8,
    "gradient_accumulation_steps": 1,
    "mixed_precision": "bf16",
    "seed": 42,
    "data.train.minimum_history_frames": 1,
    "native_cache_train_mode": "full",
    "model.loss.lambda_video": 1.0,
    "model.loss.lambda_action": 1.0,
    "model.native_cache.memory_tokens": 8,
    "model.native_cache.anchor_frames": 2,
    "model.native_cache.recent_frames": 4,
}
if os.environ["ABLATION"] == "fixed_l4":
    expected.update({
        "data.train._target_": "fastwam.datasets.lerobot.fixed_length_k8_dataset.FixedLengthK8RobotVideoDataset",
        "data.train.fixed_segment_length": 4,
        "data.train.confirmation_delay": 4,
    })
else:
    expected.update({
        "data.train._target_": "fastwam.datasets.lerobot.physical_settle_rate_debt_dataset.PhysicalSettleRateDebtRobotVideoDataset",
        "model.native_cache.reader_use_anchor": os.environ["READER_ANCHOR"] == "true",
        "model.native_cache.reader_use_memory": os.environ["READER_MEMORY"] == "true",
        "model.native_cache.reader_use_recent": os.environ["READER_RECENT"] == "true",
    })
for key, want in expected.items():
    got = OmegaConf.select(c, key)
    if got != want:
        raise SystemExit(f"{key}: {got!r} != {want!r}")
if "robotwin_uncond_3cam_384.pt" in os.environ["RESOLVED"].lower():
    raise SystemExit("RobotWin pretrained checkpoint leaked into scratch config")
PY

echo "task=rearrange_blocks ablation=${ABLATION} scratch=true joint_video_action=true lr=2e-4 steps=30000 save_weights=30000 save_training_state=false"
echo "reader_anchor=${reader_anchor} reader_middle_memory=${reader_memory} reader_recent=${reader_recent} output=${output_dir}"
[[ ${preflight_only} == 0 ]] || { echo "preflight_status=ok"; exit 0; }
# Volcano/worker launchers may create OUTPUT_DIR, config.yaml, and train.log
# before this script starts.  Treat that as harmless scaffolding, but never
# resume over weights, optimizer state, or another live invocation.
if [[ -e ${output_dir} && ! -d ${output_dir} ]]; then
  die "OUTPUT_DIR exists but is not a directory: ${output_dir}"
fi
if [[ -d ${output_dir} ]]; then
  material=$(find "${output_dir}" -mindepth 1 \
    \( -name 'step_*.pt' -o -name '*.safetensors' -o -name 'training_state*' \
       -o -name 'optimizer*' -o -name 'scheduler*' -o -name '.fastwam_training_started' \) \
    -print -quit)
  [[ -z ${material} ]] || die "Refusing to overwrite material or active output: ${material}"
  unknown=$(find "${output_dir}" -mindepth 1 \
    ! -path "${output_dir}/config.yaml" \
    ! -path "${output_dir}/train.log" \
    ! -path "${output_dir}/contracts" \
    ! -path "${output_dir}/contracts/*" \
    -print -quit)
  [[ -z ${unknown} ]] || die "Refusing to overwrite unknown existing output: ${unknown}"
fi
IFS=',' read -r -a visible_gpus <<<"${CUDA_VISIBLE_DEVICES}"
[[ ${#visible_gpus[@]} -eq 8 ]] || die "Need exactly eight visible GPUs"
if [[ ${require_idle_gpus} == 1 ]]; then
  active=$(nvidia-smi -i "${CUDA_VISIBLE_DEVICES}" --query-compute-apps=pid --format=csv,noheader,nounits | sed '/^[[:space:]]*$/d' | sort -u)
  [[ -z ${active} ]] || die "Selected GPUs are busy: ${active}"
fi
mkdir -p "${output_dir}/contracts"
run_lock=${output_dir}/.fastwam_training_started
if ! (set -o noclobber; printf '%s\n' "pid=$$ host=$(hostname) started=$(date --iso-8601=seconds)" >"${run_lock}") 2>/dev/null; then
  die "Another launcher already claimed this output: ${run_lock}"
fi
printf '%s\n' "${resolved}" >"${output_dir}/contracts/train_resolved.yaml"
printf '%s\n' \
  "ablation=${ABLATION}" \
  "reader_anchor=${reader_anchor}" \
  "reader_middle_memory=${reader_memory}" \
  "reader_recent=${reader_recent}" \
  "save_steps=30000" \
  "save_training_state=false" \
  >"${output_dir}/contracts/ablation.txt"
if [[ ${launch_check_only} == 1 ]]; then
  rm -f -- "${run_lock}"
  echo "launch_check_status=ok"
  exit 0
fi
exec > >(tee -a "${output_dir}/train.log") 2>&1
set +e
accelerate launch \
  --config_file scripts/accelerate_configs/accelerate_zero2_ds.yaml \
  --num_processes 8 \
  scripts/train.py "task=${task}" "${extra[@]}"
status=$?
set -e
rm -f -- "${run_lock}"
exit "${status}"
