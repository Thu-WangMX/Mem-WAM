#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/comparison_env.sh"
cd "${COMPARISON_ROOT}"
backend="${MEMORY_BACKEND:-helios}"
case "${backend}" in
  helios) dataset=rmbench_helios; model=fastwam_helios; default_bs=16; default_ga=1 ;;
  memorywam) dataset=rmbench_rearrange_blocks_fixed_l4_k8; model=fastwam_segment_compress_k8; default_bs=1; default_ga=16 ;;
  fullkv) dataset=rmbench_cover_blocks_fullattention; model=fastwam_full_kv; default_bs=1; default_ga=16 ;;
  *) echo "MEMORY_BACKEND must be helios, memorywam, or fullkv" >&2; exit 1 ;;
esac
bs="${BATCH_SIZE:-${default_bs}}"; ga="${GRAD_ACCUM:-${default_ga}}"
if [[ "${backend}" != helios && "${bs}" != 1 ]]; then
  echo 'The preserved original memory implementation requires BATCH_SIZE=1; use GRAD_ACCUM to match effective batch.' >&2; exit 1
fi
gpus="${NUM_GPUS:-8}"
if [[ -n "${NUM_EPOCHS:-}" ]]; then
  [[ "${NUM_EPOCHS}" =~ ^[1-9][0-9]*$ ]] || { echo 'NUM_EPOCHS must be a positive integer' >&2; exit 1; }
  schedule_args=("max_steps=null" "num_epochs=${NUM_EPOCHS}")
  schedule_desc="epochs=${NUM_EPOCHS}"
elif [[ -n "${MAX_STEPS:-}" ]]; then
  [[ "${MAX_STEPS}" =~ ^[1-9][0-9]*$ ]] || { echo 'MAX_STEPS must be a positive integer' >&2; exit 1; }
  schedule_args=("max_steps=${MAX_STEPS}" "num_epochs=1")
  schedule_desc="steps=${MAX_STEPS}"
else
  schedule_args=("max_steps=null" "num_epochs=500")
  schedule_desc="epochs=500"
fi
run_name="${RMBENCH_TASK}_${backend}_$(date +%Y%m%d_%H%M%S)"
output="${OUTPUT_DIR:-${COMPARISON_ROOT}/runs/${run_name}}"
args=(task=rmbench_helios "data=${dataset}" "model=${model}"
  "output_dir=${output}" "batch_size=${bs}" "gradient_accumulation_steps=${ga}"
  "num_gpus=${gpus}" "${schedule_args[@]}" "save_every=2000" "save_final_checkpoint=true"
  model.video_scheduler.training_sampling_scheme=logit_normal
  model.action_scheduler.training_sampling_scheme=logit_normal
  "wandb.enabled=${WANDB_ENABLED:-false}" "wandb.project=${WANDB_PROJECT:-fastwam-memory-comparison}"
  "wandb.name=${run_name}" "wandb.mode=${WANDB_MODE:-offline}")
echo "backend=${backend} task=${RMBENCH_TASK} bs/gpu=${bs} ga=${ga} gpus=${gpus} ${schedule_desc} save_every=2000 save_final=true ZeRO=1 output=${output}"
if [[ "${1:-}" == --config ]]; then
  shift
  exec python -B scripts/train.py "${args[@]}" "$@" --cfg job --resolve
fi
exec bash scripts/train_zero1.sh "${gpus}" "${args[@]}" "$@"
