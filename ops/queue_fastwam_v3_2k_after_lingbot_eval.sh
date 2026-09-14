#!/usr/bin/env bash
set -euo pipefail

root=/mnt/vepfs02/output/kevin_wang/memorywam
repo="${root}/code/fastwam_memory_fullkv"
eval_pid_file="${root}/ops/lingbot_official_putback/eval_strict_fullkv_step3000_eight_gpu.pid"
prep_log="${root}/ops/fastwam_fullkv_v3_prepare.log"
train_log="${root}/ops/fastwam_fullkv_v3_2k_train.log"
status_file="${root}/ops/fastwam_fullkv_v3_2k_status.env"

eval_pid="$(cat "${eval_pid_file}")"
{
  echo "state=WAITING_FOR_LINGBOT_EVAL"
  echo "queued_at=$(date -Is)"
  echo "eval_pid=${eval_pid}"
  echo "max_steps=2000"
  echo "save_every=1000"
  echo "world_size=8"
  echo "batch_per_gpu=1"
} >"${status_file}"

while kill -0 "${eval_pid}" 2>/dev/null; do
  sleep 10
done

# The evaluation launcher's EXIT trap terminates all eight model servers. Wait
# until CUDA allocations are actually released before beginning preprocessing.
for _ in $(seq 1 60); do
  used="$(
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits |
      awk '{sum += $1} END {print sum + 0}'
  )"
  if [[ "${used}" -lt 1024 ]]; then
    break
  fi
  sleep 5
done

{
  echo "state=PREPARING_V3_ASSETS"
  echo "prepare_started_at=$(date -Is)"
} >>"${status_file}"

if ! "${repo}/ops/prepare_putback_memorywam_fullkv_v3.sh" >"${prep_log}" 2>&1; then
  {
    echo "state=PREP_FAILED"
    echo "failed_at=$(date -Is)"
  } >>"${status_file}"
  exit 1
fi

{
  echo "state=TRAINING"
  echo "train_started_at=$(date -Is)"
} >>"${status_file}"

if ! "${repo}/ops/train_putback_memorywam_fullkv_2k_v3.sh" >"${train_log}" 2>&1; then
  {
    echo "state=TRAIN_FAILED"
    echo "failed_at=$(date -Is)"
  } >>"${status_file}"
  exit 1
fi

{
  echo "state=SUCCEEDED"
  echo "finished_at=$(date -Is)"
} >>"${status_file}"
