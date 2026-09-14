#!/usr/bin/env bash
set -u
set -o pipefail

repo_root="${REPO_ROOT:-/mnt/vepfs02/output/kevin_wang/memorywam/code/fastwam_memory_fullkv}"
python_bin="${PYTHON_BIN:-/root/kevin_wang/envs/fastwam_guidemem/bin/python}"
run_dir="${RUN_DIR:-/mnt/vepfs02/output/kevin_wang/memorywam/train/fastwam_fullkv_putback_rgb_v3_resume2k_extra3k_total5k_seed42_20260801}"
diagnostics="${DIAGNOSTICS_DIR:-/mnt/vepfs02/output/kevin_wang/memorywam/diagnostics/resume2k_extra3k_20260801}"
training_pid="${TRAINING_PID:-95160}"

export PATH="$(dirname "${python_bin}"):${PATH}"
export PYTHONPATH="${repo_root}/src:${repo_root}${PYTHONPATH:+:${PYTHONPATH}}"
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-/mnt/vepfs02/datasets/kevin_wang/code/projects/FastWAM/checkpoints}"
export DIFFSYNTH_SKIP_DOWNLOAD=true
export RMBENCH_PUTBACK_LEROBOT="${RMBENCH_PUTBACK_LEROBOT:-/mnt/vepfs02/output/kevin_wang/memorywam/data/rmbench_lerobot_v21_rgb_v3/put_back_block}"
export FASTWAM_PUTBACK_STATS="${FASTWAM_PUTBACK_STATS:-/mnt/vepfs02/output/kevin_wang/memorywam/data/fastwam_putback_stats_rgb_v3/dataset_stats.json}"
export FASTWAM_FULL_KV_LATENTS="${FASTWAM_FULL_KV_LATENTS:-/mnt/vepfs02/output/kevin_wang/memorywam/data/fastwam_putback_temporal_fullkv_rgb_v3_stride16}"
export FASTWAM_PUTBACK_TEXT_CACHE="${FASTWAM_PUTBACK_TEXT_CACHE:-/mnt/vepfs02/output/kevin_wang/memorywam/data/fastwam_putback_text_cache_rgb_v3}"
export FASTWAM_ACTION_DIT_INIT="${FASTWAM_ACTION_DIT_INIT:-/mnt/vepfs02/output/kevin_wang/memorywam/model_assets/fastwam_fullkv_official_init/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt}"
export PYTHONUTF8=1
export LANG=C.UTF-8
export LC_ALL=C.UTF-8

mkdir -p "${diagnostics}"
cd "${repo_root}" || exit 1

wait_for_checkpoint() {
  local checkpoint="$1"
  local previous_size=0
  while true; do
    local current_size=0
    if [[ -f "${checkpoint}" ]]; then
      current_size="$(stat -c %s "${checkpoint}" 2>/dev/null || echo 0)"
    fi
    if [[ "${current_size}" -gt 10000000000 && "${current_size}" -eq "${previous_size}" ]]; then
      return 0
    fi
    previous_size="${current_size}"
    if ! kill -0 "${training_pid}" 2>/dev/null && [[ "${current_size}" -le 10000000000 ]]; then
      echo "Training exited before checkpoint became ready: ${checkpoint}" >&2
      return 1
    fi
    sleep 15
  done
}

run_open_loop() {
  local checkpoint="$1"
  local total_label="$2"
  local steps="$3"
  local output="${diagnostics}/openloop_${total_label}_n${steps}.json"
  echo "Starting open-loop evaluation: label=${total_label} steps=${steps} checkpoint=${checkpoint}"
  CUDA_VISIBLE_DEVICES=7 "${python_bin}" -u scripts/eval_full_kv_open_loop.py \
    --checkpoints "${checkpoint}" \
    --stats "${FASTWAM_PUTBACK_STATS}" \
    --output "${output}" \
    --device cuda:0 \
    --max-samples 4 \
    --num-inference-steps "${steps}" \
    --seed 0 \
    >"${diagnostics}/openloop_${total_label}_n${steps}.log" 2>&1
}

relative_steps=(1000 2000 3000)
total_labels=(total3k total4k total5k)
for index in 0 1 2; do
  checkpoint="${run_dir}/checkpoints/weights/step_$(printf '%06d' "${relative_steps[$index]}").pt"
  if wait_for_checkpoint "${checkpoint}"; then
    run_open_loop "${checkpoint}" "${total_labels[$index]}" 10 || true
  fi
done

while kill -0 "${training_pid}" 2>/dev/null; do
  sleep 15
done

final_checkpoint="${run_dir}/checkpoints/weights/step_003000.pt"
if [[ -s "${final_checkpoint}" ]]; then
  run_open_loop "${final_checkpoint}" total5k 50 || true
  run_tag="fastwam_fullkv_resume2k_extra3k_total5k_20260801"
  CHECKPOINT="${final_checkpoint}" \
  RUN_TAG="${run_tag}" \
  LOG_ROOT="${diagnostics}/closed_loop_five_seed" \
  DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH}" \
    bash ops/eval_putback_five_seed.sh \
    >"${diagnostics}/closed_loop_five_seed.launcher.log" 2>&1 || true
else
  echo "Final checkpoint is missing: ${final_checkpoint}" >&2
  exit 1
fi
