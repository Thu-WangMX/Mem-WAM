#!/usr/bin/env bash
set -euo pipefail

repo_root="${REPO_ROOT:-/mnt/vepfs02/output/kevin_wang/memorywam/code/fastwam_dynamic_surprise_memory}"
runtime_bin="${RUNTIME_BIN:-/root/kevin_wang/envs/fastwam_guidemem/bin}"
output_dir="${OUTPUT_DIR:-/mnt/vepfs02/output/kevin_wang/memorywam/train/dynamic_surprise_memory_putback_k8_25k_seed42}"
manifest_root="${MANIFEST_ROOT:-/mnt/vepfs02/output/kevin_wang/memorywam/data/putback_dynamic_surprise_segments_v1}"
export REPO_ROOT="${repo_root}" RUNTIME_BIN="${runtime_bin}" OUTPUT_DIR="${output_dir}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export LAYERWISE_K8_25K_OUTPUT="${output_dir}/.boundary_work"

# Same immutable PutBack assets and exact seed-42 initialization as the K8 baseline.
source "${repo_root}/ops/putback_dynamic_surprise_env.sh"

for required in \
  "${repo_root}/scripts/generate_putback_surprise_manifest.py" \
  "${RMBENCH_PUTBACK_LEROBOT}/meta/info.json" \
  "${MEMORYWAM_PUTBACK_STATS}" \
  "${MEMORYWAM_PUTBACK_CONTINUOUS_LATENTS}/manifest.json" \
  "${FASTWAM_ACTION_DIT_INIT}"; do
  [[ -s "${required}" ]] || { echo "ERROR: missing required artifact: ${required}" >&2; exit 2; }
done
[[ -d "${MEMORYWAM_PUTBACK_TEXT_CACHE}" ]] || { echo "ERROR: missing text cache" >&2; exit 2; }
IFS=',' read -r -a visible_gpus <<<"${CUDA_VISIBLE_DEVICES}"
[[ ${#visible_gpus[@]} -eq 8 ]] || { echo "ERROR: formal run requires eight GPUs" >&2; exit 2; }

generate_manifest() {
  local boundary_step="$1" output="$2" checkpoint="${3:-}" conditioning="${4:-}" conditioning_step="${5:-}"
  [[ ! -e "${output}" ]] || return 0
  args=(--repo "${repo_root}" --output "${output}" --boundary-step "${boundary_step}" --episode-count 50)
  if [[ "${boundary_step}" == "-1" ]]; then
    args+=(--initialization-only)
  else
    args+=(--checkpoint "${checkpoint}" --conditioning-manifest "${conditioning}" --conditioning-boundary-step "${conditioning_step}")
  fi
  torchrun --standalone --nproc_per_node=8 "${repo_root}/scripts/generate_putback_surprise_manifest.py" "${args[@]}"
}

init_manifest="${manifest_root}/boundary_init_seed42"
generate_manifest -1 "${init_manifest}"

previous_manifest="${init_manifest}"
previous_boundary=-1
for phase_end in 5000 10000 15000 20000 25000; do
  if [[ "${phase_end}" == "5000" ]]; then
    resume_state=""
    save_steps='[1000]'
  else
    previous_phase=$((phase_end - 5000))
    resume_state="${output_dir}/training_state/step_$(printf '%06d' "${previous_phase}")"
    save_steps='[]'
  fi
  PHASE_END_STEP="${phase_end}" \
  DYNAMIC_SURPRISE_MANIFEST="${previous_manifest}" \
  DYNAMIC_SURPRISE_BOUNDARY_STEP="${previous_boundary}" \
  RESUME_STATE="${resume_state}" \
  SAVE_STEPS="${save_steps}" \
    bash "${repo_root}/ops/train_putback_dynamic_surprise_phase.sh"

  if [[ "${phase_end}" != "25000" ]]; then
    next_manifest="${manifest_root}/boundary_step_$(printf '%06d' "${phase_end}")"
    checkpoint="${output_dir}/weights/step_$(printf '%06d' "${phase_end}").pt"
    generate_manifest "${phase_end}" "${next_manifest}" "${checkpoint}" "${previous_manifest}" "${previous_boundary}"
    previous_manifest="${next_manifest}"
    previous_boundary="${phase_end}"
  fi
done

echo "complete policy=${output_dir}/weights/step_025000.pt boundary=${manifest_root}/boundary_step_020000"
