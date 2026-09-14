#!/usr/bin/env bash
set -euo pipefail

repo_root="${REPO_ROOT:-/mnt/vepfs02/output/kevin_wang/memorywam/code/fastwam_dynamic_surprise_memory}"
runtime_bin="${RUNTIME_BIN:-/root/kevin_wang/envs/fastwam_guidemem/bin}"
output_dir="${OUTPUT_DIR:-/mnt/vepfs02/output/kevin_wang/memorywam/train/dynamic_surprise_memory_putback_k8_25k_seed42}"
manifest_root="${MANIFEST_ROOT:-/mnt/vepfs02/output/kevin_wang/memorywam/data/putback_dynamic_surprise_segments_v1}"
preflight_only="${PREFLIGHT_ONLY:-0}"
phase_steps="5000 10000 15000 20000 25000 30000 35000 40000"

export REPO_ROOT="${repo_root}" RUNTIME_BIN="${runtime_bin}" OUTPUT_DIR="${output_dir}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export LAYERWISE_K8_25K_OUTPUT="${output_dir}/.boundary_work"
source "${repo_root}/ops/putback_dynamic_surprise_env.sh"

die() { echo "ERROR: $*" >&2; exit 2; }
weight_path() { printf '%s/checkpoints/weights/step_%06d.pt' "${output_dir}" "$1"; }
state_path() { printf '%s/checkpoints/state/step_%06d' "${output_dir}" "$1"; }
boundary_path() {
  if [[ "$1" == "-1" ]]; then
    printf '%s/boundary_init_seed42' "${manifest_root}"
  else
    printf '%s/boundary_step_%06d' "${manifest_root}" "$1"
  fi
}

[[ "${preflight_only}" == "0" || "${preflight_only}" == "1" ]] || die "PREFLIGHT_ONLY must be 0 or 1"
for required in \
  "${repo_root}/scripts/generate_putback_surprise_manifest.py" \
  "${repo_root}/ops/train_putback_dynamic_surprise_phase.sh" \
  "${RMBENCH_PUTBACK_LEROBOT}/meta/info.json" \
  "${MEMORYWAM_PUTBACK_STATS}" \
  "${MEMORYWAM_PUTBACK_CONTINUOUS_LATENTS}/manifest.json" \
  "${FASTWAM_ACTION_DIT_INIT}"; do
  [[ -s "${required}" ]] || die "missing required artifact: ${required}"
done
[[ -d "${MEMORYWAM_PUTBACK_TEXT_CACHE}" ]] || die "missing text cache: ${MEMORYWAM_PUTBACK_TEXT_CACHE}"
IFS=',' read -r -a visible_gpus <<<"${CUDA_VISIBLE_DEVICES}"
[[ ${#visible_gpus[@]} -eq 8 ]] || die "formal run requires exactly eight visible GPUs"

state_is_complete() {
  local state="$1"
  [[ -s "${state}/trainer_state.json" && -s "${state}/scheduler.bin" ]] || return 1
  [[ -d "${state}/pytorch_model_fsdp_0" && -d "${state}/optimizer_0" ]] || return 1
  [[ $(find "${state}/pytorch_model_fsdp_0" -maxdepth 1 -name '*.distcp' -size +0c | wc -l) -ge 8 ]] || return 1
  [[ $(find "${state}/optimizer_0" -maxdepth 1 -name '*.distcp' -size +0c | wc -l) -ge 8 ]] || return 1
  [[ $(find "${state}" -maxdepth 1 -name 'random_states_*.pkl' -size +0c | wc -l) -ge 8 ]] || return 1
}

manifest_is_complete() {
  local manifest="$1" boundary_step="$2"
  [[ -s "${manifest}/manifest.json" ]] || return 1
  "${runtime_bin}/python" - "${manifest}" "${boundary_step}" >/dev/null 2>&1 <<'PY'
import sys
from fastwam.memory.dynamic_surprise import DynamicSurpriseManifestStore
store = DynamicSurpriseManifestStore(
    sys.argv[1], expected_boundary_step=int(sys.argv[2]), expected_episode_count=50
)
for episode in range(50):
    store.segments_for_episode(episode)
PY
}

highest_complete_state=0
if [[ -d "${output_dir}/checkpoints/state" ]]; then
  shopt -s nullglob
  for state in "${output_dir}"/checkpoints/state/step_*; do
    name="$(basename "${state}")"
    [[ "${name}" =~ ^step_([0-9]{6})$ ]] || die "unexpected state directory: ${state}"
    step=$((10#${BASH_REMATCH[1]}))
    (( step % 5000 == 0 && step >= 5000 && step <= 40000 )) || die "unexpected state step: ${state}"
    state_is_complete "${state}" || die "incomplete FSDP state requires inspection: ${state}"
    [[ -s "$(weight_path "${step}")" ]] || die "state exists without portable weight: ${state}"
    (( step > highest_complete_state )) && highest_complete_state=${step}
  done
  shopt -u nullglob
fi
if (( highest_complete_state == 0 )) && [[ -e "${output_dir}" ]]; then
  die "output exists but contains no complete resumable 5k state: ${output_dir}"
fi

generate_manifest() {
  local boundary_step="$1" output="$2" checkpoint="${3:-}" conditioning="${4:-}" conditioning_step="${5:-}"
  if manifest_is_complete "${output}" "${boundary_step}"; then
    echo "manifest_status=complete boundary_step=${boundary_step} path=${output}"
    return 0
  fi
  if [[ -e "${output}" ]]; then
    quarantine="${output}.incomplete.$(date -u +%Y%m%dT%H%M%SZ).$$"
    if [[ "${preflight_only}" == "1" ]]; then
      echo "would_quarantine=${output} destination=${quarantine}"
    else
      mv "${output}" "${quarantine}"
      echo "quarantined=${quarantine}"
    fi
  fi
  if [[ "${boundary_step}" != "-1" ]]; then
    if [[ "${preflight_only}" == "1" && ! -s "${checkpoint}" ]]; then
      echo "would_generate_boundary=${boundary_step} output=${output} checkpoint=${checkpoint} after_training=yes"
      return 0
    fi
    [[ -s "${checkpoint}" ]] || die "missing boundary checkpoint: ${checkpoint}"
    manifest_is_complete "${conditioning}" "${conditioning_step}" || die "invalid conditioning manifest: ${conditioning}"
  fi
  if [[ "${preflight_only}" == "1" ]]; then
    echo "would_generate_boundary=${boundary_step} output=${output} checkpoint=${checkpoint:-initialization}"
    return 0
  fi
  args=(--repo "${repo_root}" --output "${output}" --boundary-step "${boundary_step}" --episode-count 50)
  if [[ "${boundary_step}" == "-1" ]]; then
    args+=(--initialization-only)
  else
    args+=(--checkpoint "${checkpoint}" --conditioning-manifest "${conditioning}" --conditioning-boundary-step "${conditioning_step}")
  fi
  torchrun --standalone --nproc_per_node=8 \
    "${repo_root}/scripts/generate_putback_surprise_manifest.py" "${args[@]}"
  manifest_is_complete "${output}" "${boundary_step}" || die "generated manifest failed validation: ${output}"
}

echo "launcher=dynamic_surprise_putback_k8_40k"
echo "highest_complete_state=${highest_complete_state}"
echo "output_dir=${output_dir}"
echo "manifest_root=${manifest_root}"
echo "gpus=${CUDA_VISIBLE_DEVICES}"

init_manifest="$(boundary_path -1)"
generate_manifest -1 "${init_manifest}"

current_state_step=${highest_complete_state}
for phase_end in ${phase_steps}; do
  if (( phase_end <= highest_complete_state )); then
    [[ -s "$(weight_path "${phase_end}")" ]] || die "completed phase lacks portable weight: ${phase_end}"
    echo "phase_status=complete phase_end=${phase_end} weight=$(weight_path "${phase_end}")"
  else
    (( phase_end == current_state_step + 5000 )) || die "non-contiguous resume plan at phase ${phase_end}"
    boundary_step=$((phase_end - 5000))
    if (( boundary_step == 0 )); then
      phase_manifest="${init_manifest}"
      expected_boundary=-1
    else
      phase_manifest="$(boundary_path "${boundary_step}")"
      expected_boundary=${boundary_step}
    fi
    if [[ "${preflight_only}" == "1" ]]; then
      echo "would_train phase_start=${current_state_step} phase_end=${phase_end} boundary_step=${expected_boundary} manifest=${phase_manifest}"
      current_state_step=${phase_end}
    else
      manifest_is_complete "${phase_manifest}" "${expected_boundary}" || die "phase manifest is incomplete: ${phase_manifest}"
      resume_state=""
      (( current_state_step > 0 )) && resume_state="$(state_path "${current_state_step}")"
      save_steps='[]'
      (( phase_end == 5000 )) && save_steps='[1000]'
      PHASE_END_STEP="${phase_end}" \
      DYNAMIC_SURPRISE_MANIFEST="${phase_manifest}" \
      DYNAMIC_SURPRISE_BOUNDARY_STEP="${expected_boundary}" \
      RESUME_STATE="${resume_state}" \
      SAVE_STEPS="${save_steps}" \
        bash "${repo_root}/ops/train_putback_dynamic_surprise_phase.sh"
      state_is_complete "$(state_path "${phase_end}")" || die "phase did not produce complete state: ${phase_end}"
      [[ -s "$(weight_path "${phase_end}")" ]] || die "phase did not produce portable weight: ${phase_end}"
      current_state_step=${phase_end}
    fi
  fi

  if (( phase_end < 40000 )); then
    if (( phase_end == 5000 )); then
      conditioning_manifest="${init_manifest}"
      conditioning_step=-1
    else
      conditioning_step=$((phase_end - 5000))
      conditioning_manifest="$(boundary_path "${conditioning_step}")"
    fi
    generate_manifest \
      "${phase_end}" \
      "$(boundary_path "${phase_end}")" \
      "$(weight_path "${phase_end}")" \
      "${conditioning_manifest}" \
      "${conditioning_step}"
  fi
done

echo "complete policy_step=40000 boundary_step=35000 policy=$(weight_path 40000) boundary=$(boundary_path 35000)"
